import sys
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
from src.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME
from src.database import fetch_data_from_db

# Reconfigure stdout/stderr on Windows to prevent Unicode/charmap encoding crashes when writing emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends custom financial risk ratios and engineered features.
    """
    df_feat = df.copy()
    
    # 1. Debt-to-Income (DTI) Ratio
    df_feat['annuity_income_ratio'] = df_feat['amt_annuity'] / (df_feat['amt_income_total'] + 1e-5)
    
    # 2. Credit-to-Income Ratio
    df_feat['credit_income_ratio'] = df_feat['amt_credit'] / (df_feat['amt_income_total'] + 1e-5)
    
    # 3. Goods Price to Credit Ratio
    df_feat['goods_credit_ratio'] = df_feat['amt_goods_price'] / (df_feat['amt_credit'] + 1e-5)
    
    # 4. Age in Years
    if 'days_birth' in df_feat.columns:
        df_feat['age_years'] = -df_feat['days_birth'] / 365.25
        
    # 5. Employment to Age Ratio
    if 'days_employed' in df_feat.columns and 'days_birth' in df_feat.columns:
        df_feat['emp_age_ratio'] = df_feat.apply(
            lambda row: -row['days_employed'] / -row['days_birth'] if row['days_employed'] < 0 else 0.0,
            axis=1
        )
        
    # 6. Combined External Source Risk Score
    ext_cols = [c for c in ['ext_source_2', 'ext_source_3'] if c in df_feat.columns]
    if len(ext_cols) > 0:
        df_feat['ext_source_mean'] = df_feat[ext_cols].mean(axis=1)
        if len(ext_cols) == 2:
            df_feat['ext_source_prod'] = df_feat['ext_source_2'] * df_feat['ext_source_3']
            
    return df_feat

def preprocess_data(df: pd.DataFrame):
    """
    Cleans, engineers features, and prepares data for XGBoost.
    Handles encoding of categorical variables.
    """
    logging.info("Preprocessing data...")
    
    # Copy and engineer features
    data = engineer_features(df)
    
    # Drop timestamp column and client ID from features
    if 'created_at' in data.columns:
        data = data.drop(columns=['created_at'])
    if 'sk_id_curr' in data.columns:
        data = data.drop(columns=['sk_id_curr'])
        
    # Split features and target
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Handle categorical variables (LabelEncoding is fine for trees)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Convert to string and handle nulls
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    return X, y, label_encoders

def train_model():
    """Fetches data, preprocesses it, trains an XGBoost classifier, and logs to MLflow."""
    # 1. Fetch data from Supabase database
    try:
        df = fetch_data_from_db("loans")
    except Exception as e:
        logging.error(f"Failed to fetch data for training: {e}")
        return None
        
    if len(df) < 50:
        logging.warning("Not enough records in the database to train a model. Need at least 50 records.")
        return None
        
    # 2. Preprocess features
    X, y, encoders = preprocess_data(df)
    
    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Initialize MLflow Experiment
    if MLFLOW_TRACKING_URI:
        logging.info("Setting MLflow tracking URI from configuration...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Calculate scale_pos_weight dynamically to balance recall and precision
    scale_pos_weight_value = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    logging.info(f"Calculated scale_pos_weight dynamically: {scale_pos_weight_value:.4f}")
    
    # XGBoost Hyperparameters
    params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight_value,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    
    # Enforce monotonic constraints mapped to feature names
    constraints = {}
    for col in X_train.columns:
        if col in ['ext_source_2', 'ext_source_3', 'ext_source_mean', 'ext_source_prod']:
            constraints[col] = -1
        elif col in ['annuity_income_ratio', 'credit_income_ratio']:
            constraints[col] = 1
            
    logging.info("Starting model training run...")
    with mlflow.start_run() as run:
        # Train model
        model = xgb.XGBClassifier(**params, monotone_constraints=constraints)
        model.fit(X_train, y_train)
        
        # Predict & evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Post-training threshold optimization to maximize F1-score
        best_threshold = 0.5
        best_f1 = 0.0
        for th in np.linspace(0.1, 0.9, 81):
            y_pred_temp = (y_prob >= th).astype(int)
            f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_threshold = th
                
        logging.info(f"Optimal threshold found: {best_threshold:.3f} (F1-score: {best_f1:.4f})")
        y_pred_opt = (y_prob >= best_threshold).astype(int)
        
        # Calculate metrics using optimal threshold
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred_opt),
            "precision": precision_score(y_test, y_pred_opt, zero_division=0),
            "recall": recall_score(y_test, y_pred_opt, zero_division=0),
            "f1": best_f1
        }
        
        logging.info(f"Metrics at optimal threshold {best_threshold:.3f}: {metrics}")

        # Generate and save evaluation curve data for Chart.js dashboard
        try:
            import json
            import os
            from sklearn.metrics import roc_curve, precision_recall_curve
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            
            # Downsample to exactly 21 points for clean rendering in the UI
            indices_roc = np.linspace(0, len(fpr) - 1, 21, dtype=int)
            indices_pr = np.linspace(0, len(precision) - 1, 21, dtype=int)
            
            curves_data = {
                "roc": {
                    "fpr": [float(fpr[i]) for i in indices_roc],
                    "tpr": [float(tpr[i]) for i in indices_roc]
                },
                "pr": {
                    "recall": [float(recall[i]) for i in indices_pr],
                    "precision": [float(precision[i]) for i in indices_pr]
                }
            }
            
            os.makedirs("reports", exist_ok=True)
            with open("reports/evaluation_curves.json", "w") as f:
                json.dump(curves_data, f)
            logging.info("Saved ROC and Precision-Recall curve data to reports/evaluation_curves.json")
        except Exception as curve_err:
            logging.error(f"Failed to generate evaluation curve data: {curve_err}")
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Log the preprocessing info as an artifact
        mlflow.log_dict({"categorical_encoders": list(encoders.keys())}, "preprocessing_info.json")
        
        # Log and Register the Model in MLflow Model Registry
        logging.info(f"Logging and registering model under name '{MODEL_NAME}'...")
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        # Automatically transition the newly registered version to Production stage
        try:
            client = mlflow.tracking.MlflowClient()
            # Fetch versions in 'None' stage (newly registered)
            latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
            if latest_versions:
                # Match current run_id to locate the correct version
                target_version = None
                for mv in latest_versions:
                    if mv.run_id == run.info.run_id:
                        target_version = mv.version
                        break
                if not target_version:
                    target_version = latest_versions[0].version
                
                logging.info(f"Automatically promoting model version {target_version} to 'Production' stage...")
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=target_version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logging.info("Model version promoted to Production stage successfully.")
        except Exception as promo_err:
            logging.warning(f"Could not automatically promote model version to Production: {promo_err}")
            
