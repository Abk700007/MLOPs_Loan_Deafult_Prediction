import os
import sys
import pandas as pd
import numpy as np
import logging
import io
import contextlib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
import xgboost as xgb
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

# Reconfigure stdout/stderr on Windows to prevent Unicode/charmap encoding crashes when writing emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(errors='backslashreplace')
        sys.stderr.reconfigure(errors='backslashreplace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set MLflow environment variables if configured
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Loan Default Prediction Service",
    description="API and Dashboard to predict borrower defaults and manage retraining pipelines.",
    version="1.0.0"
)

# Global model variable
model = None

class LoanApplication(BaseModel):
    code_gender: str = Field(description="Gender of the client (M, F, XNA)")
    flag_own_car: str = Field(description="Does the client own a car? (Y, N)")
    flag_own_realty: str = Field(description="Does the client own real estate? (Y, N)")
    cnt_children: int = Field(default=0, ge=0, description="Number of children")
    amt_income_total: float = Field(ge=0.0, description="Total income of client")
    amt_credit: float = Field(ge=0.0, description="Credit amount of the loan")
    amt_annuity: float = Field(ge=0.0, description="Loan annuity")
    amt_goods_price: float = Field(ge=0.0, description="Price of the goods for which the loan is given")
    days_birth: int = Field(le=0, description="Age in days (negative number, e.g. -12000)")
    days_employed: int = Field(description="Days employed (negative number, or 365243 for retired/unemployed)")
    ext_source_2: float = Field(ge=0.0, le=1.0, description="Score from external data source 2")
    ext_source_3: float = Field(default=0.5, ge=0.0, le=1.0, description="Score from external data source 3")

    class Config:
        json_schema_extra = {
            "example": {
                "code_gender": "F",
                "flag_own_car": "N",
                "flag_own_realty": "Y",
                "cnt_children": 0,
                "amt_income_total": 135000.0,
                "amt_credit": 312682.0,
                "amt_annuity": 15000.0,
                "amt_goods_price": 297000.0,
                "days_birth": -15000,
                "days_employed": -2500,
                "ext_source_2": 0.65,
                "ext_source_3": 0.52
            }
        }

class FallbackModelWrapper:
    """Wrapper to mimic the MLflow PyFuncModel structure for the local fallback model."""
    def __init__(self, raw_model):
        self.raw = raw_model
        
    class _ModelImplWrapper:
        def __init__(self, raw_model):
            self.raw_model = raw_model
        def get_raw_model(self):
            return self.raw_model
            
    @property
    def _model_impl(self):
        return self._ModelImplWrapper(self.raw)

@app.on_event("startup")
def load_registered_model():
    """Loads the model on startup. Tries DagsHub registry first, then falls back to local model if offline/registry fails."""
    global model
    logging.info("Starting up API and attempting to load model...")
    local_fallback_path = "models/fallback_model.json"

    # --- Priority 1: Try DagsHub MLflow Registry (requires network) ---
    logging.info("Attempting to load from MLflow registry...")
    for stage in ["Production", "Staging", "1"]:
        uri_suffix = stage if stage == "1" else stage
        model_uri = f"models:/{MODEL_NAME}/{uri_suffix}"
        try:
            logging.info(f"Trying model URI: {model_uri}")
            loaded = mlflow.pyfunc.load_model(model_uri)

            # Try to unwrap to raw XGBoost model and save as local fallback
            try:
                raw_model = loaded.unwrap_python_model()
            except Exception:
                try:
                    raw_model = loaded._model_impl.python_model if hasattr(loaded._model_impl, "python_model") else None
                except Exception:
                    raw_model = None

            if raw_model is None:
                # For xgboost flavour, access via internal booster
                try:
                    booster = loaded._model_impl.xgb_model if hasattr(loaded._model_impl, "xgb_model") else None
                    if booster is not None:
                        os.makedirs("models", exist_ok=True)
                        booster.save_model(local_fallback_path)
                        logging.info(f"Saved XGBoost booster to {local_fallback_path} for offline fallback.")
                        # Reload as FallbackModelWrapper for consistent predict interface
                        xgb_cls = type(booster)
                        new_raw = xgb.XGBClassifier()
                        new_raw.load_model(local_fallback_path)
                        model = FallbackModelWrapper(new_raw)
                        logging.info(f"Model wrapped and ready from stage '{stage}'.")
                        return
                except Exception as booster_err:
                    logging.warning(f"Could not extract booster for caching: {booster_err}")

            # Use the pyfunc model directly
            model = loaded
            logging.info(f"Model loaded from registry stage '{stage}'.")

            # Attempt to save fallback copy for next startup
            try:
                os.makedirs("models", exist_ok=True)
                inner = getattr(loaded, "_model_impl", None)
                if inner:
                    xgb_model = getattr(inner, "xgb_model", None) or getattr(inner, "_xgb_model", None)
                    if xgb_model:
                        xgb_model.save_model(local_fallback_path)
                        logging.info(f"Cached model to {local_fallback_path} for next startup.")
            except Exception as cache_err:
                logging.warning(f"Could not cache model locally: {cache_err}")
            return
        except Exception as e:
            logging.warning(f"Could not load from '{model_uri}': {e}")

    # --- Priority 2: Load from local fallback if MLflow loading failed ---
    logging.info("MLflow registry load failed. Attempting local fallback model...")
    if os.path.exists(local_fallback_path):
        try:
            import xgboost as xgb
            raw_model = xgb.XGBClassifier()
            raw_model.load_model(local_fallback_path)
            model = FallbackModelWrapper(raw_model)
            logging.info(f"Model loaded successfully from local fallback: {local_fallback_path}")
            return
        except Exception as e_local:
            logging.error(f"Local fallback model failed to load: {e_local}")

    logging.error("All model load attempts failed. Prediction endpoint will return 503.")
    model = None

@app.get("/health")
def health_check():
    """Endpoint to check health of service and if the model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME
    }

def preprocess_application(app_data: LoanApplication) -> pd.DataFrame:
    """Preprocesses Pydantic data class into the exact shape, encoding, and engineered features needed by XGBoost."""
    from src.train import engineer_features
    
    gender_map = {'F': 0, 'M': 1, 'XNA': 2}
    car_map = {'N': 0, 'Y': 1}
    realty_map = {'N': 0, 'Y': 1}
    
    data_dict = {
        "code_gender": gender_map.get(app_data.code_gender.upper(), 0),
        "flag_own_car": car_map.get(app_data.flag_own_car.upper(), 0),
        "flag_own_realty": realty_map.get(app_data.flag_own_realty.upper(), 0),
        "cnt_children": app_data.cnt_children,
        "amt_income_total": app_data.amt_income_total,
        "amt_credit": app_data.amt_credit,
        "amt_annuity": app_data.amt_annuity,
        "amt_goods_price": app_data.amt_goods_price,
        "days_birth": app_data.days_birth,
        "days_employed": app_data.days_employed,
        "ext_source_2": app_data.ext_source_2,
        "ext_source_3": app_data.ext_source_3
    }
    
    df_base = pd.DataFrame([data_dict])
    df_feat = engineer_features(df_base)
    
    # Order columns to match the training features exactly
    feature_order = [
        "code_gender", "flag_own_car", "flag_own_realty", "cnt_children",
        "amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
        "days_birth", "days_employed", "ext_source_2", "ext_source_3",
        "annuity_income_ratio", "credit_income_ratio", "goods_credit_ratio",
        "age_years", "emp_age_ratio", "ext_source_mean", "ext_source_prod"
    ]
    
    return df_feat[feature_order]

@app.post("/predict")
def predict_default(application: LoanApplication):
    """Receives loan application data and returns calibrated probability of default."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently not loaded or registered on MLflow. Please check logs.")
        
    try:
        features_df = preprocess_application(application)
        raw_model = model._model_impl.get_raw_model()
        
        # 1. Get raw weighted probability from XGBoost
        prob_w = float(raw_model.predict_proba(features_df)[0][1])
        
        # 2. Load optimal threshold from metadata if available (fallback to 0.5)
        optimal_threshold = 0.5
        metadata_path = "reports/pipeline_metadata.json"
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    optimal_threshold = metadata.get("optimal_threshold", 0.5)
            except Exception:
                pass
                
        # 3. Get calibration weight from database or fallback to baseline
        w = 11.419  # Default baseline scale_pos_weight
        try:
            from src.database import get_db_summary
            total, defaults = get_db_summary("loans")
            if total > 0 and defaults > 0:
                w = (total - defaults) / defaults
        except Exception:
            pass
            
        # Calibrate predicted probability: p = prob_w / (prob_w + w * (1.0 - prob_w) + 1e-9)
        calibrated_prob = prob_w / (prob_w + w * (1.0 - prob_w) + 1e-9)
        calibrated_prob = min(max(calibrated_prob, 0.0), 1.0)
        
        # 4. Make prediction based on optimal threshold
        prediction = 1 if prob_w >= optimal_threshold else 0
        
        return {
            "default_probability": float(calibrated_prob),
            "default_prediction": prediction,
            "risk_status": "High Risk" if prob_w >= optimal_threshold else "Low Risk",
            "optimal_threshold": optimal_threshold
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# =====================================================================
# FULLSTACK INTERACTIVE DASHBOARD & MLOPS WIDGETS
# =====================================================================

@app.get("/db-status")
def db_status():
    """Returns basic counts and default statistics from Supabase."""
    try:
        from src.database import get_db_summary
        total, defaults = get_db_summary("loans")
        rate = (defaults / total) if total > 0 else 0
        
        report_exists = os.path.exists("reports/data_drift_report.html")
        
        global model
        model_state = "Loaded Successfully" if model is not None else "Not Registered/None"
        
        return {
            "total_records": total,
            "defaults": defaults,
            "default_rate": f"{rate:.2%}",
            "model_status": model_state,
            "drift_report_available": report_exists
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/simulate-drift")
def simulate_drift():
    """Infects the database with 1,000 highly shifted/drifted records to test auto-retraining."""
    try:
        from src.database import fetch_data_from_db, save_data_to_db
        df = fetch_data_from_db("loans")
        if len(df) < 500:
            raise HTTPException(status_code=400, detail="Seed database first (min 500 rows required).")
            
        # Sample 1000 records to alter
        drift_batch = df.sample(n=min(len(df), 1000), random_state=42).copy()
        
        # Modify Primary Keys to prevent collision
        max_id = df['sk_id_curr'].max()
        drift_batch['sk_id_curr'] = range(max_id + 1, max_id + 1 + len(drift_batch))
        
        # Induce massive drift on predictive variables
        drift_batch['ext_source_2'] = drift_batch['ext_source_2'] * 0.05  # Severe credit degradation
        drift_batch['ext_source_3'] = drift_batch['ext_source_3'] * 0.05
        drift_batch['amt_income_total'] = drift_batch['amt_income_total'] * 4.0  # Hyperinflation
        drift_batch['amt_credit'] = drift_batch['amt_credit'] * 3.0
        drift_batch['target'] = 1  # 100% defaults
        
        # Remove timestamp columns if present
        if 'created_at' in drift_batch.columns:
            drift_batch = drift_batch.drop(columns=['created_at'])
            
        save_data_to_db(drift_batch)
        total_records = len(df) + len(drift_batch)
        return {
            "success": True,
            "records_added": len(drift_batch),
            "total_records": total_records,
            "message": "Injected 1,000 drifted borrower records into Supabase."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Global MLOps pipeline execution status dictionary for real-time dashboard polling
pipeline_status = {
    "status": "idle",
    "step": "none",
    "validation_passed": None,
    "drift_detected": None,
    "drift_share": 0.0,
    "retrained": None,
    "run_id": None,
    "new_version": None,
    "error": None,
    "logs": ""
}

def run_retraining_pipeline_task():
    """Background task to run the ML retraining pipeline end-to-end with status tracking and persistent file logging."""
    global pipeline_status
    
    # Silence verbose logs to keep emulator log clean
    logging.getLogger("great_expectations").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    def log_and_write(msg):
        global pipeline_status
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} - INFO - {msg}"
        logging.info(msg)
        # Write to in-memory log queue
        pipeline_status["logs"] += f"{log_line}\n"
        # Write to persistent file audit log
        try:
            os.makedirs("reports", exist_ok=True)
            with open("reports/pipeline_execution.log", "a", encoding="utf-8") as f:
                f.write(f"{log_line}\n")
        except Exception:
            pass

    try:
        from src.database import fetch_data_from_db
        from src.validation import validate_data
        from src.drift import check_for_drift
        from src.train import train_model
        
        log_and_write("[MLOps] Initializing background pipeline trigger...")
        pipeline_status["step"] = "ingest"
        
        # 1. Ingestion / Local batch ingestion
        csv_path = "data/application_train.csv"
        if os.path.exists(csv_path):
            log_and_write("[MLOps] Ingesting new batch from local Kaggle dataset...")
            from src.ingestion import ingest_data
            ingest_data(limit_records=1000)
        else:
            log_and_write("[MLOps] Local CSV 'data/application_train.csv' not found. Skipping chunk ingestion.")
            log_and_write("[MLOps] Processing directly with current Supabase records...")
            
        # 2. Validation
        pipeline_status["step"] = "validate"
        df = fetch_data_from_db("loans")
        latest_batch = df.tail(1000)
        log_and_write(f"[MLOps] Validating latest batch of {len(latest_batch)} records...")
        is_valid = validate_data(latest_batch)
        log_and_write(f"[MLOps] Validation result: {is_valid}")
        pipeline_status["validation_passed"] = is_valid
        
        if not is_valid:
            log_and_write("[MLOps] Validation failed. Retraining aborted.")
            pipeline_status["status"] = "failed"
            pipeline_status["error"] = "Data validation failed."
            return
            
        # 3. Drift Check
        pipeline_status["step"] = "drift"
        log_and_write("[MLOps] Evaluating dataset drift using Evidently AI...")
        drift, drift_share = check_for_drift()
        log_and_write(f"[MLOps] Drift check completed. Drift detected: {drift} (Drift Share: {drift_share:.2%})")
        pipeline_status["drift_detected"] = drift
        pipeline_status["drift_share"] = float(drift_share)
        
        # 4. Retraining & Hot-Reload
        run_id = None
        new_version = None
        if drift:
            pipeline_status["step"] = "retrain"
            log_and_write("[MLOps] Concept drift detected. Retraining model...")
            run_id = train_model()
            log_and_write(f"[MLOps] Model retrained successfully. Registered to DagsHub. Run ID: {run_id}")
            
            pipeline_status["step"] = "reload"
            log_and_write("[MLOps] Reloading active model in serving layer...")
            load_registered_model()
            
            pipeline_status["retrained"] = True
            pipeline_status["run_id"] = run_id
            
            # Resolve version number
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                latest_versions = client.get_latest_versions(MODEL_NAME)
                for mv in latest_versions:
                    if mv.run_id == run_id:
                        new_version = mv.version
                        break
                pipeline_status["new_version"] = new_version
            except Exception as e:
                log_and_write(f"[WARNING] Could not resolve new version number: {e}")
        else:
            log_and_write("[MLOps] Drift is below threshold. Retraining skipped. Production model retained.")
            pipeline_status["retrained"] = False
            
        log_and_write("[MLOps] Pipeline completed successfully.")
        pipeline_status["step"] = "reload" # Marked step complete
        pipeline_status["status"] = "success"
        
    except Exception as e:
        log_err = f"[MLOps] Pipeline execution crashed: {e}"
        logging.error(log_err)
        pipeline_status["logs"] += f"{log_err}\n"
        pipeline_status["status"] = "failed"
        pipeline_status["error"] = str(e)

@app.post("/trigger-retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Triggers the retraining pipeline in the background using FastAPI BackgroundTasks."""
    global pipeline_status
    if pipeline_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Retraining pipeline is already executing.")
        
    # Reset status fields
    pipeline_status = {
        "status": "running",
        "step": "ingest",
        "validation_passed": None,
        "drift_detected": None,
        "drift_share": 0.0,
        "retrained": None,
        "run_id": None,
        "new_version": None,
        "error": None,
        "logs": ">>> Initializing MLOps retraining pipeline...\n"
    }
    
    # Enqueue task
    background_tasks.add_task(run_retraining_pipeline_task)
    return {"success": True, "message": "Pipeline triggered asynchronously in background."}

@app.get("/pipeline-status")
def get_pipeline_status():
    """Returns the current state and execution logs of the background retraining task."""
    global pipeline_status
    return pipeline_status

@app.post("/reset-db")
def reset_db():
    """Resets the Supabase loans database to the original 50,000 clean records."""
    try:
        from src.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Delete all simulated/drifted records
        cursor.execute("DELETE FROM loans WHERE sk_id_curr > 157876")
        
        # Force PostgreSQL to update statistics so the Supabase UI row count updates instantly
        cursor.execute("ANALYZE loans")
        conn.commit()
        cursor.close()
        conn.close()
        
        # Reset metadata baseline training size to 50000
        import json
        metadata_path = "reports/pipeline_metadata.json"
        metadata = {"last_train_db_size": 50000}
        os.makedirs("reports", exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
            
        # Regenerate the Evidently HTML report using the clean database
        from src.drift import check_for_drift
        check_for_drift()
            
        logging.info("Database reset: kept first 50,000 records, updated stats, and generated clean drift report.")
        return {"success": True, "message": "Database successfully reset to original 50,000 clean records."}
    except Exception as e:
        logging.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift-report")
def get_drift_report():
    """Serves the Evidently AI Data Drift HTML report."""
    report_path = "reports/data_drift_report.html"
    if os.path.exists(report_path):
        return FileResponse(report_path)
    return HTMLResponse("<h3>No drift report generated yet. Run the retraining pipeline first.</h3>")

@app.get("/model-metrics")
def get_model_metrics():
    """Queries MLflow for the active registered model's metrics."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Resolve latest version
        latest_versions = client.get_latest_versions(MODEL_NAME)
        if not latest_versions:
            raise ValueError("No models registered")
            
        prod_version = None
        for mv in latest_versions:
            if mv.current_stage == "Production":
                prod_version = mv
                break
        if not prod_version:
            for mv in latest_versions:
                if mv.current_stage == "Staging":
                    prod_version = mv
