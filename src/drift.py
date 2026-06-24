import os
import pandas as pd
import numpy as np
import logging
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from src.config import DRIFT_THRESHOLD
from src.database import fetch_data_from_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

def check_for_drift() -> tuple[bool, float]:
    """
    Compares historical baseline data (reference) against a new batch of data (current).
    Uses Evidently AI to generate a drift dashboard and detect dataset drift.
    Returns (drift_detected, share_drifted_features).
    """
    try:
        df = fetch_data_from_db("loans")
    except Exception as e:
        logging.error(f"Error fetching data from DB for drift check: {e}")
        return False, 0.0

    if len(df) < 100:
        logging.warning("Not enough data to check for drift. Need at least 100 rows.")
        return False, 0.0
        
    # Load last_train_db_size from pipeline_metadata.json
    import json
    metadata_path = "reports/pipeline_metadata.json"
    last_train_db_size = 50000
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                last_train_db_size = metadata.get("last_train_db_size", 50000)
        except Exception as err:
            logging.warning(f"Error reading pipeline metadata: {err}")

    # If database size is less than or equal to training size, no new data exists
    if len(df) <= last_train_db_size:
        logging.info(f"No new data since last retraining run (size: {len(df)}). Generating 0.00% baseline drift report.")
        # Self-comparison of shuffled database segments
        shuffled = df.sample(frac=1.0, random_state=42).copy()
        split_idx = len(shuffled) // 2
        reference_data = shuffled.iloc[:split_idx].copy()
        current_data = shuffled.iloc[split_idx:].copy()
    else:
        # Compare reference (data up to last_train_db_size) vs current (new data added since last_train_db_size)
        reference_data = df.iloc[:last_train_db_size].copy()
        current_data = df.iloc[last_train_db_size:].copy()
    
    # Drop columns that are not predictive features
    columns_to_drop = ['sk_id_curr', 'created_at']
    for col in columns_to_drop:
        if col in reference_data.columns:
            reference_data = reference_data.drop(columns=[col])
        if col in current_data.columns:
            current_data = current_data.drop(columns=[col])
            
    logging.info(f"Comparing Reference ({len(reference_data)} rows) and Current ({len(current_data)} rows) datasets...")
    
    # Setup Evidently Report
    report = Report(metrics=[
        DataDriftPreset(drift_share=0.35),
        TargetDriftPreset()
    ])
    
    try:
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        report_html_path = "reports/data_drift_report.html"
        report.save_html(report_html_path)
        logging.info(f"Evidently drift dashboard saved to {report_html_path}")
        
        # Extract drift metrics
        report_dict = report.as_dict()
        metrics_results = report_dict.get("metrics", [])
        
        drift_detected = False
        share_drifted_features = 0.0
        for metric in metrics_results:
            metric_type = metric.get("metric")
            metric_result = metric.get("result", {})
            
            if "DatasetDriftMetric" in metric_type or "dataset_drift" in metric_result:
                drift_detected = metric_result.get("dataset_drift", False)
                share_drifted_features = metric_result.get("share_of_drifted_columns", 0.0)
                logging.info(f"Dataset Drift Detected: {drift_detected} (Drifted Feature Share: {share_drifted_features:.2%}, Threshold: 35%)")
                break
        
        # Special override: if database has not received new rows, force drift to False and 0.0%
        if len(df) <= last_train_db_size:
            drift_detected = False
            share_drifted_features = 0.0

        return drift_detected, share_drifted_features

    except Exception as e:
        logging.error(f"Error running Evidently drift analysis: {e}")
        return False, 0.0

if __name__ == "__main__":
    drift, share = check_for_drift()
    print(f"Drift detected: {drift} (Share: {share:.2%})")
