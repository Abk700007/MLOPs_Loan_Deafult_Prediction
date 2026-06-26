from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# Configure logging
logging.basicConfig(level=logging.INFO)

# Python execution functions wrapping our core modules
def run_ingestion():
    from src.ingestion import ingest_data
    logging.info("Starting data ingestion task...")
    ingest_data(batch_size=500)  # Seed/fetch a new batch of 500 records

def run_validation():
    from src.database import fetch_data_from_db
    from src.validation import validate_data
    
    logging.info("Fetching data from DB for validation...")
    df = fetch_data_from_db("loans")
    
    # We validate the latest batch (e.g. the last 500 rows ingested)
    latest_data = df.tail(500)
    success = validate_data(latest_data)
    
    if not success:
        raise ValueError("Data validation failed! Aborting pipeline run to prevent corrupted model training.")

def run_drift_check(ti):
    from src.drift import check_for_drift
    logging.info("Starting data drift detection task...")
    drift_detected, _ = check_for_drift()
    
    # Push drift status to Airflow XCom for branching decisions
    ti.xcom_push(key="drift_detected", value=drift_detected)
    logging.info(f"Drift check completed. Drift detected: {drift_detected}")

def check_branch(ti):
    # Pull drift status from the drift check task
    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="check_drift")
    if drift_detected:
        logging.info("Data drift detected! Branching to retraining task.")
        return "retrain_model"
    else:
        logging.info("No significant data drift detected. Skipping retraining.")
        return "skip_retraining"

def run_retraining():
    from src.train import train_model
    logging.info("Starting model retraining task...")
    run_id = train_model()
    if run_id:
        logging.info(f"Model successfully retrained and logged to MLflow. Run ID: {run_id}")
    else:
        raise ValueError("Model training failed or returned no Run ID.")

# Define DAG configuration
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "loan_default_mlops_pipeline",
    default_args=default_args,
    description="Automated retraining pipeline: Ingestion -> Validation -> Drift -> Retraining",
    schedule_interval=timedelta(days=7), # Run weekly
    catchup=False,
) as dag:

    # 1. Ingestion Task
    ingest_task = PythonOperator(
        task_id="run_ingestion",
        python_callable=run_ingestion,
    )

    # 2. Validation Task
    validate_task = PythonOperator(
        task_id="run_validation",
        python_callable=run_validation,
    )

    # 3. Drift Check Task
    drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=run_drift_check,
    )

    # 4. Branching decision task based on XCom value
    branch_task = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=check_branch,
    )

    # 5. Model Retraining Task (if drift is True)
    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=run_retraining,
    )

    # 6. Skip Retraining Task (if drift is False)
    skip_task = EmptyOperator(
        task_id="skip_retraining",
    )

    # Define task dependencies
    ingest_task >> validate_task >> drift_task >> branch_task
    branch_task >> retrain_task
    branch_task >> skip_task
