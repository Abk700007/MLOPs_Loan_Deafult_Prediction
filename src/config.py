import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# PostgreSQL Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "loan_db")

# Database URI helper
def get_db_uri():
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Set request timeouts and retries for MLflow client to prevent hanging
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = os.getenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

# If using DagsHub, set environment variables for MLflow client
if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# Model Configurations
MODEL_NAME = "loan_default_xgboost"
EXPERIMENT_NAME = "loan_default_prediction"

# Data Drift & Performance Thresholds
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.1"))  # p-value threshold for Kolmogorov-Smirnov test
