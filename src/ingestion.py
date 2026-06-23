import os
import pandas as pd
import logging
from src.database import initialize_database, save_data_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

# Path to the actual Kaggle dataset file (expected locally or in Google Colab)
KAGGLE_CSV_PATH = "data/application_train.csv"

def verify_dataset_exists():
    """Checks if the Home Credit Default Risk dataset exists locally."""
    if os.path.exists(KAGGLE_CSV_PATH):
        logging.info(f"Kaggle dataset found at {KAGGLE_CSV_PATH}")
        return True
    
    # Informative message detailing how to download the Kaggle dataset
    error_msg = (
        f"\n{'='*80}\n"
        f"CRITICAL: Home Credit Default Risk dataset NOT found at '{KAGGLE_CSV_PATH}'\n\n"
        f"Since mock data is not permitted, please follow these steps to download the actual data:\n"
        f"1. Download 'application_train.csv' from Kaggle:\n"
        f"   https://www.kaggle.com/competitions/home-credit-default-risk/data\n"
        f"2. Create a folder named 'data' in your project root directory.\n"
        f"3. Place 'application_train.csv' inside the 'data' folder.\n\n"
        f"For Google Colab Execution:\n"
        f"You can download the dataset directly to Colab by running:\n"
        f"  !pip install kaggle\n"
        f"  # Upload your kaggle.json file first\n"
        f"  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
        f"  !kaggle competitions download -c home-credit-default-risk\n"
        f"  !unzip -o home-credit-default-risk.zip -d data/\n"
        f"{'='*80}\n"
    )
    logging.error(error_msg)
    return False

def ingest_data(limit_records=50000):
    """
    Initializes the database schema and loads the actual Kaggle Home Credit dataset.
    Processes the CSV in chunks for efficient memory utilization.
    """
    initialize_database()
    
    if not verify_dataset_exists():
        raise FileNotFoundError(f"Missing required Kaggle dataset file: {KAGGLE_CSV_PATH}")
        
    logging.info(f"Starting data ingestion from {KAGGLE_CSV_PATH}...")
    
    # Columns required by our database schema
    cols_needed = [
        "SK_ID_CURR", "TARGET", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2", "EXT_SOURCE_3"
    ]
    
    # Chunksize determines how many rows we read and insert at a time
    # 5000 is a safe size for optimal network batch size and avoids Supabase timeout issues
    chunk_size = 5000
    records_loaded = 0
    
    try:
        # Read in chunks
        for chunk in pd.read_csv(KAGGLE_CSV_PATH, usecols=cols_needed, chunksize=chunk_size):
            # If we've reached the ingestion limit, stop
            if limit_records and records_loaded >= limit_records:
                logging.info(f"Reached ingestion limit of {limit_records} rows. Stopping.")
                break
                
            logging.info(f"Processing batch of {len(chunk)} rows (Total loaded so far: {records_loaded})...")
            
            # Save the chunk to the database
            save_data_to_db(chunk)
            records_loaded += len(chunk)
            
        logging.info(f"Ingestion completed successfully. Total records loaded: {records_loaded}")
        
    except Exception as e:
        logging.error(f"Failed to ingest data from CSV: {e}")
        raise

if __name__ == "__main__":
    # Note: 50,000 records is ample for model training and respects database storage limits.
    ingest_data(limit_records=50000)
