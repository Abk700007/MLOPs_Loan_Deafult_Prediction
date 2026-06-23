import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
import logging
from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

# Global database connection pool initialized lazily
_db_pool = None

def get_pool():
    """Initializes and returns the database connection pool lazily."""
    global _db_pool
    if _db_pool is None:
        try:
            logging.info("Initializing Supabase database connection pool...")
            _db_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=15,
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                dbname=DB_NAME,
                connect_timeout=10  # Fail fast instead of hanging indefinitely
            )
            logging.info("Database connection pool initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize database connection pool: {e}")
            raise
    return _db_pool

def get_connection():
    """Retrieves a database connection from the pool."""
    try:
        pool = get_pool()
        return pool.getconn()
    except Exception as e:
        logging.error(f"Error fetching connection from pool: {e}")
        raise

def release_connection(conn):
    """Returns a connection back to the database pool."""
    global _db_pool
    if _db_pool is not None and conn is not None:
        try:
            _db_pool.putconn(conn)
        except Exception as e:
            logging.error(f"Error releasing connection back to pool: {e}")

def initialize_database():
    """Creates the necessary tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # We will define a subset of features from Home Credit Default Risk dataset
    create_table_query = """
    CREATE TABLE IF NOT EXISTS loans (
        sk_id_curr INT PRIMARY KEY,
        target INT,
        code_gender VARCHAR(10),
        flag_own_car VARCHAR(5),
        flag_own_realty VARCHAR(5),
        cnt_children INT,
        amt_income_total DOUBLE PRECISION,
        amt_credit DOUBLE PRECISION,
        amt_annuity DOUBLE PRECISION,
        amt_goods_price DOUBLE PRECISION,
        days_birth INT,
        days_employed INT,
        ext_source_2 DOUBLE PRECISION,
        ext_source_3 DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        cursor.execute(create_table_query)
        conn.commit()
        logging.info("Database initialized and 'loans' table created successfully.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to initialize database: {e}")
        raise
    finally:
        cursor.close()
        release_connection(conn)

def save_data_to_db(df: pd.DataFrame, table_name: str = "loans"):
    """Inserts a pandas DataFrame into the specified table in PostgreSQL using high-speed bulk inserts."""
    if df.empty:
        logging.warning("DataFrame is empty. Skipping save.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    
    # Normalize columns to lowercase matching PostgreSQL schema
    df.columns = [col.lower() for col in df.columns]
    columns = list(df.columns)
    
    # Compile SQL using %s template for execute_values
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s ON CONFLICT (sk_id_curr) DO NOTHING"
    
    try:
        # Convert df values to Python types for inserting
        records = [tuple(row) for row in df.itertuples(index=False, name=None)]
        
        # execute_values sends the entire batch of 5000 rows in a single query
        # This is 100x to 1000x faster than executemany over the internet
        execute_values(cursor, query, records)
        conn.commit()
        logging.info(f"Successfully loaded {len(records)} records into table '{table_name}'.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error loading records into table: {e}")
        raise
    finally:
        cursor.close()
        release_connection(conn)

def fetch_data_from_db(table_name: str = "loans") -> pd.DataFrame:
    """Queries all records from the database and returns them as a pandas DataFrame."""
    conn = get_connection()
    query = f"SELECT * FROM {table_name}"
    try:
        df = pd.read_sql_query(query, conn)
        logging.info(f"Successfully fetched {len(df)} records from table '{table_name}'.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from database: {e}")
        raise
    finally:
        release_connection(conn)

def get_db_summary(table_name: str = "loans"):
    """Fetches total records and default counts from the database quickly without downloading the entire dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    query = f"SELECT COUNT(*), SUM(target) FROM {table_name}"
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        total = row[0] if row else 0
        defaults = int(row[1]) if row and row[1] is not None else 0
        return total, defaults
    except Exception as e:
        logging.error(f"Error fetching database summary: {e}")
        return 0, 0
    finally:
        cursor.close()
        release_connection(conn)
