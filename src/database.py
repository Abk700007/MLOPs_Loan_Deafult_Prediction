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
