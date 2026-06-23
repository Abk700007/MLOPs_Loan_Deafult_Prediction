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

