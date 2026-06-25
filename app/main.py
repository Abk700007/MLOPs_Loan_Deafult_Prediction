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
