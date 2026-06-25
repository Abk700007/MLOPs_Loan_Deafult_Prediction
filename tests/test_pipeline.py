import pytest
from fastapi.testclient import TestClient
import pandas as pd
from app.main import app, preprocess_application, LoanApplication

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture that triggers startup lifespan/events."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Verify that the health check endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "model_loaded" in json_data

def test_preprocessing():
    """Verify that categorical features are encoded correctly."""
    app_data = LoanApplication(
        code_gender="F",
        flag_own_car="Y",
        flag_own_realty="N",
        cnt_children=1,
        amt_income_total=150000.0,
        amt_credit=300000.0,
        amt_annuity=12000.0,
        amt_goods_price=280000.0,
        days_birth=-14000,
        days_employed=-1200,
        ext_source_2=0.5,
        ext_source_3=0.4
    )
    
    df = preprocess_application(app_data)
    
    # Check shape (now 19 columns due to 7 engineered features)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 19)
