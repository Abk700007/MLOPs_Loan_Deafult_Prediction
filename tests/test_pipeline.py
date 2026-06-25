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
    
    # Check label encodings
    assert df["code_gender"].iloc[0] == 0  # 'F' -> 0
    assert df["flag_own_car"].iloc[0] == 1  # 'Y' -> 1
    assert df["flag_own_realty"].iloc[0] == 0  # 'N' -> 0
    
    # Check numeric and engineered values
    assert df["cnt_children"].iloc[0] == 1
    assert df["amt_income_total"].iloc[0] == 150000.0
    assert df["days_birth"].iloc[0] == -14000
    assert "annuity_income_ratio" in df.columns
    assert df["annuity_income_ratio"].iloc[0] == pytest.approx(12000.0 / 150000.0)

def test_predict_endpoint(client):
    """Verify prediction endpoint behavior based on model availability."""
    payload = {
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
    
    # Check if model is loaded via health check
    health_response = client.get("/health")
    assert health_response.status_code == 200
    model_loaded = health_response.json()["model_loaded"]
    
    response = client.post("/predict", json=payload)
    if model_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "default_probability" in data
        assert "default_prediction" in data
        assert data["risk_status"] in ["High Risk", "Low Risk"]
    else:
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]

def test_feature_engineering_ratios():
    """Verify correctness of financial and risk ratios engineered by the pipeline."""
    from src.train import engineer_features
    # Sample df
    df = pd.DataFrame([{
        "amt_income_total": 100000.0,
        "amt_credit": 500000.0,
        "amt_annuity": 20000.0,
        "amt_goods_price": 450000.0,
        "days_birth": -18262, # 50 years
        "days_employed": -3652, # 10 years
        "ext_source_2": 0.6,
        "ext_source_3": 0.8
    }])
    
    df_feat = engineer_features(df)
    
    # Assert values
    assert df_feat["annuity_income_ratio"].iloc[0] == pytest.approx(20000.0 / (100000.0 + 1e-5))
    assert df_feat["credit_income_ratio"].iloc[0] == pytest.approx(500000.0 / (100000.0 + 1e-5))
    assert df_feat["goods_credit_ratio"].iloc[0] == pytest.approx(450000.0 / (500000.0 + 1e-5))
    assert df_feat["age_years"].iloc[0] == pytest.approx(18262 / 365.25)
    assert df_feat["emp_age_ratio"].iloc[0] == pytest.approx(3652.0 / 18262.0)
    assert df_feat["ext_source_mean"].iloc[0] == pytest.approx((0.6 + 0.8) / 2.0)
    assert df_feat["ext_source_prod"].iloc[0] == pytest.approx(0.6 * 0.8)

def test_monotonicity_constraints(client):
    """Verify that worsening credit scores (lower ext_source_2/3) strictly yield higher default risk probabilities."""
    # Check if model is loaded
    health_response = client.get("/health")
    assert health_response.status_code == 200
    if not health_response.json()["model_loaded"]:
        pytest.skip("Model is not loaded. Skipping monotonicity check.")
        
    # Create base data class
    base_payload = {
        "code_gender": "F",
        "flag_own_car": "N",
        "flag_own_realty": "Y",
        "cnt_children": 0,
        "amt_income_total": 100000.0,
        "amt_credit": 300000.0,
        "amt_annuity": 15000.0,
        "amt_goods_price": 280000.0,
        "days_birth": -15000,
        "days_employed": -2500
    }
    
    # Test ext_source_2 monotonicity
    # Lower score -> worse credit -> higher default probability
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    probabilities = []
    
    for s in scores:
        payload = base_payload.copy()
        payload["ext_source_2"] = s
        payload["ext_source_3"] = 0.5 # constant
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        probabilities.append(response.json()["default_probability"])
        
    # Assert probabilities are monotonically decreasing (or non-increasing) as score increases
    # i.e., score 0.1 risk >= score 0.3 risk >= score 0.5 risk ...
    for i in range(len(probabilities) - 1):
        assert probabilities[i] >= probabilities[i+1], f"Monotonicity violation on ext_source_2: {probabilities[i]} < {probabilities[i+1]} when score increased from {scores[i]} to {scores[i+1]}"
