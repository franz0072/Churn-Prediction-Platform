
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed_churn.csv"


def test_model_loads_and_predicts():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Churn"]).head(5)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    assert len(predictions) == 5
    assert len(probabilities) == 5
    assert set(predictions).issubset({0, 1})
    assert all(0 <= p <= 1 for p in probabilities)


from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Customer Churn Prediction API is running"
    }


def test_predict_endpoint():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()
    assert result["churn_prediction"] in [0, 1]
    assert result["churn_label"] in ["Stay", "Churn"]
    assert 0 <= result["churn_probability"] <= 1
