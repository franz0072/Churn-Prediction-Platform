
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Load the trained model once when the API starts
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained machine learning model.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        customer_df = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(customer_df)[0])
        probability = float(model.predict_proba(customer_df)[0][1])

        return {
            "churn_prediction": prediction,
            "churn_label": "Churn" if prediction == 1 else "Stay",
            "churn_probability": round(probability, 4),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the input data.",
        ) from exc
