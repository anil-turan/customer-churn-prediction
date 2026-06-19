import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.pipeline import add_derived_features

# Resolve model path relative to this file so uvicorn can be launched
# from any working directory without breaking the pickle load.
MODEL_PATH = Path(__file__).resolve().parents[2] / "saved_models" / "churn_pipeline.pkl"

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)


class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
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


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM churn pipeline"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.model_dump()])
    df = add_derived_features(df)
    try:
        proba = pipeline.predict_proba(df)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if proba >= 0.7:
        tier = "high"
    elif proba >= 0.4:
        tier = "medium"
    else:
        tier = "low"

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_prediction=bool(proba >= 0.5),
        risk_tier=tier,
    )
