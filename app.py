"""
FastAPI application for real-time customer churn prediction.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Storage for model bundle in application state
pipeline_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path("artifacts/churn_model_pipeline.joblib")
    if not model_path.exists():
        raise RuntimeError("Artifact churn_model_pipeline.joblib not found. Run train.py first.")
    pipeline_state["bundle"] = joblib.load(model_path)
    yield
    pipeline_state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


class CustomerData(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, example=12)
    contract_type: Literal["Month-to-Month", "One-Year", "Two-Year"] = Field(..., example="Month-to-Month")
    internet_service: Literal["Fiber Optic", "DSL", "No"] = Field(..., example="Fiber Optic")
    payment_method: Literal["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"] = Field(..., example="Electronic Check")
    monthly_charges: float = Field(..., ge=0.0, example=85.50)
    total_charges: Optional[float] = Field(None, ge=0.0, example=1026.00)
    support_tickets: int = Field(..., ge=0, example=3)
    paperless_billing: int = Field(..., ge=0, le=1, example=1)


class PredictionResponse(BaseModel):
    churn_risk_score: float
    churn_prediction: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision_threshold_used: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "bundle" in pipeline_state}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    bundle = pipeline_state.get("bundle")
    if not bundle:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready.")

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    threshold = bundle["optimal_threshold"]

    # Convert request to single-row dataframe
    df = pd.DataFrame([customer.model_dump()])

    # Transform and infer
    X_transformed = preprocessor.transform(df)
    churn_probability = float(model.predict_proba(X_transformed)[:, 1][0])
    is_churn = 1 if churn_probability >= threshold else 0

    if churn_probability >= 0.70:
        risk_level = "HIGH"
    elif churn_probability >= threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        churn_risk_score=round(churn_probability, 4),
        churn_prediction=is_churn,
        risk_level=risk_level,
        decision_threshold_used=round(threshold, 4)
    )