from typing import List, Optional

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("churn_model.pkl")


class CustomerFeatures(BaseModel):
    customer_id: Optional[str] = None
    # monthly_bill: float = Field(..., ge=0)
    complaints: int = Field(..., ge=0)
    downtime_hours: float = Field(..., ge=0)
    resolution_time: int = Field(..., ge=0)
    duration_time: int = Field(..., ge=0)
    sentiment: int = Field(..., ge=0)
    # payment_delay_days: float = Field(..., ge=0)
    # contract_months: float = Field(..., ge=0)


class PredictionResult(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_probability: float


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResult]


FEATURE_ORDER = [
    # "monthly_bill",
    "complaints",
    "downtime_hours",
    "resolution_time",
    "duration_time",
    "sentiment"
]


def _to_feature_vector(customer: CustomerFeatures) -> list[float]:
    return [float(getattr(customer, feature)) for feature in FEATURE_ORDER]


def _to_prediction_result(customer: CustomerFeatures, prediction: int, probability: float) -> PredictionResult:
    return PredictionResult(
        customer_id=customer.customer_id,
        churn_prediction=int(prediction),
        churn_probability=float(probability),
    )


@app.get("/")
def home():
    return {"message": "Churn API running"}


@app.post("/predict")
def predict(customer: CustomerFeatures) -> PredictionResult:
    data = pd.DataFrame([_to_feature_vector(customer)], columns=FEATURE_ORDER)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return _to_prediction_result(customer, int(prediction), float(probability))


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    data = pd.DataFrame(
        [_to_feature_vector(customer) for customer in payload.customers],
        columns=FEATURE_ORDER,
    )
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    results = [
        _to_prediction_result(customer, int(pred), float(prob))
        for customer, pred, prob in zip(payload.customers, predictions, probabilities)
    ]

    return BatchPredictResponse(predictions=results)