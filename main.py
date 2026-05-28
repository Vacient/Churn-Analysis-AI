from typing import List, Optional

from fastapi import FastAPI
import joblib
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from services.sentiment_service import SentimentResult, get_sentiment_result

load_dotenv()

app = FastAPI()

model = joblib.load("churn_model.pkl")


class CustomerFeatures(BaseModel):
    customer_id: Optional[str] = None
    # monthly_bill: float = Field(..., ge=0)
    complaints: int = Field(..., ge=0)
    downtime_hours: float = Field(..., ge=0)
    resolution_time: int = Field(..., ge=0)
    duration_time: int = Field(..., ge=0)
    description: str = Field(..., min_length=1)
    # payment_delay_days: float = Field(..., ge=0)
    # contract_months: float = Field(..., ge=0)


class PredictionResult(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_probability: float
    sentiment_label: str
    sentiment_score: int


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
    "sentiment",
]


def _to_feature_vector(customer: CustomerFeatures, sentiment_score: int) -> list[float]:
    base_features = {
        "complaints": customer.complaints,
        "downtime_hours": customer.downtime_hours,
        "resolution_time": customer.resolution_time,
        "duration_time": customer.duration_time,
        "sentiment": sentiment_score,
    }
    return [float(base_features[feature]) for feature in FEATURE_ORDER]


def _to_prediction_result(
    customer: CustomerFeatures,
    prediction: int,
    probability: float,
    sentiment: SentimentResult,
) -> PredictionResult:
    return PredictionResult(
        customer_id=customer.customer_id,
        churn_prediction=int(prediction),
        churn_probability=float(probability),
        sentiment_label=sentiment.sentiment,
        sentiment_score=sentiment.score,
    )


@app.get("/")
def home():
    return {"message": "Churn API running"}


@app.post("/predict")
def predict(customer: CustomerFeatures) -> PredictionResult:
    sentiment = get_sentiment_result(customer.description)
    data = pd.DataFrame(
        [_to_feature_vector(customer, sentiment.score)],
        columns=FEATURE_ORDER,
    )
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return _to_prediction_result(customer, int(prediction), float(probability), sentiment)


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    sentiments = [get_sentiment_result(customer.description) for customer in payload.customers]
    data = pd.DataFrame(
        [
            _to_feature_vector(customer, sentiment.score)
            for customer, sentiment in zip(payload.customers, sentiments)
        ],
        columns=FEATURE_ORDER,
    )
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    results = [
        _to_prediction_result(customer, int(pred), float(prob), sentiment)
        for customer, pred, prob, sentiment in zip(
            payload.customers,
            predictions,
            probabilities,
            sentiments,
        )
    ]

    return BatchPredictResponse(predictions=results)