from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("churn_model.pkl")


@app.get("/")
def home():
    return {"message": "Churn API running"}


@app.post("/predict")
def predict(
    monthly_bill: int,
    complaints: int,
    downtime_hours: int,
    payment_delay_days: int,
    contract_months: int,
):

    data = np.array([
        [
            monthly_bill,
            complaints,
            downtime_hours,
            payment_delay_days,
            contract_months,
        ]
    ])

    prediction = model.predict(data)

    return {"churn_prediction": int(prediction[0])}