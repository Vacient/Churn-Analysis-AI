from faker import Faker
import pandas as pd
import random


fake = Faker()


data = []

for i in range(1000):
    tickets = random.randint(0, 10)
    # complaints = random.randint(0, 10)
    downtime = round(random.uniform(0.0, 24.0), 2)
    # payment_delay = random.randint(0, 5)
    resolution_time = random.randint(1, 50)
    duration_time = random.randint(1, 65)
    sentiment = random.choice([-1, 0, 1])

    risk = (
        tickets * 0.40 +
        downtime * 0.25 +
        resolution_time * 0.15 +
        duration_time * 0.10 +
        (1 - sentiment) * 5
    )

    risk_score = min(max(int(risk * 3), 0), 100)

    churn = 1 if risk_score > 50 else 0

    customer = {
        "customer_id": i + 1,
        # "name": fake.name(),
        # "email": fake.email(),
        # "phone": fake.phone_number(),
        # "monthly_bill": random.randint(15, 60),
        "complaints": tickets,
        "downtime_hours": downtime,
        # "payment_delay_days": payment_delay,
        # "contract_months": random.randint(1, 36),
        "resolution_time": resolution_time,
        "duration_time": duration_time,
        "sentiment": sentiment,
        "churn": churn,
    }

    data.append(customer)


df = pd.DataFrame(data)

print(df.head())


df.to_csv("telecom_churn.csv", index=False)

print("Dataset saved!")
