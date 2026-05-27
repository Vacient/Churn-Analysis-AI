from faker import Faker
import pandas as pd
import random


fake = Faker()


data = []

for i in range(1000):
    complaints = random.randint(0, 10)
    downtime = random.randint(0, 20)
    payment_delay = random.randint(0, 5)

    risk = complaints * 0.15 + downtime * 0.05 + payment_delay * 0.2

    churn = "Yes" if risk > 2 else "No"

    customer = {
        "customer_id": i + 1,
        "name": fake.name(),
        # "email": fake.email(),
        "phone": fake.phone_number(),
        "monthly_bill": random.randint(15, 60),
        "complaints": complaints,
        "downtime_hours": downtime,
        "payment_delay_days": payment_delay,
        "contract_months": random.randint(1, 36),
        "churn": churn,
    }

    data.append(customer)


df = pd.DataFrame(data)

print(df.head())


df.to_csv("telecom_churn.csv", index=False)

print("Dataset saved!")
