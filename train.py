from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


import pandas as pd

df = pd.read_csv("telecom_churn.csv")

print(df.head())


print(df.info())
print(df.isnull().sum())


df = df.drop(columns=["customer_id","name", "phone"])

df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

X = df.drop("churn", axis=1)
y = df["churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(predictions[:10])


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


joblib.dump(model, "churn_model.pkl")
