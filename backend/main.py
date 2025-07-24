from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")

app = FastAPI()

class Transaction(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "Fraud Detection API is up"}

@app.post("/predict")
def predict(transaction: Transaction):
    prediction = model.predict([transaction.features])
    return {"fraud": bool(prediction[0])}
