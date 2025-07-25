from fastapi import FastAPI
import joblib
import os

app = FastAPI()

# ✅ Safe model load:
model_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

# Your routes here
@app.get("/")
def home():
    return {"message": "Model loaded successfully"}

