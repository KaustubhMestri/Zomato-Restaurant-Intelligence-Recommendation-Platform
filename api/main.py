from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Zomato ML API")

# Load models
success_model = joblib.load("/Users/kaustubhmestri/Projects/Zomato Intelligence/models/xgb_success_model.pkl")
rating_model = joblib.load("/Users/kaustubhmestri/Projects/Zomato Intelligence/models/xgb_rating_model.pkl")

@app.get("/")
def home():
    return {"message": "Zomato ML API is running"}

@app.post("/predict-success")
def predict_success(data: dict):
    df = pd.DataFrame([data])
    prob = success_model.predict_proba(df)[0][1]
    return {"success_probability": round(float(prob), 3)}

@app.post("/predict-rating")
def predict_rating(data: dict):
    df = pd.DataFrame([data])
    rating = rating_model.predict(df)[0]
    return {"predicted_rating": round(float(rating), 2)}

