from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Zomato ML API")

# -------------------------------------------------
# Correct project-root–based paths (IMPORTANT)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

success_model = joblib.load(MODEL_DIR / "xgb_success_model.pkl")
rating_model = joblib.load(MODEL_DIR / "xgb_rating_model.pkl")

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "Zomato ML API running"}

# -------------------------------------------------
# Predict restaurant success
# -------------------------------------------------
@app.post("/predict-success")
def predict_success(data: dict):
    df = pd.DataFrame([data])
    prob = success_model.predict_proba(df)[0][1]
    return {"success_probability": round(float(prob), 3)}

# -------------------------------------------------
# Predict restaurant rating
# -------------------------------------------------
@app.post("/predict-rating")
def predict_rating(data: dict):
    df = pd.DataFrame([data])
    rating = rating_model.predict(df)[0]
    return {"predicted_rating": round(float(rating), 2)}
