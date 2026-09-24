"""Extract model coefficients and scaler parameters from pickle files."""
import joblib
import json
import numpy as np

model = joblib.load("cardio_model.pkl")
scaler = joblib.load("scaler.pkl")

params = {
    "coefficients": model.coef_[0].tolist(),
    "intercept": model.intercept_[0],
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "feature_names": ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"],
    "classes": model.classes_.tolist()
}

print(json.dumps(params, indent=2))
