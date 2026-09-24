from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

import os

app = Flask(__name__)
CORS(app)

# Helper function to find required files in local or project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "..", "ml project", "Project"),
    r"C:\Users\OM\Desktop\ml project\Project",
]

def get_file_path(filename):
    for d in SEARCH_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return filename

# Load dataset (optional, handled gracefully if file is absent)
try:
    df = pd.read_csv(get_file_path("cardio_cleaned.csv"))
except Exception:
    df = None

# Load trained model and scaler
model = joblib.load(get_file_path("cardio_model.pkl"))
scaler = joblib.load(get_file_path("scaler.pkl"))

# Features used by the model
features = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]


@app.route("/")
def home():
    return "Cardiovascular Prediction API is Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Get input values
    values = []

    for feature in features:
        values.append(float(data[feature]))

    # Convert input into DataFrame
    input_data = pd.DataFrame([values], columns=features)

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Heuristic adjustments: If user inputs bad habits, explicitly force the probability up 
    # to avoid the data correlation artifact where older models predict lower probability for smokers
    if data["smoke"] == "1" or data["smoke"] == 1 or data["smoke"] == 1.0:
        probability += 6.5
    if data["alco"] == "1" or data["alco"] == 1 or data["alco"] == 1.0:
        probability += 4.2
    if data["active"] == "0" or data["active"] == 0 or data["active"] == 0.0:
        probability += 3.8
    
    probability = min(99.0, probability) # cap at 99%
    
    if probability > 50.0:
        prediction = 1
    else:
        prediction = 0

    # Result
    if prediction == 1:
        result = "Cardiovascular Disease Detected"
    else:
        result = "No Cardiovascular Disease Detected"

    return jsonify({
        "prediction": int(prediction),
        "result": result,
        "probability": round(probability, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)