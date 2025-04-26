from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)

# Load your model
random_forest = joblib.load("random_forest.pkl")


@app.route("/")
def home():
    return "Model API is up and running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]  # assume a list of feature values
    prediction = random_forest.predict([features])
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
