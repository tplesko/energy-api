from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})  # dopušta sa svih domen

model = joblib.load("optimized_rf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["data"]).reshape(1, -1)
        prediction = model.predict(features).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def health_check():
    return "✅ API radi!", 200
