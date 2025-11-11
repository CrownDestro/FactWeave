#!/usr/bin/env python3
"""
Flask API for Fake News Detection using the Hybrid Model
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

# ---------------------------------------------------------
# 🔹 Initialize app and model
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)

MODEL_PATH = "data/model/hybrid_model.pkl"
print("🔹 Loading hybrid model and SBERT encoder...")
rf, xgb, meta, scaler, best_thresh = joblib.load(MODEL_PATH)
encoder = SentenceTransformer("all-mpnet-base-v2")

# ---------------------------------------------------------
# 🔹 Helper: Predict function (same as your CLI version)
# ---------------------------------------------------------
def predict_text(text: str):
    # SBERT embedding
    embedding = encoder.encode([text])
    df = pd.DataFrame(embedding, columns=[str(i) for i in range(embedding.shape[1])])

    # Add neutral SNA placeholders
    sna_features = [
        "degree", "pagerank", "clustering", "community",
        "closeness", "eigenvector", "label_propagated"
    ]
    for f in sna_features:
        df[f] = 0.5

    # Align columns with scaler training order
    expected_cols = list(getattr(scaler, "feature_names_in_", df.columns))
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0.5
    df = df.reindex(columns=expected_cols)

    # Scale + Predict
    X_scaled = scaler.transform(df)
    rf_prob = rf.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb.predict_proba(X_scaled)[:, 1]
    meta_input = np.vstack([rf_prob, xgb_prob]).T
    final_prob = meta.predict_proba(meta_input)[:, 1][0]

    # --- Calibrate + Flip Probabilities ---
    calibrated_prob = final_prob / (final_prob + (1 - final_prob) / 10)
    fake_pct = (1 - calibrated_prob) * 100
    real_pct = calibrated_prob * 100
    label = "Fake" if fake_pct > real_pct else "Real"

    return {
        "text": str(text),
        "fake_probability": float(round(fake_pct, 2)),
        "real_probability": float(round(real_pct, 2)),
        "predicted_label": str(label)
    }


# ---------------------------------------------------------
# 🔹 Flask Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running 🚀",
        "usage": "Send POST /predict with {'text': 'your news text'}"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    result = predict_text(data["text"])
    return jsonify(result)

# ---------------------------------------------------------
# 🔹 Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
