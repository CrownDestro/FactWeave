#!/usr/bin/env python3
"""
CORRECTED Flask API for Fake News Detection
--------------------------------------------
✅ Uses RF-only model (no meta-ensemble)
✅ Scales ONLY SNA features (embeddings NOT scaled)
✅ Proper feature alignment matching training
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
model_package = joblib.load(MODEL_PATH)

# ✅ Correct model package structure (RF-only)
rf = model_package['rf']
ensemble_type = model_package.get('ensemble_type', 'rf_only')
sna_scaler = model_package['sna_scaler']  # ✅ Correct scaler name
best_thresh = model_package['threshold']
feature_info = model_package['feature_info']

encoder = SentenceTransformer("all-mpnet-base-v2")

print(f"✅ Model loaded successfully!")
print(f"🔧 Model type: {ensemble_type}")
print(f"📊 Decision threshold: {best_thresh:.4f} ({best_thresh*100:.2f}%)")
print(f"📐 Features: {feature_info['n_embedding_features']} embeddings + {feature_info['n_sna_features']} SNA")

# ---------------------------------------------------------
# 🔹 Helper: Predict function
# ---------------------------------------------------------
def predict_text(text: str):
    """
    Predict fake/real probability for given text.
    
    Args:
        text: News article text
        
    Returns:
        dict: Prediction results with probabilities and label
    """
    # 1. SBERT embedding (already normalized by model)
    embedding = encoder.encode([text], normalize_embeddings=True)
    
    # 2. Add neutral SNA placeholders
    n_sna = feature_info['n_sna_features']
    sna_features = np.array([[0.5] * n_sna])
    
    # 3. Scale ONLY SNA features (embeddings are NOT scaled)
    sna_scaled = sna_scaler.transform(sna_features)
    
    # 4. Combine: RAW embedding + SCALED SNA
    X_combined = np.hstack([embedding, sna_scaled])
    
    # 5. Get RF prediction (P(fake))
    prob_fake = float(rf.predict_proba(X_combined)[:, 1][0])
    prob_real = 1.0 - prob_fake

    # 6. Make prediction using optimized threshold
    predicted_label = "Fake" if prob_fake > best_thresh else "Real"
    
    # 7. Calculate confidence
    if prob_fake > best_thresh:
        # Confidence in "Fake" prediction
        confidence = (prob_fake - best_thresh) / (1 - best_thresh) * 100
    else:
        # Confidence in "Real" prediction
        confidence = (best_thresh - prob_fake) / best_thresh * 100
    confidence = min(confidence, 100)

    # Convert to percentages
    fake_pct = prob_fake * 100
    real_pct = prob_real * 100

    return {
        "text": str(text)[:200] + ("..." if len(str(text)) > 200 else ""),  # Truncate for response
        "fake_probability": float(round(fake_pct, 2)),
        "real_probability": float(round(real_pct, 2)),
        "predicted_label": str(predicted_label),
        "confidence": float(round(confidence, 2)),
        "threshold": float(round(best_thresh * 100, 2)),  # As percentage
        "model_type": ensemble_type
    }


# ---------------------------------------------------------
# 🔹 Flask Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running 🚀",
        "version": "4.0-corrected",
        "model_type": ensemble_type,
        "model_threshold": round(best_thresh, 4),
        "threshold_percentage": f"{best_thresh*100:.2f}%",
        "features": {
            "embeddings": feature_info['n_embedding_features'],
            "sna_features": feature_info['n_sna_features']
        },
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {"text": "your news text here"},
            "example": {
                "text": "Breaking: Scientists discover new planet in solar system"
            }
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    data = request.get_json()
    
    # Validate input
    if not data or "text" not in data:
        return jsonify({
            "error": "Missing 'text' in request body",
            "example": {"text": "Your news article here"}
        }), 400
    
    text = data["text"]
    
    # Validate text content
    if not text or not isinstance(text, str):
        return jsonify({
            "error": "Invalid text format. Must be a non-empty string."
        }), 400
    
    if len(text.strip()) < 10:
        return jsonify({
            "error": "Text too short. Please provide at least 10 characters."
        }), 400
    
    if len(text) > 10000:
        return jsonify({
            "warning": "Text is very long (>10k chars). Processing first 10k characters.",
            "text_truncated": True
        }), 400

    try:
        result = predict_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "type": type(e).__name__
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_type": ensemble_type,
        "threshold": round(best_thresh, 4),
        "threshold_percentage": f"{best_thresh*100:.2f}%",
        "encoder": "all-mpnet-base-v2",
        "features": {
            "embeddings": feature_info['n_embedding_features'],
            "sna_features": feature_info['n_sna_features']
        }
    })

@app.route("/info", methods=["GET"])
def info():
    """Model information endpoint"""
    return jsonify({
        "model": {
            "type": ensemble_type,
            "architecture": "Random Forest (768D embeddings + 7 SNA features)",
            "threshold": {
                "value": round(best_thresh, 4),
                "percentage": f"{best_thresh*100:.2f}%",
                "description": "Optimized using Youden's J statistic"
            }
        },
        "features": {
            "text_embeddings": {
                "dimension": feature_info['n_embedding_features'],
                "model": "all-mpnet-base-v2 (sentence-transformers)",
                "preprocessing": "Normalized (not scaled)"
            },
            "sna_features": {
                "count": feature_info['n_sna_features'],
                "features": feature_info['sna_cols'],
                "preprocessing": "StandardScaler applied"
            }
        },
        "performance": {
            "note": "See reports/ directory for detailed metrics"
        }
    })

@app.route("/batch", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint for multiple texts"""
    data = request.get_json()
    
    if not data or "texts" not in data:
        return jsonify({
            "error": "Missing 'texts' array in request body",
            "example": {"texts": ["text1", "text2", "text3"]}
        }), 400
    
    texts = data["texts"]
    
    if not isinstance(texts, list):
        return jsonify({
            "error": "Invalid format. 'texts' must be an array."
        }), 400
    
    if len(texts) > 100:
        return jsonify({
            "error": "Too many texts. Maximum 100 texts per batch."
        }), 400
    
    try:
        results = []
        for i, text in enumerate(texts):
            if not text or len(str(text).strip()) < 10:
                results.append({
                    "index": i,
                    "error": "Text too short or empty"
                })
            else:
                result = predict_text(str(text))
                result["index"] = i
                results.append(result)
        
        return jsonify({
            "count": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "details": str(e)
        }), 500

# ---------------------------------------------------------
# 🔹 Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Fake News Detection API...")
    print("🔗 Access at: http://localhost:5000")
    print("📖 API docs: http://localhost:5000/")
    print("🏥 Health check: http://localhost:5000/health")
    print("ℹ️  Model info: http://localhost:5000/info")
    print(f"🎯 Using {ensemble_type} with threshold {best_thresh:.4f}\n")
    app.run(host="0.0.0.0", port=5000, debug=True)