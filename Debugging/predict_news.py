#!/usr/bin/env python3
"""
FINAL CORRECT VERSION - Prediction Script
------------------------------------------
✅ Matches training: raw embeddings + scaled SNA
✅ Uses sna_scaler (not unified scaler)
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_PATH = "data/model/hybrid_model.pkl"

def predict_news(text: str):
    """
    Predict whether news text is fake or real.
    """
    # Load trained model package
    model_package = joblib.load(MODEL_PATH)
    rf = model_package['rf']
    xgb = model_package['xgb']
    meta = model_package['meta']
    sna_scaler = model_package['sna_scaler']  # ← Only SNA scaler
    best_thresh = model_package['threshold']
    feature_info = model_package['feature_info']
    
    encoder = SentenceTransformer("all-mpnet-base-v2")

    print("🧠 Encoding text with SBERT...")
    embedding = encoder.encode([text])
    
    # Create SNA features (neutral placeholders)
    n_sna = feature_info['n_sna_features']
    sna_features = np.array([[0.5] * n_sna])
    
    # Scale ONLY SNA features
    sna_scaled = sna_scaler.transform(sna_features)
    
    # Combine: RAW embedding + SCALED SNA
    X_combined = np.hstack([embedding, sna_scaled])

    # Ensemble predictions
    rf_prob = rf.predict_proba(X_combined)[:, 1]
    xgb_prob = xgb.predict_proba(X_combined)[:, 1]
    meta_input = np.vstack([rf_prob, xgb_prob]).T
    
    # Meta-model output: P(fake)
    prob_fake = float(meta.predict_proba(meta_input)[:, 1][0])
    prob_real = 1.0 - prob_fake

    # Make prediction
    predicted_label = "Fake" if prob_fake > best_thresh else "Real"

    # Convert to percentages
    fake_pct = prob_fake * 100
    real_pct = prob_real * 100

    # Display information
    print("\n📰 Input Text:")
    print(f"   {text[:100]}{'...' if len(text) > 100 else ''}\n")
    print(f"🎯 Probability Analysis:")
    print(f"   Fake: {fake_pct:.2f}%")
    print(f"   Real: {real_pct:.2f}%")
    print(f"📊 Model threshold: {best_thresh:.4f} ({best_thresh*100:.2f}%)")
    print(f"🏷️  Predicted Label: {predicted_label}")
    
    # Confidence
    if prob_fake > best_thresh:
        confidence = (prob_fake - best_thresh) / (1 - best_thresh) * 100
    else:
        confidence = (best_thresh - prob_fake) / best_thresh * 100
    confidence = min(confidence, 100)
    
    print(f"💪 Confidence: {confidence:.1f}%")

    return {
        "text": text,
        "fake_probability": round(fake_pct, 2),
        "real_probability": round(real_pct, 2),
        "predicted_label": predicted_label,
        "confidence": round(confidence, 2),
        "threshold": round(best_thresh, 4)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_news.py '<news text>'")
        print("\nExample:")
        print("  python3 predict_news.py 'Barack Obama bans the Pledge of Allegiance'")
    else:
        text = " ".join(sys.argv[1:])
        result = predict_news(text)
        
        # Visual separator
        print("\n" + "="*60)
        print(f"VERDICT: {result['predicted_label'].upper()}")
        print("="*60)