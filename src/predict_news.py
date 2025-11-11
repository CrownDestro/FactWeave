#!/usr/bin/env python3
"""
Predict fake/real probability for a given text input using the trained hybrid model.
Handles feature name order and alignment automatically.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_PATH = "data/model/hybrid_model.pkl"

def predict_news(text: str):
    # Load model components
    rf, xgb, meta, scaler, best_thresh = joblib.load(MODEL_PATH)
    model = SentenceTransformer("all-mpnet-base-v2")

    print("🧠 Encoding text with SBERT...")
    embedding = model.encode([text])
    embedding_df = pd.DataFrame(embedding, columns=[str(i) for i in range(embedding.shape[1])])

    # Add dummy SNA features
    possible_features = [
        "degree", "pagerank", "clustering", "community",
        "closeness", "eigenvector", "label_propagated"
    ]
    for col in possible_features:
        embedding_df[col] = 0.5  # neutral placeholder

    # Align features to scaler's training order
    expected_cols = list(getattr(scaler, "feature_names_in_", embedding_df.columns))
    for c in expected_cols:
        if c not in embedding_df.columns:
            embedding_df[c] = 0.5  # fill missing
    embedding_df = embedding_df.reindex(columns=expected_cols)  # ✅ reorder correctly

    # Scale safely
    X_scaled = scaler.transform(embedding_df)

    # Get predictions
    rf_prob = rf.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb.predict_proba(X_scaled)[:, 1]
    meta_input = np.vstack([rf_prob, xgb_prob]).T
    final_prob = meta.predict_proba(meta_input)[:, 1][0]

    # final_prob is P(fake) because label=1 means fake
    fake_prob = float(final_prob)
    real_prob = 1.0 - fake_prob

    # Use saved best_thresh from training for consistent decision boundary
    label = "Fake" if fake_prob > best_thresh else "Real"

    print("\n🧾 Input Text:")
    print(f"   {text}\n")
    print(f"🎯 Fake probability: {fake_prob * 100:.2f}%")
    print(f"✅ Real probability: {real_prob * 100:.2f}%")
    print(f"🧩 Predicted Label: {label}")

    return {
        "text": text,
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "predicted_label": label
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict_news.py '<news text>'")
    else:
        text = sys.argv[1]
        predict_news(text)