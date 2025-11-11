#!/usr/bin/env python3
"""
Generate SHAP summary for hybrid model explainability (auto feature alignment).
"""

import shap, joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, io

MODEL_PATH = "data/model/hybrid_model.pkl"
HYBRID_PATH = "data/model/text_embeddings.csv"

print("📦 Loading model and data...")
rf, xgb, meta, scaler, best_thresh = joblib.load(MODEL_PATH)

# --- Load embeddings safely ---
with open(HYBRID_PATH, "rb") as f:
    raw_bytes = f.read()

data = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)
print(f"✅ Loaded embeddings: {data.shape}")

# --- Clean non-numeric values ---
numeric_cols = [c for c in data.columns if c.isdigit()]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data[numeric_cols] = data[numeric_cols].apply(lambda col: col.fillna(col.mean()))

# --- Sample subset for SHAP (to save memory) ---
data = data.sample(500, random_state=42)
X = data[numeric_cols]

# --- Align features with scaler ---
expected_features = getattr(scaler, "feature_names_in_", None)
if expected_features is not None:
    missing = [f for f in expected_features if f not in X.columns]
    extra = [c for c in X.columns if c not in expected_features]
    if extra:
        print(f"⚠️ Dropping unseen columns: {extra}")
        X = X.drop(columns=extra, errors="ignore")
    if missing:
        print(f"⚠️ Missing features (added as 0s): {missing}")
        for f in missing:
            X[f] = 0

# --- Scale and run SHAP ---
X_scaled = scaler.transform(X)

print("🔍 Running SHAP explainability...")
explainer = shap.Explainer(rf)
shap_values = explainer(X_scaled)

# --- Save plot ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png")
print("✅ Saved SHAP summary → reports/shap_summary.png")
