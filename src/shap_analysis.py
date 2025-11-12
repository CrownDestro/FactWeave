#!/usr/bin/env python3
"""
CORRECTED SHAP Analysis for RF-only model
------------------------------------------
✅ Properly loads RF-only model package
✅ Correctly handles SNA scaling (embeddings NOT scaled)
✅ Aligns features properly
"""

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

MODEL_PATH = "data/model/hybrid_model.pkl"
HYBRID_PATH = "data/model/text_embeddings.csv"
GRAPH_TRAIN = "data/features/graph_features_train.csv"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

print("📦 Loading model and data...")

# Load model package (RF-only architecture)
model_package = joblib.load(MODEL_PATH)
rf = model_package['rf']
sna_scaler = model_package['sna_scaler']  # ✅ Correct scaler name
best_thresh = model_package['threshold']
feature_info = model_package['feature_info']
ensemble_type = model_package.get('ensemble_type', 'rf_only')

print(f"✅ Loaded model: {ensemble_type}")
print(f"📊 Threshold: {best_thresh:.4f}")

# --- Load embeddings safely ---
print("\n📂 Loading embeddings...")
with open(HYBRID_PATH, "rb") as f:
    raw_bytes = f.read()

text_df = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)
print(f"✅ Loaded embeddings: {text_df.shape}")

# --- Clean non-numeric values in embedding columns ---
numeric_cols = [c for c in text_df.columns if c.isdigit()]
text_df[numeric_cols] = text_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
text_df[numeric_cols] = text_df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

# --- Load graph features ---
print("\n📂 Loading graph features...")
graph_train = pd.read_csv(GRAPH_TRAIN)

# Merge to get full feature set
train_merged = pd.merge(text_df, graph_train, left_on="id", right_on="node_id", how="inner")
print(f"✅ Merged data: {train_merged.shape}")

# --- Sample subset for SHAP (to save memory) ---
sample_size = min(500, len(train_merged))
sampled_data = train_merged.sample(sample_size, random_state=42)
print(f"📊 Using {sample_size} samples for SHAP analysis")

# --- Extract features (same as training) ---
embedding_cols = feature_info['embedding_cols']
sna_cols = feature_info['sna_cols']

X_emb = sampled_data[embedding_cols].values
X_sna = sampled_data[[c for c in sna_cols if c in sampled_data.columns]].fillna(0).values

print(f"\n🔧 Feature extraction:")
print(f"   Embeddings: {X_emb.shape} (NOT scaled)")
print(f"   SNA features: {X_sna.shape} (will be scaled)")

# --- Scale ONLY SNA features (like in training) ---
X_sna_scaled = sna_scaler.transform(X_sna)

# --- Combine features ---
X_combined = np.hstack([X_emb, X_sna_scaled])
print(f"✅ Combined features: {X_combined.shape}")

# --- Create feature names for interpretability ---
feature_names = (
    [f"emb_{i}" for i in range(X_emb.shape[1])] + 
    [f"sna_{col}" for col in sna_cols if col in sampled_data.columns]
)

# --- Run SHAP explainability ---
print("\n🔍 Running SHAP explainability (this may take a few minutes)...")
print("   Creating TreeExplainer for RandomForest...")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_combined)

# For binary classification, shap_values might be a list [class_0, class_1]
# We want explanations for the "Fake" class (class 1)
if isinstance(shap_values, list):
    shap_values_fake = shap_values[1]  # Explanations for fake news (class 1)
else:
    shap_values_fake = shap_values

print("✅ SHAP values computed")

# --- Create DataFrame for better visualization ---
X_df = pd.DataFrame(X_combined, columns=feature_names)

# --- Save summary plot ---
print("\n📊 Creating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_fake, 
    X_df, 
    max_display=20,  # Show top 20 features
    show=False
)
plt.title("SHAP Feature Importance (Fake News Prediction)", fontsize=14, pad=20)
plt.tight_layout()
output_path = os.path.join(REPORT_DIR, "shap_summary.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved SHAP summary → {output_path}")

# --- Save bar plot (mean absolute SHAP values) ---
print("\n📊 Creating SHAP bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_fake, 
    X_df, 
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("Top 20 Features by Mean |SHAP Value|", fontsize=14, pad=20)
plt.tight_layout()
bar_path = os.path.join(REPORT_DIR, "shap_bar.png")
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved SHAP bar plot → {bar_path}")

# --- Calculate and display top features ---
print("\n📈 Top 10 Most Important Features:")
mean_abs_shap = np.abs(shap_values_fake).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[-10:][::-1]

for rank, idx in enumerate(top_indices, 1):
    feature_name = feature_names[idx]
    importance = mean_abs_shap[idx]
    print(f"   {rank:2d}. {feature_name:20s}: {importance:.6f}")

# --- Additional analysis: SNA vs Embedding importance ---
print("\n📊 Feature Group Importance:")
sna_start_idx = len(embedding_cols)
emb_importance = mean_abs_shap[:sna_start_idx].mean()
sna_importance = mean_abs_shap[sna_start_idx:].mean()

print(f"   Embedding features (avg): {emb_importance:.6f}")
print(f"   SNA features (avg):       {sna_importance:.6f}")

if sna_importance > emb_importance:
    print("   → SNA features are more important on average")
else:
    print("   → Embedding features are more important on average")

print("\n🎉 SHAP analysis complete!")
print(f"📁 Results saved to: {REPORT_DIR}/")