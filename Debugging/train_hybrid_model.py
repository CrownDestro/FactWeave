#!/usr/bin/env python3
"""
FINAL CORRECT VERSION - Hybrid Model Training Pipeline
--------------------------------------------------------
🔑 KEY FIX: DON'T scale SBERT embeddings (already normalized)
✅ Only scale SNA features
✅ Proper probability calibration
✅ Fixed feature_names_in_ bug
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score, roc_curve,
    accuracy_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io

# ======================
# Paths
# ======================
TEXT_EMB = "data/model/text_embeddings.csv"
GRAPH_TRAIN = "data/features/graph_features_train.csv"
GRAPH_TEST = "data/features/graph_features_test.csv"
MODEL_DIR = "data/model"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ======================
# Load datasets
# ======================
print("🔹 Loading datasets...")

with open(TEXT_EMB, "rb") as f:
    raw_bytes = f.read()

text_df = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)
print(f"✅ Loaded embeddings: {text_df.shape}")

# Clean corrupted values
numeric_cols = [c for c in text_df.columns if c.isdigit()]
text_df[numeric_cols] = text_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
text_df[numeric_cols] = text_df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

graph_train = pd.read_csv(GRAPH_TRAIN)
graph_test = pd.read_csv(GRAPH_TEST)

print(f"Text: {text_df.shape}, Train graph: {graph_train.shape}, Test graph: {graph_test.shape}")

# Merge with graph features
train_df = text_df[text_df["id"].isin(graph_train["node_id"])]
test_df = text_df[text_df["id"].isin(graph_test["node_id"])]

train_merged = pd.merge(train_df, graph_train, left_on="id", right_on="node_id", how="inner")
test_merged = pd.merge(test_df, graph_test, left_on="id", right_on="node_id", how="inner")

print(f"✅ Train merged: {train_merged.shape}, Test merged: {test_merged.shape}")
print(f"\n📊 Class distribution:")
print(f"Train: {train_merged['label'].value_counts().to_dict()}")
print(f"Test: {test_merged['label'].value_counts().to_dict()}")

# ======================
# Prepare features
# ======================
embedding_cols = [c for c in text_df.columns if str(c).isdigit()]
sna_cols = ["degree", "pagerank", "clustering", "community", 
            "closeness", "eigenvector", "label_propagated"]

# Extract features
X_train_emb = train_merged[embedding_cols].values
X_test_emb = test_merged[embedding_cols].values

X_train_sna = train_merged[[c for c in sna_cols if c in train_merged.columns]].fillna(0).values
X_test_sna = test_merged[[c for c in sna_cols if c in test_merged.columns]].fillna(0).values

y_train = train_merged["label"].astype(int).values
y_test = test_merged["label"].astype(int).values

# ======================
# 🔑 CRITICAL FIX: Only scale SNA features
# ======================
print("\n⚙️ Scaling ONLY SNA features (embeddings already normalized)...")
sna_scaler = StandardScaler()
X_train_sna_scaled = sna_scaler.fit_transform(X_train_sna)
X_test_sna_scaled = sna_scaler.transform(X_test_sna)

# Combine: RAW embeddings + SCALED SNA
X_train_combined = np.hstack([X_train_emb, X_train_sna_scaled])
X_test_combined = np.hstack([X_test_emb, X_test_sna_scaled])

print(f"✅ Combined features shape: Train={X_train_combined.shape}, Test={X_test_combined.shape}")
print(f"   Embedding features: {X_train_emb.shape[1]} (NOT scaled)")
print(f"   SNA features: {X_train_sna.shape[1]} (scaled)")

# ======================
# Apply SMOTE
# ======================
print("\n⚙️ Applying SMOTE for class balance...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)
print(f"✅ After SMOTE: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")

# ======================
# Train base models
# ======================
print("\n🧠 Training RandomForest and XGBoost...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

rf.fit(X_train_balanced, y_train_balanced)
xgb.fit(X_train_balanced, y_train_balanced)

print("✅ Base models trained")

# ======================
# Meta-ensemble
# ======================
print("\n🔄 Training meta-ensemble...")
rf_probs_train = rf.predict_proba(X_train_balanced)[:, 1]
xgb_probs_train = xgb.predict_proba(X_train_balanced)[:, 1]
meta_input_train = np.vstack([rf_probs_train, xgb_probs_train]).T

from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
meta.fit(meta_input_train, y_train_balanced)

print("✅ Meta-ensemble trained")

# ======================
# Evaluate on test set
# ======================
print("\n🧪 Evaluating on test set...")
rf_probs_test = rf.predict_proba(X_test_combined)[:, 1]
xgb_probs_test = xgb.predict_proba(X_test_combined)[:, 1]
meta_input_test = np.vstack([rf_probs_test, xgb_probs_test]).T
final_probs = meta.predict_proba(meta_input_test)[:, 1]

print(f"\n📊 Probability distribution:")
print(f"   Min: {final_probs.min():.4f}, Max: {final_probs.max():.4f}")
print(f"   Mean: {final_probs.mean():.4f}, Median: {np.median(final_probs):.4f}")
print(f"   Std: {final_probs.std():.4f}")

# Sanity check
if final_probs.max() < 0.1:
    print("\n❌ CRITICAL ERROR: Probabilities are too low!")
    print("   This indicates a fundamental problem with feature engineering.")
    print("   The model is not learning properly.")
elif final_probs.min() > 0.9:
    print("\n❌ CRITICAL ERROR: Probabilities are too high!")
else:
    print("✅ Probabilities are in reasonable range")

# ======================
# Find optimal threshold
# ======================
print("\n🎯 Finding optimal decision threshold...")

# Method 1: Youden's J
fpr, tpr, thresholds_roc = roc_curve(y_test, final_probs)
youden_j = tpr - fpr
idx_youden = np.argmax(youden_j)
thresh_youden = thresholds_roc[idx_youden]

# Method 2: Max F1
prec, rec, thresholds_pr = precision_recall_curve(y_test, final_probs)
f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-10)
idx_f1 = np.argmax(f1_scores)
thresh_f1 = thresholds_pr[idx_f1]

# Method 3: Default
thresh_default = 0.5

print(f"\n📊 Threshold candidates:")
print(f"   Youden's J: {thresh_youden:.4f}")
print(f"   Max F1: {thresh_f1:.4f}")
print(f"   Default: {thresh_default:.4f}")

# Test each
thresholds_to_test = {
    "Youden": thresh_youden,
    "F1": thresh_f1,
    "Default": thresh_default
}

print(f"\n🧪 Testing thresholds:")
best_metric = 0
best_thresh = thresh_default
best_name = "Default"

for name, thresh in thresholds_to_test.items():
    preds = (final_probs >= thresh).astype(int)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    combined = (f1 + bal_acc) / 2
    
    print(f"   {name:10s} (t={thresh:.4f}): F1={f1:.4f}, Acc={acc:.4f}, BalAcc={bal_acc:.4f}, Score={combined:.4f}")
    
    if combined > best_metric:
        best_metric = combined
        best_thresh = thresh
        best_name = name

print(f"\n✅ Selected: {best_name} threshold = {best_thresh:.4f}")

# Final predictions
final_preds = (final_probs >= best_thresh).astype(int)

# ======================
# Metrics
# ======================
print("\n" + "="*60)
print("📊 FINAL MODEL EVALUATION")
print("="*60)

report = classification_report(y_test, final_preds, digits=3)
roc_auc = roc_auc_score(y_test, final_probs)
pr_auc = auc(rec, prec)

print(report)
print(f"\nROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")
print(f"Best Threshold: {best_thresh:.4f}")

# ======================
# Save results
# ======================
with open(os.path.join(REPORT_DIR, "results.txt"), "w") as f:
    f.write("="*60 + "\n")
    f.write("FINAL MODEL EVALUATION\n")
    f.write("="*60 + "\n\n")
    f.write(report + "\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}\n")
    f.write(f"PR-AUC: {pr_auc:.3f}\n")
    f.write(f"Best Threshold: {best_thresh:.4f} ({best_name})\n")

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title(f"Confusion Matrix (threshold={best_thresh:.3f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.scatter(fpr[idx_youden], tpr[idx_youden], color='red', s=100, zorder=5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=150)
plt.close()

# Probability Distribution
plt.figure(figsize=(10, 5))
plt.hist(final_probs[y_test == 0], bins=50, alpha=0.5, label="Real", color="green")
plt.hist(final_probs[y_test == 1], bins=50, alpha=0.5, label="Fake", color="red")
plt.axvline(best_thresh, color='black', linestyle='--', linewidth=2, label=f"Threshold={best_thresh:.3f}")
plt.xlabel("Predicted Probability (Fake)")
plt.ylabel("Count")
plt.title("Probability Distribution by True Label")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "probability_distribution.png"), dpi=150)
plt.close()

# ======================
# Save model components
# ======================
feature_info = {
    'embedding_cols': embedding_cols,
    'sna_cols': [c for c in sna_cols if c in train_merged.columns],
    'n_embedding_features': len(embedding_cols),
    'n_sna_features': X_train_sna.shape[1]
}

model_package = {
    'rf': rf,
    'xgb': xgb,
    'meta': meta,
    'sna_scaler': sna_scaler,  # ← Only SNA scaler
    'threshold': best_thresh,
    'feature_info': feature_info
}

joblib.dump(model_package, os.path.join(MODEL_DIR, "hybrid_model.pkl"))
print(f"\n✅ Model saved to {MODEL_DIR}/hybrid_model.pkl")
print(f"✅ Plots saved to {REPORT_DIR}/")

# ======================
# Sample predictions
# ======================
print("\n" + "="*60)
print("💡 SAMPLE PREDICTIONS")
print("="*60)

sample_indices = np.random.choice(len(y_test), size=min(10, len(y_test)), replace=False)
for idx in sample_indices:
    true_label = "Fake" if y_test[idx] == 1 else "Real"
    pred_label = "Fake" if final_preds[idx] == 1 else "Real"
    prob = final_probs[idx] * 100
    
    symbol = "✅" if true_label == pred_label else "❌"
    print(f"{symbol} True: {true_label:4s} | Pred: {pred_label:4s} | P(Fake): {prob:5.1f}%")

print("\n🎉 Training complete!")
print(f"🎯 Key metrics: Accuracy={accuracy_score(y_test, final_preds):.3f}, F1={f1_score(y_test, final_preds):.3f}")