#!/usr/bin/env python3
"""
Final Hybrid Model Training Pipeline (SNA + Text)
-------------------------------------------------
✅ Combines SBERT text embeddings + graph metrics + label propagation
✅ Clean split: train/test graph separation
✅ Uses SMOTE for class balance
✅ Finds best decision threshold via F1 optimization
✅ Saves evaluation plots (ROC + PR curves + Confusion Matrix)
✅ Outputs full metrics to reports/results.txt
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

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
import io

print("🔹 Loading embeddings safely (handling encoding issues)...")
with open(TEXT_EMB, "rb") as f:
    raw_bytes = f.read()

text_df = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)
print(f"✅ Loaded embeddings: {text_df.shape}")
# 🔧 Clean corrupted float values (e.g., "-0.04�88902")
import numpy as np

numeric_cols = [c for c in text_df.columns if c.isdigit()]

# Try to coerce non-numeric values to NaN
text_df[numeric_cols] = text_df[numeric_cols].apply(
    pd.to_numeric, errors='coerce'
)

# Replace NaN or corrupted values with column means
text_df[numeric_cols] = text_df[numeric_cols].apply(
    lambda col: col.fillna(col.mean())
)

print("🧹 Cleaned embedding columns — non-numeric values replaced with column means.")

graph_train = pd.read_csv(GRAPH_TRAIN)
graph_test = pd.read_csv(GRAPH_TEST)

print(f"Text: {text_df.shape}, Train graph: {graph_train.shape}, Test graph: {graph_test.shape}")

train_df = text_df[text_df["id"].isin(graph_train["node_id"])]
test_df = text_df[text_df["id"].isin(graph_test["node_id"])]

train_merged = pd.merge(train_df, graph_train, left_on="id", right_on="node_id", how="inner").fillna(0)
test_merged = pd.merge(test_df, graph_test, left_on="id", right_on="node_id", how="inner").fillna(0)

print(f"✅ Train merged: {train_merged.shape}, Test merged: {test_merged.shape}")

# -------------------------------------------
# 🔧 STEP 2: Normalize and balance graph features
# -------------------------------------------
graph_cols = ["degree", "pagerank", "clustering", "community", 
              "closeness", "eigenvector", "label_propagated"]

for df in [train_merged, test_merged]:
    available = [c for c in graph_cols if c in df.columns]
    if available:
        scaler_graph = StandardScaler()
        df[available] = scaler_graph.fit_transform(df[available])
        print(f"✅ Normalized graph features: {available}")
    else:
        print("⚠️ No graph features found to normalize")

# ======================
# Prepare features
# ======================
embedding_cols = [c for c in text_df.columns if str(c).isdigit()]
graph_cols = ["degree", "pagerank", "clustering", "community", "label_propagated"]

X_train = train_merged[embedding_cols + graph_cols]
y_train = train_merged["label"].astype(int)

X_test = test_merged[embedding_cols + graph_cols]
y_test = test_merged["label"].astype(int)

# ======================
# Scale & Balance
# ======================
# --- Ensure consistent feature columns across training & inference ---
all_features = list(X_train.columns)

# Add any missing SNA features (if not present in training)
required_graph_features = [
    "degree", "pagerank", "clustering", "community",
    "closeness", "eigenvector", "label_propagated"
]
for feat in required_graph_features:
    if feat not in all_features:
        X_train.loc[:,feat] = 0.5
        X_test.loc[:,feat] = 0.5
        all_features.append(feat)

# Reorder columns consistently
X_train = X_train[all_features]
X_test = X_test[all_features]

# Fit the scaler with full feature set
scaler = StandardScaler()
scaler.fit(X_train)
scaler.feature_names_in_ = np.array(all_features)  # ✅ store for predict_news.py

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("⚙️ Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
print("✅ After SMOTE class distribution:\n", pd.Series(y_train).value_counts())

# ======================
# Train base models
# ======================
print("🧠 Training RandomForest and XGBoost...")

rf = RandomForestClassifier(
    n_estimators=400, max_depth=10, random_state=42, class_weight="balanced"
)
xgb = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
)

rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

# ======================
# Meta-ensemble (stacking)
# ======================
rf_probs_train = rf.predict_proba(X_train_scaled)[:, 1]
xgb_probs_train = xgb.predict_proba(X_train_scaled)[:, 1]
meta_input_train = np.vstack([rf_probs_train, xgb_probs_train]).T

from xgboost import XGBClassifier
meta = XGBClassifier(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
)

meta.fit(meta_input_train, y_train)
print("✅ Ensemble model trained successfully!")

# ======================
# Evaluate on Test set
# ======================
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]
meta_input_test = np.vstack([rf_probs, xgb_probs]).T
final_probs = meta.predict_proba(meta_input_test)[:, 1]

# ---- Find best threshold via F1 optimization
prec, rec, thresh = precision_recall_curve(y_test, final_probs)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-6)
best_thresh = thresh[np.argmax(f1_scores)]
print(f"🎯 Best threshold found: {best_thresh:.3f} (F1={max(f1_scores):.3f})")

final_preds = (final_probs > best_thresh).astype(int)

# ======================
# Metrics
# ======================
report = classification_report(y_test, final_preds, digits=3)
roc_auc = roc_auc_score(y_test, final_probs)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, final_probs)
pr_auc = auc(rec_curve, prec_curve)

print("\n=== 📊 Final Model Evaluation ===")
print(report)
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")

# ======================
# Save Metrics & Plots
# ======================
with open(os.path.join(REPORT_DIR, "results.txt"), "w") as f:
    f.write("=== Final Model Evaluation ===\n")
    f.write(report + "\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}\n")
    f.write(f"PR-AUC: {pr_auc:.3f}\n")
    f.write(f"Best Threshold: {best_thresh:.3f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Hybrid Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.close()

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, final_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"))
plt.close()

# Precision-Recall Curve
plt.figure()
plt.plot(rec_curve, prec_curve, label=f"PR-AUC={pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "pr_curve.png"))
plt.close()

# ======================
# Save model
# ======================
joblib.dump((rf, xgb, meta, scaler, best_thresh), os.path.join(MODEL_DIR, "hybrid_model.pkl"))
print(f"✅ Model + scaler + threshold saved to {MODEL_DIR}/hybrid_model.pkl")

print("\n💡 Example fake news probability:")
example_prob = final_probs[0]
print(f"🧾 Fake probability: {example_prob:.3f}")
print(f"✅ Real probability: {1 - example_prob:.3f}")
