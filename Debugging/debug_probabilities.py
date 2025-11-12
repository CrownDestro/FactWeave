#!/usr/bin/env python3
"""
Debug where probabilities collapse in the pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import io

print("="*70)
print("🔍 DEBUGGING PROBABILITY COLLAPSE")
print("="*70)

# Load data
with open("data/model/text_embeddings.csv", "rb") as f:
    raw_bytes = f.read()

text_df = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)

graph_train = pd.read_csv("data/features/graph_features_train.csv")
graph_test = pd.read_csv("data/features/graph_features_test.csv")

train_df = text_df[text_df["id"].isin(graph_train["node_id"])]
test_df = text_df[text_df["id"].isin(graph_test["node_id"])]

train_merged = pd.merge(train_df, graph_train, left_on="id", right_on="node_id", how="inner")
test_merged = pd.merge(test_df, graph_test, left_on="id", right_on="node_id", how="inner")

# Extract features
embedding_cols = [c for c in text_df.columns if str(c).isdigit()]
sna_cols = ["degree", "pagerank", "clustering", "community", 
            "closeness", "eigenvector", "label_propagated"]

X_train_emb = train_merged[embedding_cols].values
X_test_emb = test_merged[embedding_cols].values

X_train_sna = train_merged[[c for c in sna_cols if c in train_merged.columns]].fillna(0).values
X_test_sna = test_merged[[c for c in sna_cols if c in test_merged.columns]].fillna(0).values

y_train = train_merged["label"].astype(int).values
y_test = test_merged["label"].astype(int).values

# Scale SNA only
sna_scaler = StandardScaler()
X_train_sna_scaled = sna_scaler.fit_transform(X_train_sna)
X_test_sna_scaled = sna_scaler.transform(X_test_sna)

# Combine
X_train_combined = np.hstack([X_train_emb, X_train_sna_scaled])
X_test_combined = np.hstack([X_test_emb, X_test_sna_scaled])

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)

print(f"\n✅ Data loaded and preprocessed")
print(f"   Train: {X_train_balanced.shape}, Test: {X_test_combined.shape}")

# ======================
# TEST 1: Just RandomForest
# ======================
print("\n" + "="*70)
print("TEST 1: RandomForest Only")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)
rf_probs = rf.predict_proba(X_test_combined)[:, 1]

print(f"RF Probability Distribution:")
print(f"   Min: {rf_probs.min():.6f}")
print(f"   Max: {rf_probs.max():.6f}")
print(f"   Mean: {rf_probs.mean():.6f}")
print(f"   Std: {rf_probs.std():.6f}")

# ======================
# TEST 2: Just XGBoost
# ======================
print("\n" + "="*70)
print("TEST 2: XGBoost Only")
print("="*70)

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
xgb.fit(X_train_balanced, y_train_balanced)
xgb_probs = xgb.predict_proba(X_test_combined)[:, 1]

print(f"XGB Probability Distribution:")
print(f"   Min: {xgb_probs.min():.6f}")
print(f"   Max: {xgb_probs.max():.6f}")
print(f"   Mean: {xgb_probs.mean():.6f}")
print(f"   Std: {xgb_probs.std():.6f}")

# ======================
# TEST 3: Simple Average
# ======================
print("\n" + "="*70)
print("TEST 3: Simple Average of RF + XGB")
print("="*70)

avg_probs = (rf_probs + xgb_probs) / 2

print(f"Averaged Probability Distribution:")
print(f"   Min: {avg_probs.min():.6f}")
print(f"   Max: {avg_probs.max():.6f}")
print(f"   Mean: {avg_probs.mean():.6f}")
print(f"   Std: {avg_probs.std():.6f}")

# ======================
# TEST 4: Meta-Ensemble (LogisticRegression)
# ======================
print("\n" + "="*70)
print("TEST 4: Meta-Ensemble (LogisticRegression)")
print("="*70)

# Get training predictions
rf_probs_train = rf.predict_proba(X_train_balanced)[:, 1]
xgb_probs_train = xgb.predict_proba(X_train_balanced)[:, 1]
meta_input_train = np.vstack([rf_probs_train, xgb_probs_train]).T

print(f"Meta-model training input statistics:")
print(f"   RF probs  - Min: {rf_probs_train.min():.6f}, Max: {rf_probs_train.max():.6f}")
print(f"   XGB probs - Min: {xgb_probs_train.min():.6f}, Max: {xgb_probs_train.max():.6f}")

from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
meta.fit(meta_input_train, y_train_balanced)

# Test predictions
meta_input_test = np.vstack([rf_probs, xgb_probs]).T
meta_probs = meta.predict_proba(meta_input_test)[:, 1]

print(f"\nMeta-Ensemble Probability Distribution:")
print(f"   Min: {meta_probs.min():.6f}")
print(f"   Max: {meta_probs.max():.6f}")
print(f"   Mean: {meta_probs.mean():.6f}")
print(f"   Std: {meta_probs.std():.6f}")

print(f"\nMeta-model coefficients:")
print(f"   RF weight: {meta.coef_[0][0]:.6f}")
print(f"   XGB weight: {meta.coef_[0][1]:.6f}")
print(f"   Intercept: {meta.intercept_[0]:.6f}")

# ======================
# ANALYSIS
# ======================
print("\n" + "="*70)
print("📊 ANALYSIS")
print("="*70)

if rf_probs.max() > 0.5 and xgb_probs.max() > 0.5:
    print("✅ Individual models produce reasonable probabilities")
    
    if avg_probs.max() > 0.5:
        print("✅ Simple average produces reasonable probabilities")
    else:
        print("❌ Simple average collapses probabilities (shouldn't happen)")
    
    if meta_probs.max() < 0.1:
        print("❌ Meta-ensemble collapses probabilities!")
        print("\n🔍 ROOT CAUSE: LogisticRegression meta-model is the problem")
        print("\n💡 SOLUTION: Use simple averaging instead of meta-ensemble")
    else:
        print("✅ Meta-ensemble produces reasonable probabilities")
        
else:
    print("❌ Individual models already have collapsed probabilities")
    print("\n🔍 ROOT CAUSE: RandomForest/XGBoost are not learning properly")
    print("\n💡 Possible reasons:")
    print("   1. Features have no discriminative power")
    print("   2. Training data is too noisy")
    print("   3. Model hyperparameters are wrong")
    print("   4. SMOTE is creating unrealistic samples")

# ======================
# RECOMMENDATION
# ======================
print("\n" + "="*70)
print("💡 RECOMMENDATION")
print("="*70)

if rf_probs.max() > 0.5:
    print("\n✅ USE SIMPLE AVERAGING instead of meta-ensemble")
    print("   Replace train_hybrid_model.py with the ROBUST version")
    print("   (It uses simple averaging: (RF + XGB) / 2)")
else:
    print("\n🔧 Try these fixes:")
    print("   1. Train on embeddings ONLY (no SNA features)")
    print("   2. Use stronger model settings (more trees, deeper)")
    print("   3. Remove SMOTE (use class_weight instead)")
    print("   4. Check if source text data is meaningful")