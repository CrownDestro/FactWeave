#!/usr/bin/env python3
"""
Data Diagnostic Script - Find why probabilities collapse
---------------------------------------------------------
Checks embeddings, features, and model inputs for issues
"""

import pandas as pd
import numpy as np
import io

print("="*70)
print("🔍 DIAGNOSING DATA QUALITY ISSUES")
print("="*70)

# ======================
# 1. Check Embeddings
# ======================
print("\n📊 1. CHECKING TEXT EMBEDDINGS")
print("-"*70)

try:
    with open("data/model/text_embeddings.csv", "rb") as f:
        raw_bytes = f.read()
    
    df = pd.read_csv(
        io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
        on_bad_lines="skip",
        engine="python"
    )
    
    print(f"✅ Shape: {df.shape}")
    print(f"✅ Columns: {df.columns.tolist()[:10]}...")
    
    # Get embedding columns
    embedding_cols = [c for c in df.columns if str(c).isdigit()]
    print(f"✅ Embedding dimensions: {len(embedding_cols)}")
    
    # Check embedding statistics
    emb_data = df[embedding_cols].values
    print(f"\n📈 Embedding Statistics:")
    print(f"   Min: {emb_data.min():.6f}")
    print(f"   Max: {emb_data.max():.6f}")
    print(f"   Mean: {emb_data.mean():.6f}")
    print(f"   Std: {emb_data.std():.6f}")
    
    # Check for NaN/Inf
    nan_count = np.isnan(emb_data).sum()
    inf_count = np.isinf(emb_data).sum()
    print(f"   NaN values: {nan_count}")
    print(f"   Inf values: {inf_count}")
    
    # Check L2 norms (should be ~1 for SBERT)
    norms = np.linalg.norm(emb_data, axis=1)
    print(f"\n🔍 L2 Norms (should be ~1.0 for SBERT):")
    print(f"   Min norm: {norms.min():.6f}")
    print(f"   Max norm: {norms.max():.6f}")
    print(f"   Mean norm: {norms.mean():.6f}")
    
    if norms.mean() < 0.1:
        print("   ⚠️  WARNING: Norms are too small! Embeddings might be corrupted.")
    elif norms.mean() > 10:
        print("   ⚠️  WARNING: Norms are too large! Embeddings might be corrupted.")
    else:
        print("   ✅ Norms look reasonable")
    
    # Check variance across dimensions
    variances = emb_data.var(axis=0)
    print(f"\n📊 Variance across dimensions:")
    print(f"   Min variance: {variances.min():.6f}")
    print(f"   Max variance: {variances.max():.6f}")
    print(f"   Mean variance: {variances.mean():.6f}")
    
    if variances.max() < 0.001:
        print("   ❌ CRITICAL: Variance is too low! Embeddings have no information!")
    
    # Check if all embeddings are identical
    unique_rows = len(np.unique(emb_data, axis=0))
    print(f"\n🔢 Unique embeddings: {unique_rows} / {len(emb_data)}")
    if unique_rows < len(emb_data) * 0.9:
        print(f"   ⚠️  WARNING: Many duplicate embeddings ({100*(1-unique_rows/len(emb_data)):.1f}% duplicates)")
    
    # Sample embeddings
    print(f"\n📋 Sample embedding values (first 10 dims):")
    print(emb_data[0, :10])
    
except Exception as e:
    print(f"❌ Error reading embeddings: {e}")
    import traceback
    traceback.print_exc()

# ======================
# 2. Check Graph Features
# ======================
print("\n" + "="*70)
print("🕸️  2. CHECKING GRAPH FEATURES")
print("-"*70)

try:
    train_graph = pd.read_csv("data/features/graph_features_train.csv")
    test_graph = pd.read_csv("data/features/graph_features_test.csv")
    
    print(f"✅ Train shape: {train_graph.shape}")
    print(f"✅ Test shape: {test_graph.shape}")
    print(f"\n📋 Columns: {train_graph.columns.tolist()}")
    
    print(f"\n📊 Train Graph Feature Statistics:")
    print(train_graph.describe().T)
    
    # Check for zero variance features
    variances = train_graph.var()
    zero_var_features = variances[variances == 0].index.tolist()
    if zero_var_features:
        print(f"\n⚠️  WARNING: Zero-variance features: {zero_var_features}")
    
    # Check correlations with label
    if 'label_propagated' in train_graph.columns:
        correlations = train_graph.corr()['label_propagated'].sort_values(ascending=False)
        print(f"\n📈 Correlations with label_propagated:")
        print(correlations)
        
        if abs(correlations.drop('label_propagated')).max() < 0.05:
            print("   ⚠️  WARNING: Very weak correlations! Features may not be informative.")
    
except Exception as e:
    print(f"❌ Error reading graph features: {e}")

# ======================
# 3. Check Combined Data
# ======================
print("\n" + "="*70)
print("🔄 3. CHECKING COMBINED FEATURES")
print("-"*70)

try:
    # Merge like in training
    text_df = df
    train_df = text_df[text_df["id"].isin(train_graph["node_id"])]
    train_merged = pd.merge(train_df, train_graph, left_on="id", right_on="node_id", how="inner")
    
    print(f"✅ Merged shape: {train_merged.shape}")
    
    # Check class distribution
    print(f"\n📊 Class Distribution:")
    print(train_merged['label'].value_counts())
    
    # Extract features like in training
    X_emb = train_merged[embedding_cols].values
    sna_cols = ["degree", "pagerank", "clustering", "community", 
                "closeness", "eigenvector", "label_propagated"]
    X_sna = train_merged[[c for c in sna_cols if c in train_merged.columns]].fillna(0).values
    
    print(f"\n📏 Feature shapes:")
    print(f"   Embeddings: {X_emb.shape}")
    print(f"   SNA: {X_sna.shape}")
    
    print(f"\n📊 Combined feature ranges:")
    X_combined = np.hstack([X_emb, X_sna])
    print(f"   Combined shape: {X_combined.shape}")
    print(f"   Min: {X_combined.min():.6f}")
    print(f"   Max: {X_combined.max():.6f}")
    print(f"   Mean: {X_combined.mean():.6f}")
    print(f"   Std: {X_combined.std():.6f}")
    
    # Check if features are informative
    y = train_merged['label'].values
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    print(f"\n🧪 Quick test with simple RandomForest:")
    rf_test = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_test.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = rf_test.predict(X_val)
    y_proba = rf_test.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
        print(f"   Accuracy: {acc:.3f}")
        print(f"   AUC: {auc:.3f}")
        
        print(f"\n📊 Probability distribution from test RF:")
        print(f"   Min: {y_proba.min():.6f}")
        print(f"   Max: {y_proba.max():.6f}")
        print(f"   Mean: {y_proba.mean():.6f}")
        
        if y_proba.max() < 0.1:
            print("   ❌ CRITICAL: Even simple RF has collapsed probabilities!")
            print("   This suggests the features themselves are the problem.")
        else:
            print("   ✅ Simple RF produces reasonable probabilities")
    except:
        print(f"   Accuracy: {acc:.3f}")
        print(f"   AUC: Could not compute")
    
except Exception as e:
    print(f"❌ Error in combined analysis: {e}")
    import traceback
    traceback.print_exc()

# ======================
# 4. Recommendations
# ======================
print("\n" + "="*70)
print("💡 RECOMMENDATIONS")
print("="*70)

print("""
Based on the diagnostics above, the issue is likely one of:

1. ❌ Embeddings are corrupted (check L2 norms)
2. ❌ Embeddings have no variance (check variance stats)
3. ❌ Too many duplicate embeddings
4. ❌ Graph features are not informative
5. ❌ Class imbalance is too severe

Next steps:
- If L2 norms are wrong → Regenerate embeddings
- If variance is too low → Check source text quality
- If simple RF works but ensemble fails → Problem is in ensemble logic
- If nothing works → Dataset may be fundamentally flawed
""")

print("\n🔍 Run this script to see detailed diagnostics!")