#!/usr/bin/env python3
"""
Find examples where model predicts REAL with high confidence
(Fake probability < threshold)
"""

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import io

print("="*70)
print("🔍 FINDING CONFIDENT 'REAL' NEWS PREDICTIONS")
print("="*70)

# Load model
model_package = joblib.load("data/model/hybrid_model.pkl")
rf = model_package['rf']
sna_scaler = model_package['sna_scaler']
threshold = model_package['threshold']
feature_info = model_package['feature_info']

print(f"\n📊 Model threshold: {threshold:.4f} ({threshold*100:.2f}%)")
print(f"   Predictions with Fake < {threshold*100:.2f}% → Classified as REAL")

# Load some articles to test
with open("data/model/text_embeddings.csv", "rb") as f:
    raw_bytes = f.read()

text_df = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)

# Load original articles for text
articles_df = pd.read_csv("data/processed/articles_clean.csv")

# Merge to get text
merged = pd.merge(text_df, articles_df[['id', 'text']], on='id', how='inner')

print(f"\n✅ Loaded {len(merged)} articles")

# Extract features
embedding_cols = [c for c in text_df.columns if str(c).isdigit()]
sna_cols = ["degree", "pagerank", "clustering", "community", 
            "closeness", "eigenvector", "label_propagated"]

# Sample articles to test
print(f"\n🔄 Testing sample articles...")
sample_size = min(500, len(merged))
sample_df = merged.sample(sample_size, random_state=42)

encoder = SentenceTransformer("all-mpnet-base-v2")

results = []
for idx, row in sample_df.iterrows():
    # Get embedding from dataframe (already computed)
    embedding = row[embedding_cols].values.reshape(1, -1)
    
    # Neutral SNA features
    n_sna = feature_info['n_sna_features']
    sna_features = np.array([[0.5] * n_sna])
    sna_scaled = sna_scaler.transform(sna_features)
    
    # Combine and predict
    X_combined = np.hstack([embedding, sna_scaled])
    prob_fake = float(rf.predict_proba(X_combined)[:, 1][0])
    
    results.append({
        'text': row['text'],
        'true_label': row['label'],
        'prob_fake': prob_fake,
        'prob_real': 1 - prob_fake,
        'predicted': "Fake" if prob_fake > threshold else "Real"
    })

results_df = pd.DataFrame(results)

# Find confident REAL predictions
confident_real = results_df[results_df['prob_fake'] < threshold].copy()
confident_real = confident_real.sort_values('prob_fake')

print(f"\n📊 Results from {sample_size} articles:")
print(f"   Predicted as REAL: {len(confident_real)} ({len(confident_real)/sample_size*100:.1f}%)")
print(f"   Predicted as FAKE: {len(results_df) - len(confident_real)} ({(len(results_df) - len(confident_real))/sample_size*100:.1f}%)")

# Show top examples (most confident REAL predictions)
print("\n" + "="*70)
print("🏆 TOP 15 MOST CONFIDENT 'REAL' PREDICTIONS")
print("="*70)
print("(Lowest Fake probability = Most confident it's REAL)\n")

for i, row in confident_real.head(15).iterrows():
    true_label = "REAL" if row['true_label'] == 0 else "FAKE"
    correct = "✅" if (row['predicted'] == "Real" and row['true_label'] == 0) or \
                      (row['predicted'] == "Fake" and row['true_label'] == 1) else "❌"
    
    print(f"{correct} Fake: {row['prob_fake']*100:5.2f}% | Real: {row['prob_real']*100:5.2f}% | True: {true_label}")
    print(f"   Text: {row['text'][:150]}...")
    print()

# Show some that are just barely REAL (close to threshold)
borderline_real = results_df[
    (results_df['prob_fake'] < threshold) & 
    (results_df['prob_fake'] > threshold - 0.05)
].sort_values('prob_fake', ascending=False)

if len(borderline_real) > 0:
    print("\n" + "="*70)
    print("⚠️  BORDERLINE 'REAL' PREDICTIONS")
    print("="*70)
    print(f"(Fake probability {threshold-0.05:.2f} - {threshold:.2f} = Just barely classified as REAL)\n")
    
    for i, row in borderline_real.head(10).iterrows():
        true_label = "REAL" if row['true_label'] == 0 else "FAKE"
        correct = "✅" if (row['predicted'] == "Real" and row['true_label'] == 0) or \
                          (row['predicted'] == "Fake" and row['true_label'] == 1) else "❌"
        
        print(f"{correct} Fake: {row['prob_fake']*100:5.2f}% | Real: {row['prob_real']*100:5.2f}% | True: {true_label}")
        print(f"   Text: {row['text'][:150]}...")
        print()

# Summary statistics
print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)

print(f"\nConfident REAL predictions (Fake < {threshold*100:.2f}%):")
print(f"   Count: {len(confident_real)}")
print(f"   Fake probability range: {confident_real['prob_fake'].min()*100:.2f}% - {confident_real['prob_fake'].max()*100:.2f}%")
print(f"   Mean Fake probability: {confident_real['prob_fake'].mean()*100:.2f}%")

if len(confident_real) > 0:
    correct_real = confident_real[confident_real['true_label'] == 0]
    incorrect_real = confident_real[confident_real['true_label'] == 1]
    
    print(f"\n   Actually REAL: {len(correct_real)} (✅ correct)")
    print(f"   Actually FAKE: {len(incorrect_real)} (❌ false negatives)")
    
    if len(correct_real) > 0:
        print(f"\n   Correct REAL predictions - Fake prob range: {correct_real['prob_fake'].min()*100:.2f}% - {correct_real['prob_fake'].max()*100:.2f}%")

print("\n" + "="*70)