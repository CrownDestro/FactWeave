#!/usr/bin/env python3
"""
FIXED Embedding Generation
---------------------------
Ensures all embeddings are proper float64 arrays
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT = "data/processed/articles_clean.csv"
OUT_DIR = "data/model"
OUT_FILE = os.path.join(OUT_DIR, "text_embeddings.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("🔹 Loading articles...")
    df = pd.read_csv(INPUT)
    print(f"✅ Loaded articles: {len(df)}")
    
    # Verify text column exists and has content
    if 'text' not in df.columns:
        print("❌ Error: 'text' column not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Clean text data
    df['text'] = df['text'].fillna('').astype(str)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10]
    print(f"✅ After removing empty texts: {len(df)}")
    
    # Check sample texts
    print("\n📄 Sample texts:")
    for i in range(min(3, len(df))):
        print(f"   {i+1}. {df.iloc[i]['text'][:80]}...")
    
    # Load model
    print("\n🧠 Loading SBERT model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Encode
    texts = df["text"].astype(str).tolist()
    print(f"\n🔄 Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True  # ✅ Explicitly normalize
    )
    
    print(f"\n✅ Generated embeddings shape: {embeddings.shape}")
    print(f"   Embedding dtype: {embeddings.dtype}")
    print(f"   Min: {embeddings.min():.6f}")
    print(f"   Max: {embeddings.max():.6f}")
    print(f"   Mean: {embeddings.mean():.6f}")
    
    # Verify no NaN or Inf
    if np.isnan(embeddings).any():
        print("⚠️  WARNING: Embeddings contain NaN values!")
        embeddings = np.nan_to_num(embeddings, 0.0)
    
    if np.isinf(embeddings).any():
        print("⚠️  WARNING: Embeddings contain Inf values!")
        embeddings = np.nan_to_num(embeddings, 0.0)
    
    # Create DataFrame with PROPER data types
    print("\n💾 Creating DataFrame...")
    
    # Convert to DataFrame with explicit float64 dtype
    emb_df = pd.DataFrame(
        embeddings.astype(np.float64),  # ✅ Force float64
        columns=[str(i) for i in range(embeddings.shape[1])]
    )
    
    # Add metadata columns
    emb_df["id"] = df["id"].values
    emb_df["label"] = df["label"].astype(int).values
    emb_df["source"] = df["source"].astype(str).values
    
    # Verify data types
    print("\n🔍 Verifying data types...")
    embedding_cols = [str(i) for i in range(embeddings.shape[1])]
    
    for col in embedding_cols[:5]:  # Check first 5
        if emb_df[col].dtype not in [np.float64, np.float32]:
            print(f"   ⚠️  Column {col} has wrong dtype: {emb_df[col].dtype}")
        else:
            print(f"   ✅ Column {col}: {emb_df[col].dtype}")
    
    # Final verification
    print("\n📊 Final DataFrame info:")
    print(f"   Shape: {emb_df.shape}")
    print(f"   Columns: {emb_df.columns.tolist()[:5]}... + metadata")
    print(f"   Memory usage: {emb_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save with proper settings
    print(f"\n💾 Saving to {OUT_FILE}...")
    emb_df.to_csv(
        OUT_FILE, 
        index=False,
        float_format='%.8f'  # ✅ Explicit float format
    )
    
    # Verify saved file
    print("\n🔍 Verifying saved file...")
    test_df = pd.read_csv(OUT_FILE, nrows=5)
    print(f"   Shape: {test_df.shape}")
    print(f"   Sample values from first embedding column:")
    print(f"   {test_df['0'].values[:3]}")
    
    if test_df['0'].dtype in [np.float64, np.float32]:
        print("   ✅ File saved correctly with float values!")
    else:
        print(f"   ❌ ERROR: Saved file has wrong dtype: {test_df['0'].dtype}")
    
    print("\n✅ Embeddings saved successfully!")
    print(f"📍 Location: {OUT_FILE}")

if __name__ == "__main__":
    main()