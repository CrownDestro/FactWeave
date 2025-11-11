#!/usr/bin/env python3
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
    df = pd.read_csv(INPUT)
    print("Loaded articles:", len(df))
    # model choice: multi-qa is good for factual QA and semantics
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = df["text"].astype(str).tolist()
    # encode in batches
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    emb_df = pd.DataFrame(embeddings)
    emb_df["id"] = df["id"]
    emb_df["label"] = df["label"]
    emb_df["source"] = df["source"]
    emb_df.to_csv(OUT_FILE, index=False)
    print("Saved embeddings to", OUT_FILE)

if __name__ == "__main__":
    main()
