#!/usr/bin/env python3
"""
Preprocess GossipCop, PolitiFact, and CoAID datasets into a unified format.
Normalizes column names, removes duplicates, and merges all sources.
Saves final dataset to: data/processed/articles_clean.csv
"""

import os
import json
import pandas as pd
import glob

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/articles_clean.csv"
os.makedirs("data/processed", exist_ok=True)


# --------------------------
# 1️⃣ Load GossipCop dataset
# --------------------------
def load_gossipcop():
    fake_path = os.path.join(RAW_DIR, "gossipcop_fake.csv")
    real_path = os.path.join(RAW_DIR, "gossipcop_real.csv")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Normalize columns
    fake_df.columns = [c.lower().strip() for c in fake_df.columns]
    real_df.columns = [c.lower().strip() for c in real_df.columns]

    fake_df["label"] = 1
    real_df["label"] = 0

    # Find suitable text column
    possible_cols = ["content", "text", "body", "article", "news_title", "title", "statement"]
    text_col_fake = next((c for c in possible_cols if c in fake_df.columns), None)
    text_col_real = next((c for c in possible_cols if c in real_df.columns), None)

    if text_col_fake is None or text_col_real is None:
        raise KeyError("❌ Missing text column in GossipCop CSVs")

    fake_df = fake_df.rename(columns={text_col_fake: "text"})
    real_df = real_df.rename(columns={text_col_real: "text"})

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df[["text", "label"]].dropna().drop_duplicates()
    df["source"] = "gossipcop"
    df["id"] = ["gc_" + str(i) for i in range(len(df))]

    print(f"✅ Loaded GossipCop: {df.shape}")
    return df


# ---------------------------
# 2️⃣ Load PolitiFact dataset
# ---------------------------
def load_politifact():
    path = os.path.join(RAW_DIR, "politifact_factcheck_data.json")
    items = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                parts = line.split("}{")
                for p in parts:
                    if not p.startswith("{"):
                        p = "{" + p
                    if not p.endswith("}"):
                        p = p + "}"
                    try:
                        items.append(json.loads(p))
                    except:
                        continue

    df = pd.DataFrame(items)
    df.columns = [c.lower().strip() for c in df.columns]

    # ✅ Find usable text field
    for col in ["text", "content", "statement", "claim"]:
        if col in df.columns:
            df["text"] = df[col]
            break

    # ✅ Get correct label field
    label_col = None
    for c in ["label", "verdict", "rating", "label_text"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        print("⚠️ No label column found in PolitiFact; skipping fake/real classification.")
        df["label"] = 0
    else:
        fake_terms = ["false", "pants-fire", "barely-true", "mostly-false"]
        df["label"] = df[label_col].astype(str).str.lower().apply(
            lambda x: 1 if any(term in x for term in fake_terms) else 0
        )

    df = df[["text", "label"]].dropna().drop_duplicates()
    df["source"] = "politifact"
    df["id"] = ["pf_" + str(i) for i in range(len(df))]

    print(f"✅ Loaded PolitiFact: {df.shape}")
    print(df['label'].value_counts())
    return df



# ---------------------------
# 3️⃣ Load CoAID dataset
# ---------------------------
def load_coaid(path="data/raw/CoAID/news"):
    print("📥 Loading CoAID...")
    files = glob.glob(os.path.join(path, "*.csv"))
    print(f"Found {len(files)} files")

    dfs = []
    for f in files:
        filename = os.path.basename(f).lower()
        try:
            df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")

            # Normalize columns
            df.columns = [c.lower().strip() for c in df.columns]

            # Detect usable text column
            if "content" in df.columns:
                text_col = "content"
            elif "abstract" in df.columns:
                text_col = "abstract"
            elif "title" in df.columns:
                text_col = "title"
            else:
                print(f"⚠️ Skipping {filename} — no valid text column.")
                continue

            df["text"] = df[text_col].astype(str).fillna("")
            df = df[df["text"].str.len() > 30]  # remove short or empty texts

            df["label"] = 1 if "fake" in filename else 0
            df["source"] = "coaid"
            df["id"] = [f"coaid_{os.path.splitext(filename)[0]}_{i}" for i in range(len(df))]

            dfs.append(df[["text", "source", "id", "label"]])
            print(f"✅ Loaded {filename}: {len(df)} rows")

        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")

    if not dfs:
        print("⚠️ No CoAID files loaded.")
        return pd.DataFrame(columns=["text", "source", "id", "label"])

    merged = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded CoAID: {merged.shape}")
    print(merged['label'].value_counts())
    return merged


# ---------------------------
# 4️⃣ Merge All Datasets
# ---------------------------
if __name__ == "__main__":
    df_gc = load_gossipcop()
    df_pf = load_politifact()
    df_coaid = load_coaid()

    df_all = pd.concat([df_gc, df_pf, df_coaid], ignore_index=True)
    df_all = df_all.dropna(subset=["text"]).drop_duplicates(subset=["text"])

    print(f"✅ Final merged dataset: {df_all.shape}")
    df_all.to_csv(OUT_PATH, index=False)
    print(f"💾 Saved to {OUT_PATH}")
