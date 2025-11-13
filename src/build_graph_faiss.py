#!/usr/bin/env python3
"""
FIXED: Memory-efficient graph builder with proper encoding handling
-------------------------------------------------------------------
✅ Handles corrupted CSV files safely
✅ Train/test split before label propagation
✅ Proper error handling
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
import io

try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    print("⚠️ FAISS not found, using sklearn NearestNeighbors (slower).")

FEATURES_DIR = "data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

EMBED_PATH = "data/model/text_embeddings.csv"
print(f"📂 Loading embeddings from {EMBED_PATH}...")

#FIX: Safe loading with encoding handling
try:
    with open(EMBED_PATH, "rb") as f:
        raw_bytes = f.read()
    
    df = pd.read_csv(
        io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
        on_bad_lines="skip",
        engine="python"
    )
    print(f"Loaded embeddings with encoding fix: {df.shape}")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    print("🔧 Attempting to regenerate embeddings...")
    
    # Try to regenerate from source
    import sys
    sys.exit("Please run: python3 src/generate_embeddings.py")

# Clean numeric columns
embedding_cols = [c for c in df.columns if str(c).isdigit()]
df[embedding_cols] = df[embedding_cols].apply(pd.to_numeric, errors='coerce')
df[embedding_cols] = df[embedding_cols].apply(lambda col: col.fillna(col.mean()))

print(f"Cleaned embeddings: {df.shape}")

# Verify required columns
required_cols = ['id', 'label']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

# -------------------------------
# Train-Test Split (before graph)
# -------------------------------
print("📊 Splitting into train/test sets (80/20)...")
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df["label"]
)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
print(f"Test labels: {test_df['label'].value_counts().to_dict()}")

# Normalize embeddings
X_all = normalize(df[embedding_cols].values.astype(np.float32))
ids = df["id"].tolist()

print(f"✅ Normalized embeddings shape: {X_all.shape}")

# -------------------------------
# Build top-k similarity graph
# -------------------------------
k = 30
print(f"🔍 Building top-{k} nearest neighbor graph...")

if USE_FAISS:
    print("Using FAISS for fast similarity search...")
    index = faiss.IndexFlatIP(X_all.shape[1])
    faiss.normalize_L2(X_all)
    index.add(X_all)
    distances, indices = index.search(X_all, k + 1)
else:
    print("Using sklearn NearestNeighbors...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(X_all)
    distances, indices = nn.kneighbors(X_all)

# Build edges
edges = []
similarity_threshold = 0.7

print("Building edges with similarity threshold...")
for i in tqdm(range(len(ids)), desc="Creating graph edges"):
    for j, dist in zip(indices[i][1:], distances[i][1:]):
        sim = 1 - dist if not USE_FAISS else dist
        if sim > similarity_threshold:
            edges.append((ids[i], ids[j], float(sim)))

G = nx.Graph()
G.add_weighted_edges_from(edges)
print(f"✅ Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Check if graph is too small
if G.number_of_nodes() < 100:
    print(f"⚠️ Warning: Graph is very small ({G.number_of_nodes()} nodes)")
    print(f"   Consider lowering similarity threshold (current: {similarity_threshold})")

# -------------------------------
# Compute Graph Metrics
# -------------------------------
print("Computing graph metrics...")
print("Degree centrality...")
degree = dict(G.degree())

print("PageRank...")
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

print("Clustering coefficient...")
clustering = nx.clustering(G)

print("Closeness centrality (may take time)...")
# Only compute for largest component to save time
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)
closeness_partial = nx.closeness_centrality(subG)
closeness = {n: closeness_partial.get(n, 0) for n in G.nodes()}

print("Eigenvector centrality...")
try:
    eigenvector = nx.eigenvector_centrality(subG, max_iter=500, tol=1e-06)
    eigenvector = {n: eigenvector.get(n, 0) for n in G.nodes()}
except Exception as e:
    print(f"   ⚠️ Eigenvector centrality failed: {e}")
    eigenvector = {n: 0 for n in G.nodes()}

print("Community detection (Louvain)...")
try:
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, seed=42)
    community_map = {n: i for i, c in enumerate(communities) for n in c}
    print(f"   Found {len(communities)} communities")
except Exception as e:
    print(f"   ⚠️ Community detection failed: {e}")
    community_map = {n: 0 for n in G.nodes()}

# -------------------------------
# Label Propagation (train only)
# -------------------------------
print("🧩 Running Label Propagation (semi-supervised)...")

X_train = normalize(train_df[embedding_cols].values.astype(np.float32))
y_train = train_df["label"].fillna(-1).to_numpy()

# Check class distribution
unique, counts = np.unique(y_train[y_train != -1], return_counts=True)
print(f"   Training labels distribution: {dict(zip(unique, counts))}")

try:
    label_model = LabelPropagation(kernel='knn', n_neighbors=10, max_iter=200)
    label_model.fit(X_train, y_train)
    train_labels_propagated = label_model.transduction_
    print("   ✅ Label propagation completed")
except Exception as e:
    print(f"   ⚠️ Label propagation failed: {e}")
    print("   Using original labels instead")
    train_labels_propagated = y_train

# Test nodes get -1 (unknown)
test_labels_propagated = np.full(len(test_df), -1)

# -------------------------------
# Save features
# -------------------------------
print("\n💾 Saving graph features...")

def make_feature_df(sub_df, propagated_labels, set_type):
    features = pd.DataFrame({
        "node_id": sub_df["id"].values,
        "degree": [degree.get(n, 0) for n in sub_df["id"]],
        "pagerank": [pagerank.get(n, 0) for n in sub_df["id"]],
        "clustering": [clustering.get(n, 0) for n in sub_df["id"]],
        "community": [community_map.get(n, 0) for n in sub_df["id"]],
        "closeness": [closeness.get(n, 0) for n in sub_df["id"]],
        "eigenvector": [eigenvector.get(n, 0) for n in sub_df["id"]],
        "label_propagated": propagated_labels
    })
    
    # Check for NaN values
    nan_counts = features.isna().sum()
    if nan_counts.any():
        print(f"   ⚠️ {set_type} has NaN values:")
        print(nan_counts[nan_counts > 0])
        features = features.fillna(0)
    
    return features

train_feat = make_feature_df(train_df, train_labels_propagated, "train")
test_feat = make_feature_df(test_df, test_labels_propagated, "test")

# Save
train_path = os.path.join(FEATURES_DIR, "graph_features_train.csv")
test_path = os.path.join(FEATURES_DIR, "graph_features_test.csv")

train_feat.to_csv(train_path, index=False)
test_feat.to_csv(test_path, index=False)

print(f"✅ Saved graph features:")
print(f"   Train: {train_path} ({train_feat.shape})")
print(f"   Test:  {test_path} ({test_feat.shape})")

# Display sample statistics
print("\n📊 Feature Statistics (Train set):")
print(train_feat.describe().T[['mean', 'std', 'min', 'max']])

print("\n🎉 Graph building complete!")