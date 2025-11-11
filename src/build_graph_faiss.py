#!/usr/bin/env python3
"""
Memory-efficient graph builder using FAISS / sklearn k-NN.
Now with:
- train/test split BEFORE label propagation (no data leakage)
- label propagation only on training nodes
- realistic graph metrics for hybrid model
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
df = pd.read_csv(EMBED_PATH)
print(f"✅ Loaded embeddings: {df.shape}")

# -------------------------------
# Train-Test Split (before graph)
# -------------------------------
print("📊 Splitting into train/test sets (80/20)...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Normalize embeddings
embedding_cols = [c for c in df.columns if str(c).isdigit()]
X_all = normalize(df[embedding_cols].values.astype(np.float32))
ids = df["id"].tolist()

# -------------------------------
# Build top-k similarity graph
# -------------------------------
k =30
print(f"🔍 Building top-{k} nearest neighbor graph...")

if USE_FAISS:
    index = faiss.IndexFlatIP(X_all.shape[1])
    faiss.normalize_L2(X_all)
    index.add(X_all)
    distances, indices = index.search(X_all, k + 1)
else:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(X_all)
    distances, indices = nn.kneighbors(X_all)

edges = []
for i in tqdm(range(len(ids))):
    for j, dist in zip(indices[i][1:], distances[i][1:]):
        sim = 1 - dist if not USE_FAISS else dist
        if sim > 0.7:
            edges.append((ids[i], ids[j], float(sim)))

G = nx.Graph()
G.add_weighted_edges_from(edges)
print(f"✅ Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print("🧠 Calculating degree, pagerank, closeness, eigenvector centrality...")

# -------------------------------
# Compute Graph Metrics
# -------------------------------
print("📈 Computing graph metrics...")
pagerank = nx.pagerank(G, alpha=0.85)
closeness = nx.closeness_centrality(G)
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)
eigenvector_partial = nx.eigenvector_centrality(subG)
eigenvector = {n: eigenvector_partial.get(n, 0) for n in G.nodes()}

# Safe eigenvector computation for disconnected graphs
try:
    eigenvector = nx.eigenvector_centrality(G, max_iter=500, tol=1e-06)
except Exception as e:
    print(f"⚠️ Eigenvector centrality skipped: {e}")
    eigenvector = {n: 0 for n in G.nodes()}

degree = dict(G.degree())
clustering = nx.clustering(G)


try:
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, seed=42)
    community_map = {n: i for i, c in enumerate(communities) for n in c}
except Exception:
    print("⚠️ Louvain community detection skipped.")
    community_map = {n: 0 for n in G.nodes()}

# -------------------------------
# Label Propagation (train only)
# -------------------------------
from sklearn.semi_supervised import LabelSpreading

print("🧩 Running Label Spreading (semi-supervised, confidence-weighted)...")

X_train = normalize(train_df[embedding_cols].values.astype(np.float32))
y_train = train_df["label"].fillna(-1).to_numpy()

label_model = LabelSpreading(kernel='rbf', alpha=0.8, max_iter=200)
label_model.fit(X_train, y_train)

train_labels_propagated = label_model.transduction_
test_labels_propagated = np.full(len(test_df), -1)  # unknown for test nodes
# -------------------------------
# Save features
# -------------------------------
def make_feature_df(sub_df, propagated_labels, set_type):
    return pd.DataFrame({
        "node_id": sub_df["id"],
        "degree": [degree.get(n, 0) for n in sub_df["id"]],
        "pagerank": [pagerank.get(n, 0) for n in sub_df["id"]],
        "clustering": [clustering.get(n, 0) for n in sub_df["id"]],
        "community": [community_map.get(n, 0) for n in sub_df["id"]],
        "closeness": [closeness.get(n, 0) for n in sub_df["id"]],
        "eigenvector": [eigenvector.get(n, 0) for n in sub_df["id"]],
        "label_propagated": propagated_labels
    })

train_feat = make_feature_df(train_df, train_labels_propagated, "train")
test_feat = make_feature_df(test_df, test_labels_propagated, "test")

train_feat.to_csv(os.path.join(FEATURES_DIR, "graph_features_train.csv"), index=False)
test_feat.to_csv(os.path.join(FEATURES_DIR, "graph_features_test.csv"), index=False)
print(f"✅ Saved train/test graph features to {FEATURES_DIR}")
