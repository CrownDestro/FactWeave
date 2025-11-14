#!/usr/bin/env python3
"""
Graph Visualization and Report Generation
------------------------------------------
Creates comprehensive visualizations for project report:
1. Community detection visualization (Louvain)
2. Graph statistics plots
3. Feature importance charts
4. Model performance visualizations
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

# Paths
EMBED_PATH = "data/model/text_embeddings.csv"
GRAPH_TRAIN = "data/features/graph_features_train.csv"
MODEL_PATH = "data/model/hybrid_model.pkl"
REPORT_DIR = "reports/visualizations"
os.makedirs(REPORT_DIR, exist_ok=True)

print("📊 Starting visualization generation for project report...\n")

# ============================================
# 1. LOAD DATA
# ============================================
print("📂 Loading data...")
with open(EMBED_PATH, "rb") as f:
    raw_bytes = f.read()

text_df = pd.read_csv(
    io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
    on_bad_lines="skip",
    engine="python"
)

graph_features = pd.read_csv(GRAPH_TRAIN)
merged_df = pd.merge(text_df, graph_features, left_on="id", right_on="node_id", how="inner")

print(f"✅ Loaded {len(merged_df)} samples\n")

# ============================================
# 2. COMMUNITY DETECTION VISUALIZATION
# ============================================
print("🎨 Creating community detection visualization...")

# Build graph from embeddings
embedding_cols = [c for c in text_df.columns if str(c).isdigit()]
X_all = normalize(text_df[embedding_cols].values.astype(np.float32))
ids = text_df["id"].tolist()

# Sample for visualization (full graph too large)
sample_size = min(2000, len(X_all))
sample_indices = np.random.choice(len(X_all), sample_size, replace=False)
X_sample = X_all[sample_indices]
ids_sample = [ids[i] for i in sample_indices]
labels_sample = text_df.iloc[sample_indices]["label"].values

# Build k-NN graph
from sklearn.neighbors import NearestNeighbors
k = 15
nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=-1)
nn.fit(X_sample)
distances, indices = nn.kneighbors(X_sample)

# Create graph
G = nx.Graph()
similarity_threshold = 0.6

for i in range(len(ids_sample)):
    for j, dist in zip(indices[i][1:], distances[i][1:]):
        sim = 1 - dist
        if sim > similarity_threshold:
            G.add_edge(ids_sample[i], ids_sample[j], weight=sim)

print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Community detection
from networkx.algorithms.community import louvain_communities
communities = louvain_communities(G, seed=42)
community_map = {n: i for i, c in enumerate(communities) for n in c}

print(f"   Detected {len(communities)} communities")

# Layout for visualization
print("   Computing layout (this may take a minute)...")
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Plot 1: Communities
ax1 = axes[0]
colors_community = [community_map.get(node, 0) for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=colors_community, 
                       node_size=30, cmap='tab20', alpha=0.7, ax=ax1)
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax1)
ax1.set_title(f"Community Detection (Louvain Algorithm)\n{len(communities)} Communities Detected", 
              fontsize=16, fontweight='bold')
ax1.axis('off')

# Add legend for top communities
community_sizes = pd.Series([community_map.get(n, 0) for n in G.nodes()]).value_counts()
legend_text = "Top 5 Largest Communities:\n"
for i, (comm_id, size) in enumerate(community_sizes.head(5).items(), 1):
    legend_text += f"Community {comm_id}: {size} nodes\n"
ax1.text(0.02, 0.98, legend_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Fake vs Real
ax2 = axes[1]
node_labels_map = {ids_sample[i]: labels_sample[i] for i in range(len(ids_sample))}
colors_labels = ['red' if node_labels_map.get(node, 0) == 1 else 'green' for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=colors_labels, 
                       node_size=30, alpha=0.7, ax=ax2)
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, ax=ax2)
ax2.set_title("Article Labels\nRed: Fake News | Green: Real News", 
              fontsize=16, fontweight='bold')
ax2.axis('off')

# Add statistics
fake_count = sum(1 for c in colors_labels if c == 'red')
real_count = sum(1 for c in colors_labels if c == 'green')
stats_text = f"Sample Statistics:\nTotal: {len(colors_labels)}\nFake: {fake_count} ({fake_count/len(colors_labels)*100:.1f}%)\nReal: {real_count} ({real_count/len(colors_labels)*100:.1f}%)"
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "1_community_detection.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 1_community_detection.png\n")

# ============================================
# 3. GRAPH STATISTICS
# ============================================
print("📊 Creating graph statistics visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Degree distribution
ax1 = axes[0, 0]
degrees = dict(G.degree())
degree_values = list(degrees.values())
ax1.hist(degree_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel("Degree", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.set_title(f"Degree Distribution\nMean: {np.mean(degree_values):.1f}, Max: {max(degree_values)}", 
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# PageRank distribution
ax2 = axes[0, 1]
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
pr_values = list(pagerank.values())
ax2.hist(pr_values, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel("PageRank Score", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title(f"PageRank Distribution\nMean: {np.mean(pr_values):.6f}", 
              fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Clustering coefficient
ax3 = axes[0, 2]
clustering = nx.clustering(G)
clust_values = list(clustering.values())
ax3.hist(clust_values, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel("Clustering Coefficient", fontsize=11)
ax3.set_ylabel("Frequency", fontsize=11)
ax3.set_title(f"Clustering Distribution\nMean: {np.mean(clust_values):.3f}", 
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Community size distribution
ax4 = axes[1, 0]
comm_sizes = [len(c) for c in communities]
ax4.bar(range(len(comm_sizes)), sorted(comm_sizes, reverse=True), color='purple', alpha=0.7)
ax4.set_xlabel("Community Rank", fontsize=11)
ax4.set_ylabel("Size (nodes)", fontsize=11)
ax4.set_title(f"Community Size Distribution\n{len(communities)} Communities", 
              fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

# SNA features comparison (Fake vs Real)
ax5 = axes[1, 1]
sna_cols = ['degree', 'pagerank', 'clustering']
fake_data = merged_df[merged_df['label'] == 1][sna_cols].mean()
real_data = merged_df[merged_df['label'] == 0][sna_cols].mean()

x = np.arange(len(sna_cols))
width = 0.35
ax5.bar(x - width/2, fake_data, width, label='Fake News', color='red', alpha=0.7)
ax5.bar(x + width/2, real_data, width, label='Real News', color='green', alpha=0.7)
ax5.set_xlabel("Graph Features", fontsize=11)
ax5.set_ylabel("Mean Value", fontsize=11)
ax5.set_title("Graph Features: Fake vs Real News", fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(sna_cols)
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# Connected components
ax6 = axes[1, 2]
components = list(nx.connected_components(G))
comp_sizes = [len(c) for c in components]
ax6.text(0.5, 0.5, f"Graph Connectivity\n\n"
         f"Connected Components: {len(components)}\n"
         f"Largest Component: {max(comp_sizes)} nodes\n"
         f"({max(comp_sizes)/G.number_of_nodes()*100:.1f}% of graph)\n\n"
         f"Average Path Length: {nx.average_shortest_path_length(G.subgraph(max(components, key=len))):.2f}\n"
         f"Graph Density: {nx.density(G):.4f}",
         transform=ax6.transAxes, fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax6.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "2_graph_statistics.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 2_graph_statistics.png\n")

# ============================================
# 4. FEATURE IMPORTANCE
# ============================================
print("🎯 Creating feature importance visualization...")

# Load model
model_package = joblib.load(MODEL_PATH)
rf = model_package['rf']
feature_info = model_package['feature_info']

# Get feature importances
importances = rf.feature_importances_

# Separate embedding and SNA importances
n_emb = feature_info['n_embedding_features']
emb_importances = importances[:n_emb]
sna_importances = importances[n_emb:]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top embedding features
ax1 = axes[0]
top_emb_indices = np.argsort(emb_importances)[-20:][::-1]
top_emb_values = emb_importances[top_emb_indices]
ax1.barh(range(len(top_emb_indices)), top_emb_values, color='steelblue', alpha=0.7)
ax1.set_yticks(range(len(top_emb_indices)))
ax1.set_yticklabels([f"Emb_{i}" for i in top_emb_indices])
ax1.set_xlabel("Feature Importance", fontsize=11)
ax1.set_title("Top 20 Embedding Features\n(from 768 total)", fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3, axis='x')
ax1.invert_yaxis()

# Plot 2: SNA features
ax2 = axes[1]
sna_names = feature_info['sna_cols']
colors = ['red' if i == np.argmax(sna_importances) else 'steelblue' for i in range(len(sna_importances))]
ax2.barh(sna_names, sna_importances, color=colors, alpha=0.7)
ax2.set_xlabel("Feature Importance", fontsize=11)
ax2.set_title("Graph Feature Importance\n(Social Network Analysis)", fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3, axis='x')
ax2.invert_yaxis()

# Add statistics
total_emb_importance = emb_importances.sum()
total_sna_importance = sna_importances.sum()
stats_text = f"Feature Group Contribution:\n\n"
stats_text += f"Embeddings (768): {total_emb_importance*100:.1f}%\n"
stats_text += f"Graph Features (7): {total_sna_importance*100:.1f}%\n\n"
stats_text += f"Most Important:\n{sna_names[np.argmax(sna_importances)]} ({sna_importances.max()*100:.2f}%)"
ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, 
         fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "3_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 3_feature_importance.png\n")

# ============================================
# 5. EMBEDDING VISUALIZATION (t-SNE)
# ============================================
print("🗺️  Creating embedding visualization (t-SNE)...")

# Sample for t-SNE (too slow on full dataset)
tsne_sample_size = min(3000, len(merged_df))
sample_indices = np.random.choice(len(merged_df), tsne_sample_size, replace=False)
X_tsne = merged_df.iloc[sample_indices][embedding_cols].values
y_tsne = merged_df.iloc[sample_indices]['label'].values
comm_tsne = merged_df.iloc[sample_indices]['community'].values

print("   Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
X_embedded = tsne.fit_transform(X_tsne)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Colored by label
ax1 = axes[0]
scatter1 = ax1.scatter(X_embedded[y_tsne==0, 0], X_embedded[y_tsne==0, 1], 
                      c='green', alpha=0.5, s=20, label='Real News')
scatter2 = ax1.scatter(X_embedded[y_tsne==1, 0], X_embedded[y_tsne==1, 1], 
                      c='red', alpha=0.5, s=20, label='Fake News')
ax1.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax1.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax1.set_title("t-SNE Visualization of Text Embeddings\nColored by Label", 
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Colored by community
ax2 = axes[1]
scatter = ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=comm_tsne, cmap='tab20', alpha=0.6, s=20)
ax2.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax2.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax2.set_title("t-SNE Visualization of Text Embeddings\nColored by Community", 
              fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Community ID')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "4_tsne_embeddings.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 4_tsne_embeddings.png\n")

# ============================================
# 6. DATASET STATISTICS
# ============================================
print("📈 Creating dataset statistics visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Overall class distribution
ax1 = axes[0, 0]
class_counts = text_df['label'].value_counts()
colors_pie = ['green', 'red']
ax1.pie(class_counts, labels=['Real News', 'Fake News'], autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title(f"Overall Dataset Distribution\nTotal: {len(text_df):,} articles", 
              fontsize=13, fontweight='bold')

# Plot 2: Distribution by source
ax2 = axes[0, 1]
source_counts = text_df.groupby(['source', 'label']).size().unstack(fill_value=0)
source_counts.plot(kind='bar', stacked=False, ax=ax2, color=['green', 'red'], alpha=0.7)
ax2.set_xlabel("Data Source", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("Article Distribution by Source", fontsize=13, fontweight='bold')
ax2.legend(['Real', 'Fake'], loc='upper right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Train-test split
ax3 = axes[1, 0]
split_data = {
    'Train': [len(merged_df[merged_df['label']==0]), len(merged_df[merged_df['label']==1])],
    'Test': [len(merged_df[merged_df['label']==0])*0.2, len(merged_df[merged_df['label']==1])*0.2]
}
x = np.arange(2)
width = 0.35
ax3.bar(x - width/2, split_data['Train'], width, label='Train (80%)', color='steelblue', alpha=0.7)
ax3.bar(x + width/2, split_data['Test'], width, label='Test (20%)', color='coral', alpha=0.7)
ax3.set_ylabel("Count", fontsize=11)
ax3.set_title("Train-Test Split Distribution", fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Real News', 'Fake News'])
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Summary statistics table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

summary_data = [
    ['Metric', 'Value'],
    ['Total Articles', f"{len(text_df):,}"],
    ['Training Samples', f"{int(len(text_df)*0.8):,}"],
    ['Test Samples', f"{int(len(text_df)*0.2):,}"],
    ['', ''],
    ['Real News', f"{class_counts[0]:,} ({class_counts[0]/len(text_df)*100:.1f}%)"],
    ['Fake News', f"{class_counts[1]:,} ({class_counts[1]/len(text_df)*100:.1f}%)"],
    ['', ''],
    ['Embedding Dimensions', '768'],
    ['Graph Features', '7'],
    ['Total Features', '775'],
    ['', ''],
    ['Data Sources', str(text_df['source'].nunique())],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_data)):
    if summary_data[i][0] == '':
        continue
    table[(i, 0)].set_facecolor('#E7E6E6')
    table[(i, 1)].set_facecolor('#F2F2F2')

ax4.set_title("Dataset Summary Statistics", fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "5_dataset_statistics.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 5_dataset_statistics.png\n")

# ============================================
# 7. METHODOLOGY WORKFLOW
# ============================================
print("🔄 Creating methodology workflow diagram...")

fig, ax = plt.subplots(figsize=(12, 14))
ax.axis('off')

# Define workflow steps
steps = [
    ("Data Collection", "GossipCop, PolitiFact, CoAID\n~44,000 articles"),
    ("Data Preprocessing", "Text cleaning, normalization\nDuplicate removal"),
    ("Text Embeddings", "SBERT (all-mpnet-base-v2)\n768-dimensional vectors"),
    ("Graph Construction", "k-NN similarity graph (k=30)\nFAISS for fast search"),
    ("Graph Feature Extraction", "Degree, PageRank, Clustering\nCommunity Detection (Louvain)"),
    ("Feature Combination", "768 embeddings + 7 graph features\n= 775 total features"),
    ("Class Balancing", "SMOTE oversampling\nBalance fake/real ratio"),
    ("Model Training", "Random Forest (500 trees)\nThreshold optimization"),
    ("Model Evaluation", "ROC-AUC, PR-AUC, F1-Score\nConfusion Matrix"),
    ("Deployment", "Flask API + CLI Tool\nReal-time prediction")
]

# Draw workflow
y_start = 0.95
y_step = 0.095
box_height = 0.07
box_width = 0.7

colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(steps)))

for i, (title, desc) in enumerate(steps):
    y_pos = y_start - i * y_step
    
    # Draw box
    rect = plt.Rectangle((0.15, y_pos - box_height/2), box_width, box_height,
                         facecolor=colors[i], edgecolor='black', linewidth=2,
                         transform=ax.transAxes, zorder=1)
    ax.add_patch(rect)
    
    # Add text
    ax.text(0.5, y_pos, f"{i+1}. {title}\n{desc}", 
           transform=ax.transAxes, ha='center', va='center',
           fontsize=10, fontweight='bold', zorder=2)
    
    # Draw arrow (except for last step)
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, y_pos - box_height/2 - 0.01), 
                   xytext=(0.5, y_pos - box_height/2 - y_step + 0.01),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   transform=ax.transAxes)

ax.set_title("Fake News Detection: Methodology Workflow", 
            fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(REPORT_DIR, "6_methodology_workflow.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 6_methodology_workflow.png\n")

# ============================================
# SUMMARY
# ============================================
print("="*60)
print("🎉 VISUALIZATION GENERATION COMPLETE!")
print("="*60)
print(f"\n📁 All visualizations saved to: {REPORT_DIR}/")
print("\n📊 Generated Files:")
print("   1. 1_community_detection.png - Louvain community structure")
print("   2. 2_graph_statistics.png - Comprehensive graph metrics")
print("   3. 3_feature_importance.png - Model feature analysis")
print("   4. 4_tsne_embeddings.png - Embedding space visualization")
print("   5. 5_dataset_statistics.png - Dataset overview")
print("   6. 6_methodology_workflow.png - System pipeline")
print("\n✅ Ready for project report!")