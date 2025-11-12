#!/bin/bash
set -e  # Exit on error

echo "========================================================================"
echo "🚨 NUCLEAR RESET - Complete Fresh Start"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Backup existing data"
echo "  2. Delete corrupted files"
echo "  3. Regenerate embeddings from scratch"
echo "  4. Rebuild graph features"
echo "  5. Retrain model"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ======================
# Step 1: Backup
# ======================
echo ""
echo "📦 Step 1/5: Creating backup..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r data/model "$BACKUP_DIR/" 2>/dev/null || true
cp -r data/features "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backup created: $BACKUP_DIR"

# ======================
# Step 2: Clean
# ======================
echo ""
echo "🧹 Step 2/5: Deleting corrupted files..."
rm -f data/model/text_embeddings.csv
rm -f data/model/hybrid_model.pkl
rm -f data/features/graph_features_*.csv
rm -f reports/*.png reports/results.txt
echo "✅ Corrupted files deleted"

# ======================
# Step 3: Generate Embeddings
# ======================
echo ""
echo "🧠 Step 3/5: Generating fresh embeddings (this takes ~10-15 minutes)..."
python3 src/generate_embeddings.py
if [ $? -ne 0 ]; then
    echo "❌ Error generating embeddings!"
    exit 1
fi
echo "✅ Embeddings generated"

# Verify embeddings
echo ""
echo "🔍 Verifying embeddings..."
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('data/model/text_embeddings.csv', nrows=10)
embedding_cols = [c for c in df.columns if c.isdigit()]
emb_data = df[embedding_cols].values
print(f'Shape: {emb_data.shape}')
print(f'Dtype: {emb_data.dtype}')
print(f'Min: {emb_data.min():.6f}')
print(f'Max: {emb_data.max():.6f}')
if emb_data.dtype not in [np.float64, np.float32]:
    print('❌ ERROR: Embeddings have wrong dtype!')
    exit(1)
else:
    print('✅ Embeddings verified')
"
if [ $? -ne 0 ]; then
    echo "❌ Embeddings verification failed!"
    exit 1
fi

# ======================
# Step 4: Build Graph
# ======================
echo ""
echo "🕸️  Step 4/5: Building graph features (this takes ~2-5 minutes)..."
python3 src/build_graph_faiss.py
if [ $? -ne 0 ]; then
    echo "❌ Error building graph!"
    exit 1
fi
echo "✅ Graph features built"

# ======================
# Step 5: Train Model
# ======================
echo ""
echo "🤖 Step 5/5: Training model (this takes ~5-10 minutes)..."
python3 src/train_hybrid_model.py
if [ $? -ne 0 ]; then
    echo "❌ Error training model!"
    exit 1
fi
echo "✅ Model trained"

# ======================
# Final Verification
# ======================
echo ""
echo "========================================================================"
echo "🔍 FINAL VERIFICATION"
echo "========================================================================"

# Check probability range
echo ""
echo "📊 Checking probability distribution..."
python3 -c "
import re
with open('reports/results.txt', 'r') as f:
    content = f.read()
    # Try to extract min/max from probability distribution output
    # (this is approximate since it's not saved in results.txt)
    print('Check the training output above for:')
    print('  Min: should be > 0.05')
    print('  Max: should be > 0.80')
    print('  Threshold: should be 0.30-0.70')
"

# Test prediction
echo ""
echo "🧪 Testing sample prediction..."
python3 src/predict_news.py "Barack Obama bans the Pledge of Allegiance"

echo ""
echo "========================================================================"
echo "✅ NUCLEAR RESET COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Check the prediction output above"
echo "  2. If probabilities are still < 1%, the source data (articles_clean.csv) may be corrupted"
echo "  3. Run: ./src/test_examples.sh"
echo "  4. Start API: python3 src/app.py"
echo ""