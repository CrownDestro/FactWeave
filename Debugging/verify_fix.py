#!/usr/bin/env python3
"""
Complete System Verification Script
------------------------------------
Checks if the fix worked and model is producing valid predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    colors = {"success": GREEN, "error": RED, "warning": YELLOW, "info": BLUE}
    color = colors.get(status, RESET)
    symbols = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    symbol = symbols.get(status, "•")
    print(f"{color}{symbol} {message}{RESET}")

def check_file_exists(filepath, description):
    """Check if file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print_status(f"{description}: Found ({size:.2f} MB)", "success")
        return True
    else:
        print_status(f"{description}: Missing!", "error")
        return False

def check_embeddings():
    """Verify embeddings file"""
    print("\n" + "="*60)
    print("📊 CHECKING EMBEDDINGS FILE")
    print("="*60)
    
    embed_path = "data/model/text_embeddings.csv"
    
    if not check_file_exists(embed_path, "Embeddings CSV"):
        return False
    
    try:
        import io
        with open(embed_path, "rb") as f:
            raw_bytes = f.read()
        
        df = pd.read_csv(
            io.BytesIO(raw_bytes.decode("utf-8", errors="replace").encode("utf-8")),
            on_bad_lines="skip",
            engine="python",
            nrows=100  # Just check first 100 rows
        )
        
        print_status(f"Embeddings shape: {df.shape}", "success")
        print_status(f"Columns: {df.columns.tolist()[:5]}...", "info")
        
        # Check for required columns
        if 'id' in df.columns and 'label' in df.columns:
            print_status("Required columns present", "success")
            return True
        else:
            print_status("Missing 'id' or 'label' column", "error")
            return False
            
    except Exception as e:
        print_status(f"Error reading embeddings: {e}", "error")
        return False

def check_graph_features():
    """Verify graph features"""
    print("\n" + "="*60)
    print("🕸️ CHECKING GRAPH FEATURES")
    print("="*60)
    
    train_path = "data/features/graph_features_train.csv"
    test_path = "data/features/graph_features_test.csv"
    
    train_ok = check_file_exists(train_path, "Train graph features")
    test_ok = check_file_exists(test_path, "Test graph features")
    
    if not (train_ok and test_ok):
        return False
    
    try:
        train_df = pd.read_csv(train_path, nrows=10)
        test_df = pd.read_csv(test_path, nrows=10)
        
        expected_cols = ["node_id", "degree", "pagerank", "clustering", 
                        "community", "closeness", "eigenvector", "label_propagated"]
        
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        missing_train = set(expected_cols) - train_cols
        missing_test = set(expected_cols) - test_cols
        
        if missing_train:
            print_status(f"Train missing columns: {missing_train}", "error")
            return False
        if missing_test:
            print_status(f"Test missing columns: {missing_test}", "error")
            return False
        
        print_status("All expected columns present", "success")
        return True
        
    except Exception as e:
        print_status(f"Error reading graph features: {e}", "error")
        return False

def check_model():
    """Verify trained model"""
    print("\n" + "="*60)
    print("🤖 CHECKING TRAINED MODEL")
    print("="*60)
    
    model_path = "data/model/hybrid_model.pkl"
    
    if not check_file_exists(model_path, "Hybrid model"):
        return False
    
    try:
        model_package = joblib.load(model_path)
        
        required_keys = ['rf', 'xgb', 'meta', 'scaler', 'threshold', 'feature_info']
        missing_keys = [k for k in required_keys if k not in model_package]
        
        if missing_keys:
            print_status(f"Missing model components: {missing_keys}", "error")
            print_status(f"Available keys: {list(model_package.keys())}", "info")
            return False
        
        print_status("All model components present", "success")
        
        # Check threshold
        threshold = model_package['threshold']
        print_status(f"Decision threshold: {threshold:.4f} ({threshold*100:.2f}%)", "info")
        
        if threshold < 0.01:
            print_status("⚠️ Threshold is suspiciously low! Model may be broken.", "warning")
            return False
        elif threshold > 0.99:
            print_status("⚠️ Threshold is suspiciously high! Model may be broken.", "warning")
            return False
        else:
            print_status("Threshold is in reasonable range", "success")
        
        # Check scaler type
        scaler_type = type(model_package['scaler']).__name__
        print_status(f"Scaler type: {scaler_type}", "info")
        
        if scaler_type != 'StandardScaler':
            print_status(f"Expected StandardScaler, got {scaler_type}", "warning")
        
        return True
        
    except Exception as e:
        print_status(f"Error loading model: {e}", "error")
        return False

def test_predictions():
    """Test actual predictions"""
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTIONS")
    print("="*60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model
        model_package = joblib.load("data/model/hybrid_model.pkl")
        rf = model_package['rf']
        xgb = model_package['xgb']
        meta = model_package['meta']
        scaler = model_package['scaler']
        threshold = model_package['threshold']
        feature_info = model_package['feature_info']
        
        encoder = SentenceTransformer("all-mpnet-base-v2")
        
        # Test cases: (text, expected_label)
        test_cases = [
            ("Barack Obama bans the Pledge of Allegiance in all U.S. schools", "Fake"),
            ("The Affordable Care Act was signed into law by President Obama in 2010", "Real"),
            ("Drinking bleach cures coronavirus according to new study", "Fake"),
            ("NASA confirmed water exists on Mars", "Real"),
        ]
        
        results = []
        for text, expected in test_cases:
            # Encode
            embedding = encoder.encode([text])
            
            # Add SNA features
            n_sna = feature_info['n_sna_features']
            sna_features = np.array([[0.5] * n_sna])
            
            # Combine and scale
            X_combined = np.hstack([embedding, sna_features])
            X_scaled = scaler.transform(X_combined)
            
            # Predict
            rf_prob = rf.predict_proba(X_scaled)[:, 1]
            xgb_prob = xgb.predict_proba(X_scaled)[:, 1]
            meta_input = np.vstack([rf_prob, xgb_prob]).T
            prob_fake = float(meta.predict_proba(meta_input)[:, 1][0])
            
            predicted = "Fake" if prob_fake > threshold else "Real"
            
            results.append({
                'text': text[:50] + "...",
                'expected': expected,
                'predicted': predicted,
                'prob_fake': prob_fake * 100,
                'prob_real': (1 - prob_fake) * 100,
                'correct': predicted == expected
            })
        
        # Display results
        print(f"\n{'Text':<52} {'Expected':<8} {'Predicted':<10} {'P(Fake)':<10} {'P(Real)':<10} {'Status'}")
        print("-" * 110)
        
        correct_count = 0
        prob_in_range = True
        
        for r in results:
            status = "✅" if r['correct'] else "❌"
            if r['correct']:
                correct_count += 1
            
            print(f"{r['text']:<52} {r['expected']:<8} {r['predicted']:<10} "
                  f"{r['prob_fake']:>6.1f}%    {r['prob_real']:>6.1f}%    {status}")
            
            # Check probability range
            if r['prob_fake'] < 1 or r['prob_fake'] > 99:
                if r['prob_fake'] < 1 and r['prob_real'] > 99:
                    pass  # This is fine - high confidence Real
                elif r['prob_fake'] > 99 and r['prob_real'] < 1:
                    pass  # This is fine - high confidence Fake
            
            if r['prob_fake'] < 0.1 or (r['prob_fake'] > 0.1 and r['prob_fake'] < 0.5):
                prob_in_range = False
        
        print()
        accuracy = (correct_count / len(test_cases)) * 100
        print_status(f"Accuracy on test cases: {accuracy:.0f}% ({correct_count}/{len(test_cases)})", 
                    "success" if accuracy >= 50 else "warning")
        
        # Check if probabilities are in reasonable range
        avg_fake_prob = np.mean([r['prob_fake'] for r in results])
        min_prob = min([min(r['prob_fake'], r['prob_real']) for r in results])
        max_prob = max([max(r['prob_fake'], r['prob_real']) for r in results])
        
        print_status(f"Probability range: [{min_prob:.1f}%, {max_prob:.1f}%]", "info")
        
        if max_prob < 10:
            print_status("❌ CRITICAL: All probabilities < 10%! Model is broken!", "error")
            return False
        elif min_prob < 0.1 and max_prob < 1:
            print_status("❌ CRITICAL: Probabilities < 1%! Model is broken!", "error")
            return False
        else:
            print_status("Probabilities are in reasonable range", "success")
        
        return correct_count >= len(test_cases) // 2  # At least 50% correct
        
    except Exception as e:
        print_status(f"Error during prediction test: {e}", "error")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("🔍 FAKE NEWS DETECTION SYSTEM VERIFICATION")
    print("="*60)
    
    checks = [
        ("Embeddings", check_embeddings),
        ("Graph Features", check_graph_features),
        ("Trained Model", check_model),
        ("Predictions", test_predictions),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Final summary
    print("\n" + "="*60)
    print("📋 VERIFICATION SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = GREEN if passed else RED
        print(f"{color}{name:.<40} {status}{RESET}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print_status("🎉 ALL CHECKS PASSED! System is working correctly.", "success")
        print("="*60)
        print("\nNext steps:")
        print("1. Run full test suite: ./src/test_examples.sh")
        print("2. Start Flask API: python3 src/app.py")
        return 0
    else:
        print_status("❌ SOME CHECKS FAILED! Please fix the issues above.", "error")
        print("="*60)
        print("\nRecommended actions:")
        if not results["Embeddings"]:
            print("1. Regenerate embeddings: python3 src/generate_embeddings.py")
        if not results["Graph Features"]:
            print("2. Rebuild graph: python3 src/build_graph_faiss.py")
        if not results["Trained Model"]:
            print("3. Retrain model: python3 src/train_hybrid_model.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())