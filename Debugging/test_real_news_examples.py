#!/usr/bin/env python3
"""
Test various REAL news examples that should get low Fake probabilities
"""

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "data/model/hybrid_model.pkl"

def predict_news(text: str, verbose=False):
    """Quick prediction function"""
    model_package = joblib.load(MODEL_PATH)
    rf = model_package['rf']
    sna_scaler = model_package['sna_scaler']
    threshold = model_package['threshold']
    feature_info = model_package['feature_info']
    
    encoder = SentenceTransformer("all-mpnet-base-v2")
    embedding = encoder.encode([text], normalize_embeddings=True)
    
    n_sna = feature_info['n_sna_features']
    sna_features = np.array([[0.5] * n_sna])
    sna_scaled = sna_scaler.transform(sna_features)
    X_combined = np.hstack([embedding, sna_scaled])
    
    prob_fake = float(rf.predict_proba(X_combined)[:, 1][0])
    prob_real = 1.0 - prob_fake
    predicted = "Fake" if prob_fake > threshold else "Real"
    
    return {
        'prob_fake': prob_fake,
        'prob_real': prob_real,
        'predicted': predicted,
        'threshold': threshold
    }

# Test cases: Various types of REAL news
test_cases = [
    # Historical facts
    ("Historical Facts", [
        "World War II ended in 1945 after Germany and Japan surrendered to Allied forces.",
        "The United States declared independence from Britain on July 4, 1776.",
        "Neil Armstrong became the first person to walk on the Moon on July 20, 1969.",
        "The Berlin Wall fell on November 9, 1989, marking the end of the Cold War era.",
    ]),
    
    # Political facts
    ("Political Facts", [
        "Joe Biden was inaugurated as the 46th President of the United States on January 20, 2021.",
        "The Affordable Care Act, also known as Obamacare, was signed into law by President Obama in 2010.",
        "Kamala Harris became the first female Vice President of the United States in 2021.",
        "Donald Trump served as the 45th President of the United States from 2017 to 2021.",
    ]),
    
    # Scientific facts
    ("Scientific Facts", [
        "DNA carries genetic information and was discovered by Watson and Crick in 1953.",
        "Climate change is caused by increased greenhouse gas emissions from human activities.",
        "Vaccines work by training the immune system to recognize and fight specific diseases.",
        "The Earth orbits the Sun at an average distance of about 93 million miles.",
    ]),
    
    # Entertainment news (factual)
    ("Entertainment Facts", [
        "Taylor Swift released her album 1989 (Taylor's Version) in 2023.",
        "The Marvel Cinematic Universe has become the highest-grossing film franchise of all time.",
        "Leonardo DiCaprio won his first Oscar for Best Actor in 2016 for The Revenant.",
        "Netflix began as a DVD rental service before transitioning to streaming in 2007.",
    ]),
    
    # Sports facts
    ("Sports Facts", [
        "The FIFA World Cup is held every four years and is the most watched sporting event globally.",
        "Michael Jordan won six NBA championships with the Chicago Bulls in the 1990s.",
        "Serena Williams has won 23 Grand Slam singles titles in professional tennis.",
        "The Summer Olympics were postponed to 2021 due to the COVID-19 pandemic.",
    ]),
    
    # Technology facts
    ("Technology Facts", [
        "Apple released the first iPhone in 2007, revolutionizing the smartphone industry.",
        "Facebook was founded by Mark Zuckerberg in 2004 at Harvard University.",
        "Artificial intelligence uses machine learning algorithms to learn from data and make predictions.",
        "The internet was originally developed by DARPA for military communications in the 1960s.",
    ]),
]

print("="*70)
print("🧪 TESTING REAL NEWS EXAMPLES")
print("="*70)
print(f"\n📊 Looking for examples with Fake probability < 23.6% (threshold)")
print("="*70)

# Load model once
print("\n🔄 Loading model...")
_ = predict_news("test", verbose=False)
print("✅ Model loaded\n")

all_results = []

for category, examples in test_cases:
    print("\n" + "="*70)
    print(f"📰 {category.upper()}")
    print("="*70)
    
    for text in examples:
        result = predict_news(text)
        
        fake_pct = result['prob_fake'] * 100
        real_pct = result['prob_real'] * 100
        predicted = result['predicted']
        
        # Check if it predicts REAL
        symbol = "✅" if predicted == "Real" else "❌"
        
        all_results.append({
            'category': category,
            'text': text,
            'fake_pct': fake_pct,
            'predicted': predicted
        })
        
        print(f"\n{symbol} Fake: {fake_pct:5.2f}% | Real: {real_pct:5.2f}% → {predicted}")
        print(f"   {text[:150]}{'...' if len(text) > 150 else ''}")

# Summary
print("\n\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

import pandas as pd
results_df = pd.DataFrame(all_results)

print(f"\nTotal examples tested: {len(results_df)}")
print(f"Predicted as REAL: {(results_df['predicted'] == 'Real').sum()} ({(results_df['predicted'] == 'Real').sum()/len(results_df)*100:.1f}%)")
print(f"Predicted as FAKE: {(results_df['predicted'] == 'Fake').sum()} ({(results_df['predicted'] == 'Fake').sum()/len(results_df)*100:.1f}%)")

print(f"\nFake probability statistics:")
print(f"   Min: {results_df['fake_pct'].min():.2f}%")
print(f"   Max: {results_df['fake_pct'].max():.2f}%")
print(f"   Mean: {results_df['fake_pct'].mean():.2f}%")
print(f"   Median: {results_df['fake_pct'].median():.2f}%")

print(f"\nBy category:")
for category in results_df['category'].unique():
    cat_df = results_df[results_df['category'] == category]
    real_count = (cat_df['predicted'] == 'Real').sum()
    print(f"   {category}: {real_count}/{len(cat_df)} predicted as REAL ({real_count/len(cat_df)*100:.1f}%)")
    print(f"      Mean Fake probability: {cat_df['fake_pct'].mean():.2f}%")

print("\n" + "="*70)
print("🔍 EXAMPLES PREDICTED AS REAL (Fake < 23.6%):")
print("="*70)

real_predictions = results_df[results_df['predicted'] == 'Real'].sort_values('fake_pct')
if len(real_predictions) > 0:
    print(f"\nFound {len(real_predictions)} examples:\n")
    for idx, row in real_predictions.iterrows():
        print(f"✅ Fake: {row['fake_pct']:5.2f}%")
        print(f"   {row['text'][:150]}{'...' if len(row['text']) > 150 else ''}\n")
else:
    print("\n❌ NO EXAMPLES PREDICTED AS REAL!")
    print("   This means the model assigns Fake probability > 23.6% to ALL examples")
    print("   Even well-known factual statements are considered 'suspicious'")

print("="*70)