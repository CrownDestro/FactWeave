# FactWeave  
### A Hybrid Graph and NLP Model for Misinformation and Fake News Detection using Social Network Analysis

---

## Overview

**FactWeave** is a hybrid Social Network Analysis (SNA)-based framework designed to detect misinformation and fake news by analyzing both textual content and the network structure of news dissemination.  
The system integrates transformer-based text embeddings with graph-derived features to improve classification accuracy and interpretability.  
By leveraging community structures, propagation patterns, and explainable AI techniques, FactWeave provides an interpretable and data-driven approach to identifying misinformation.

---

## Key Features

- Hybrid detection model combining text embeddings and graph features.  
- Graph construction and analysis using CoAID, GossipCop, and PolitiFact datasets.  
- Community detection via Louvain and Label Propagation algorithms.  
- SHAP-based feature importance and model interpretability.  
- Comprehensive performance reports including ROC, PR, and confusion matrix visualizations.  

---

## Project Structure

```
FactWeave/
│
├── data/
│ ├── raw/ # Original datasets (CoAID, GossipCop, PolitiFact)
│ ├── processed/ # Cleaned and preprocessed data
│ ├── features/ # Extracted graph and text features
│ └── model/ # Trained models and generated outputs
│
├── notebooks/ # Experimental notebooks
│
├── reports/ # Evaluation reports and visualizations
│
├── src/
│ ├── app.py # Flask application for serving predictions
│ ├── build_graph.py # Graph construction logic
│ ├── build_graph_faiss.py # Graph construction using FAISS similarity
│ ├── generate_embeddings.py # Text embedding generation using transformers
│ ├── preprocess_datasets.py # Dataset cleaning and preprocessing
│ ├── label_propagation.py # Label propagation algorithm
│ ├── train_text_xgb.py # Text-based XGBoost model
│ ├── train_hybrid_model.py # Hybrid text-graph model training
│ ├── train_hybrid_with_prop.py # Hybrid model with label propagation
│ ├── shap_analysis.py # Model interpretability using SHAP
│ ├── predict_news.py # Script for making predictions
│ └── utils/ # Utility modules
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/CrownDestro/FactWeave.git
cd FactWeave
```
### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### Usage
## Step 1: Preprocess datasets
```bash
python src/preprocess_datasets.py
```
## Step 2: Generate text embeddings
```bash
python src/generate_embeddings.py
```
## Step 3: Build the graph and extract features
```bash
python src/build_graph.py
```
## Step 4: Train the hybrid model
```bash
python src/train_hybrid_model.py
```
## Step 5: Evaluate model performance
```bash
Performance visualizations including ROC curve, PR curve, and confusion matrix are stored in the reports/ directory.
```
Datasets

The project uses multiple publicly available misinformation datasets:
CoAID – COVID-19 healthcare misinformation dataset.
GossipCop – Real and fake news articles from social media sources.
PolitiFact – Political news verification dataset.

Evaluation Metrics
Model performance is assessed using the following metrics:
Accuracy
Precision, Recall, and F1-score
ROC-AUC
Modularity and Conductance (for network-based evaluation)

### RESULTS
```
============================================================
FINAL MODEL EVALUATION (RandomForest Only)
============================================================

              precision    recall  f1-score   support

           0      0.867     0.683     0.764      5590
           1      0.613     0.828     0.704      3393

    accuracy                          0.738      8983
   macro avg      0.740     0.755     0.734      8983
weighted avg      0.771     0.738     0.741      8983

ROC-AUC: 0.827
PR-AUC: 0.712
Best Threshold: 0.2359 (Youden)
```