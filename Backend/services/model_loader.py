# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import joblib
import pickle
import tensorflow as tf


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==========================================================
# LOAD ALL MODELS
# ==========================================================

print("=" * 60)
print("Loading AI Models...")
print("=" * 60)

# Regression Model
regression_model = joblib.load(
    os.path.join(MODEL_DIR, "best_regression_model.pkl")
)
print("✔ Regression Model Loaded")

# Classification Model
classification_model = joblib.load(
    os.path.join(MODEL_DIR, "best_classification_model.pkl")
)
print("✔ Classification Model Loaded")

# Clustering Model
clustering_model = joblib.load(
    os.path.join(MODEL_DIR, "best_clustering_model.pkl")
)
print("✔ Clustering Model Loaded")

# PCA Model
pca_model = joblib.load(
    os.path.join(MODEL_DIR, "pca_model.pkl")
)
print("✔ PCA Model Loaded")

# Tokenizer
tokenizer = joblib.load(
    os.path.join(MODEL_DIR, "tokenizer.pkl")
)
print("✔ Tokenizer Loaded")

# TF-IDF Vectorizer
with open(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    "rb"
) as f:
    tfidf_vectorizer = pickle.load(f)

print("✔ TF-IDF Vectorizer Loaded")

# Metadata
metadata = joblib.load(
    os.path.join(MODEL_DIR, "metadata.pkl")
)
print("✔ Metadata Loaded")

# NLP Model
nlp_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "best_nlp_model.keras")
)
print("✔ NLP Model Loaded")

print("=" * 60)
print("All Models Loaded Successfully")
print("=" * 60)