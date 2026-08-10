import joblib
from tensorflow.keras.models import load_model

# =====================================================
# LOAD ALL MODELS
# =====================================================

classification_model = joblib.load(
    "models/best_classification_model.pkl"
)

regression_model = joblib.load(
    "models/best_regression_model.pkl"
)

clustering_model = joblib.load(
    "models/best_clustering_model.pkl"
)

pca_model = joblib.load(
    "models/pca_model.pkl"
)

tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

nlp_model = load_model(
    "models/best_nlp_model.keras"
)

print("All models loaded successfully.")