import os
import sys
import numpy as np

# =====================================================
# PYTHON CONFIGURATION
# =====================================================

PYTHON_PATH = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH

# =====================================================
# IMPORTS
# =====================================================

from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import joblib

# =====================================================
# CREATE SPARK SESSION
# =====================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Recipe Clustering Models")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .getOrCreate()
)



spark.sparkContext.setLogLevel("ERROR")


# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 70)
print("Loading Dataset")
print("=" * 70)

df = spark.read.parquet(
    "data/processed/final_recipe_dataset"
)

# 1. Load dataset

cluster_df = (
    df.select(
        "recipe_id",
        "name",
        "ingredients",
        "tags",
        "average_rating",
        "total_reviews",
        "tfidf_features"
    )
    .na.drop()
)



# 3. Convert PCA features to NumPy

sample_df = (
    cluster_df
    .sample(False, 0.10, seed=42)   # Sample first
    .limit(50000)                   # Then limit
    .withColumn(
        "features_array",
        vector_to_array("tfidf_features")
    )
)

pdf = sample_df.select(
    "recipe_id",
    "name",
    "ingredients",
    "tags",
    "average_rating",
    "total_reviews",
    "features_array"
).toPandas()

print("Rows:", len(pdf))

if len(pdf) == 0:
    raise ValueError("No data available after sampling.")


X = np.array(pdf["features_array"].tolist())

print("Original Shape :", X.shape)


#2. apply pca
print("\nApplying PCA...")

n_components = min(50, X.shape[0], X.shape[1])

pca = PCA(
    n_components=n_components,
    random_state=42
)

X = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_.sum()

print("PCA Shape :", X.shape)
print("Explained Variance :", round(explained_variance, 4))


# =====================================================
# FIND BEST K
# =====================================================

print("\nFinding Best K")

best_k = 2
best_score = -1

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"K={k}  Silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K :", best_k)


# =====================================================
# TRAIN CLUSTERING MODELS
# =====================================================

models = {

    "KMeans":
        KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        ),

    "MiniBatchKMeans":
        MiniBatchKMeans(
            n_clusters=best_k,
            random_state=42,
            batch_size=1024
        ),

}

results = []

best_model = None
best_model_name = ""
best_score = -1

for name, model in models.items():

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"{name} : {score:.4f}")

    results.append([name, score])

    if score > best_score:

        best_score = score
        best_model = model
        best_model_name = name


print("\nBest Clustering Model :", best_model_name)
print("Best Silhouette :", round(best_score, 4))



# =====================================================
# SAVE OUTPUTS
# =====================================================

print("\nSaving Outputs...")

pdf["cluster"] = best_model.labels_
os.makedirs("models", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

pdf.to_csv(
    "data/output/clustering_results.csv",
    index=False
)

joblib.dump(
    pca,
    "models/pca_model.pkl"
)

print("Saved : models/pca_model.pkl")


joblib.dump(
    best_model,
    "models/best_clustering_model.pkl"
)

print("Best Model :", best_model_name)
print("Silhouette :", round(best_score, 4))
print("Saved : models/best_clustering_model.pkl")

# =====================================================
# STOP SPARK
# =====================================================

spark.stop()

print("\nClustering Completed Successfully")