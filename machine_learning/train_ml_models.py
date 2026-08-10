import os
import joblib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================================
# Spark Session
# =====================================================

spark = (
    SparkSession.builder
    .appName("Recipe ML Models")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# =====================================================
# Load Dataset
# =====================================================

print("="*60)
print("Loading Dataset")
print("="*60)

df = spark.read.parquet(
    "data/processed/final_recipe_dataset"
)

print("Total Records :", df.count())

# =====================================================
# Create Labels
# =====================================================

df = df.withColumn(
    "label",
    when(col("rating") < 3, 0)
    .when(
        (col("rating") >= 3) &
        (col("rating") < 4),
        1
    )
    .otherwise(2)
)

df = df.select(
    "text_features",
    "label"
)

# =====================================================
# Faster Training
# =====================================================

print("\nSampling Dataset (20%)...")

df = df.sample(
    fraction=0.20,
    seed=42
)

print("Sample Records :", df.count())

# =====================================================
# Spark -> Pandas
# =====================================================

print("\nConverting Spark DataFrame to Pandas...")

pdf = df.toPandas()

print("Conversion Complete")

# =====================================================
# Convert Vector to NumPy
# =====================================================

print("Preparing Features...")

X = np.vstack(
    pdf["text_features"].apply(lambda x: x.toArray())
)

y = pdf["label"].values

print("Feature Shape :", X.shape)

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training :", len(X_train))
print("Testing  :", len(X_test))

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=500
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            eval_metric="mlogloss",
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            verbose=False,
            random_state=42
        )
}

# =====================================================
# Create folders
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []

# =====================================================
# Train Models
# =====================================================

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(
        y_test,
        pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        pred,
        average="weighted"
    )

    print("Accuracy :", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1 Score :", round(f1,4))

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            pred
        )
    )

    filename = (
        "models/"
        + name.lower().replace(" ","_")
        + ".pkl"
    )

    joblib.dump(
        model,
        filename
    )

    print("Saved :", filename)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

# =====================================================
# Save Results
# =====================================================

result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

result_df.to_csv(
    "results/model_results.csv",
    index=False
)

print("\n")
print("="*70)
print(result_df)
print("="*70)

spark.stop()