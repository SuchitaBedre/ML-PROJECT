from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
import joblib


# =====================================
# Spark Session
# =====================================

spark = (
    SparkSession.builder
    .appName("Recipe Rating Prediction")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.shuffle.partitions", "50")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =====================================
# Load Dataset
# =====================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = spark.read.parquet(
    "data/processed/final_recipe_dataset"
)

print("Total Records :", df.count())


# =====================================
# Create Labels
# =====================================

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
    col("text_features").alias("features"),
    "label"
)


# =====================================
# Faster Training
# =====================================

df = df.sample(
    fraction=0.60,
    seed=42
)

print("\nLabel Distribution")

df.groupBy("label").count().show()


# =====================================
# Train Test Split
# =====================================

train, test = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

train.cache()
test.cache()

print("Train :", train.count())
print("Test  :", test.count())


# =====================================
# Evaluator
# =====================================

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)


# =====================================
# Models
# =====================================

models = {

    "logistic_regression": LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        regParam=0.05
    ),

    "decision_tree": DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=8
    ),

    "random_forest": RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=20,
        maxDepth=8,
        seed=42
    )

}


# =====================================
# Create models folder
# =====================================

os.makedirs("models", exist_ok=True)

results = []

# =====================================
# Train Models
# =====================================

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    trained_model = model.fit(train)

    predictions = trained_model.transform(test)

    accuracy = accuracy_eval.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    # Save model

    save_path = f"models/{name}"

    trained_model.write().overwrite().save(save_path)

    print(f"Model saved -> {save_path}")



# =====================================
# Final Results
# =====================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for r in results:
    print(r)

spark.stop()