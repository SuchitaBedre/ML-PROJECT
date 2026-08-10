# =====================================================
# 04_save_processed.py
# AI Powered Recipe Recommendation and Rating Prediction
# =====================================================


import os
import sys


# =====================================================
# PYTHON CONFIGURATION
# =====================================================

PYTHON_PATH = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH



# =====================================================
# SPARK SESSION
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    count
)


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Create Final Recipe Dataset")

    .config(
        "spark.driver.memory",
        "8g"
    )

    .config(
        "spark.executor.memory",
        "8g"
    )

    .config(
        "spark.sql.shuffle.partitions",
        "8"
    )

    .config(
        "spark.sql.parquet.enableVectorizedReader",
        "false"
    )

    .getOrCreate()
)


spark.sparkContext.setLogLevel("ERROR")



# =====================================================
# INPUT PATH
# =====================================================


input_path = "data/processed/recipe_features"



# =====================================================
# LOAD FEATURE DATA
# =====================================================


print("=" * 60)
print("FEATURE DATASET")
print("=" * 60)


df = spark.read.parquet(input_path)



print("Records :", df.count())

print("Columns :", df.columns)



# =====================================================
# VALIDATE REQUIRED COLUMNS
# =====================================================


required_columns = [
    "recipe_id",
    "name",
    "ingredients",
    "tags",
    "description",
    "review",
    "rating",
    "minutes",
    "n_ingredients",

    "calories",
    "total_fat",
    "protein",
    "carbohydrates",
    "sugar",
    "sodium",
    "saturated_fat",

    "steps",
    "nutrition",
    "tfidf_features"
]


for c in required_columns:

    if c not in df.columns:

        raise Exception(
            f"Missing column : {c}"
        )



# =====================================================
# AGGREGATE RECIPE RATINGS
# =====================================================


print("Aggregating recipe ratings...")


final_df = (

    df

    .groupBy(
        "recipe_id",
        "name",
        "ingredients",
        "tags",
        "description",
        "review",
        "minutes",
        "n_ingredients",

        "calories",
        "total_fat",
        "protein",
        "carbohydrates",
        "sugar",
        "sodium",
        "saturated_fat",

        "steps",
        "nutrition",
        "tfidf_features"
        )
    .agg(

        avg("rating")
        .alias("average_rating"),


        count("rating")
        .alias("total_reviews")

    )

)



# =====================================================
# REMOVE DUPLICATE RECIPES
# =====================================================


final_df = (

    final_df

    .dropDuplicates(
        [
            "recipe_id"
        ]
    )

)



# =====================================================
# REMOVE NULL VALUES
# =====================================================


final_df = (

    final_df

    .na.drop()

)



# =====================================================
# VALIDATION
# =====================================================


print("=" * 60)
print("FINAL DATASET")
print("=" * 60)



#print(
#    "Final Records :",
 #   final_df.count()
#)



print("\nSchema")

final_df.printSchema()



print("\nSample Data")

final_df.select(
    "recipe_id",
    "name",
    "average_rating",
    "total_reviews",
    "calories",
    "total_fat",
    "protein",
    "carbohydrates",
    "sugar",
    "sodium",
    "saturated_fat"
).show(5, False)



# =====================================================
# SAVE FINAL DATASET
# =====================================================


output_path = (
    "data/processed/final_recipe_dataset"
)



print("=" * 60)
print("Saving Final Dataset...")
print("=" * 60)



(
    final_df

    .coalesce(8)

    .write

    .mode("overwrite")

    .parquet(output_path)

    

)



print("=" * 60)
print("Final Dataset Saved Successfully")
print(output_path)
print("=" * 60)



# =====================================================
# STOP SPARK
# =====================================================


spark.stop()