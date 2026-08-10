import os
import sys
import numpy as np


# ======================================================
# PYTHON CONFIGURATION
# ======================================================

PYTHON_PATH = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH



# ======================================================
# IMPORTS
# ======================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf
from pyspark.sql.types import DoubleType

from pyspark.ml.functions import vector_to_array



# ======================================================
# CREATE SPARK SESSION
# ======================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Recipe Recommendation System")

    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.driver.maxResultSize", "4g")

    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.default.parallelism", "16")

    .config("spark.python.worker.reuse", "true")

    .getOrCreate()
)



spark.sparkContext.setLogLevel("ERROR")



print("=" * 60)
print("Recipe Recommendation System")
print("=" * 60)

print("Spark Version :", spark.version)
print("Python :", sys.executable)

print("=" * 60)



# ======================================================
# LOAD DATASET
# ======================================================

print("\nLoading Dataset...")


df = spark.read.parquet(

    "data/processed/final_recipe_dataset"

)



print(
    "Total Recipes :",
    df.count()
)



# ======================================================
# SELECT REQUIRED COLUMNS
# ======================================================


df = df.select(

    "recipe_id",

    "name",

    "ingredients",

    "tags",

    "description",

    "review",

    "minutes",

    "n_ingredients",

    "average_rating",

    "total_reviews",

    "tfidf_features"

)



# ======================================================
# REMOVE NULL VALUES
# ======================================================


df = df.filter(

    col("name").isNotNull()

    &

    col("tfidf_features").isNotNull()

    &

    col("average_rating").isNotNull()

)



# ======================================================
# REPARTITION
# ======================================================


df = df.repartition(16)



print(

    "Clean Dataset Count :",

    df.count()

)




# ======================================================
# CONVERT VECTOR TO ARRAY
# ======================================================


df = df.withColumn(

    "features_array",

    vector_to_array(

        "tfidf_features"

    )

)



# ======================================================
# SEARCH RECIPE
# ======================================================


keyword = "caramel"



matching_recipes = (

    df.filter(

        lower(col("name"))

        .contains(

            keyword.lower()

        )

    )

    .select(

        "recipe_id",

        "name"

    )

)



print("\nRecipes Found:")



matching_recipes.show(

    20,

    truncate=False

)



if matching_recipes.count() == 0:


    print(

        "\nNo recipe found:",

        keyword

    )


    spark.stop()

    quit()




recipe_name = matching_recipes.first()["name"]



print(

    "\nSelected Recipe:",

    recipe_name

)





# ======================================================
# GET TARGET VECTOR
# ======================================================


target_vector = (

    df.filter(

        lower(col("name"))

        ==

        recipe_name.lower()

    )

    .select(

        "features_array"

    )

    .first()[0]

)



target_vector = np.array(

    target_vector

)





# ======================================================
# COSINE SIMILARITY FUNCTION
# ======================================================


def cosine_similarity(features):


    features = np.array(features)


    denominator = (

        np.linalg.norm(target_vector)

        *

        np.linalg.norm(features)

    )


    if denominator == 0:

        return 0.0



    return float(

        np.dot(

            target_vector,

            features

        )

        /

        denominator

    )





similarity_udf = udf(

    cosine_similarity,

    DoubleType()

)





# ======================================================
# GENERATE RECOMMENDATIONS
# ======================================================


recommendations = (

    df.withColumn(

        "similarity",

        similarity_udf(

            col("features_array")

        )

    )

    .filter(

        lower(col("name"))

        !=

        recipe_name.lower()

    )

)





# ======================================================
# TOP 3 RESULTS
# ======================================================


top_recommendations = (

    recommendations

    .orderBy(

        col("similarity").desc()

    )

    .limit(3)

)





print("\n")

print("=" * 70)

print("TOP 3 RECOMMENDED RECIPES")

print("=" * 70)



top_recommendations.select(

    "recipe_id",

    "name",

    "ingredients",

    "average_rating",

    "total_reviews",

    "similarity"

).show(

    truncate=False

)





# ======================================================
# SAVE OUTPUT
# ======================================================


# ======================================================
# SAVE OUTPUT
# ======================================================

output_path = "data/output/recommendations"

(
    top_recommendations
    .write
    .mode("overwrite")
    .parquet(output_path)
)


print("\nRecommendation Saved Successfully")

print(
    "Location:",
    output_path
)



# ======================================================
# STOP SPARK
# ======================================================


spark.stop()



print("\nSpark Session Closed")