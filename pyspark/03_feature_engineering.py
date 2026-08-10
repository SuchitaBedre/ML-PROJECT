# =====================================================
# 03_feature_engineering.py
# AI Powered Recipe Recommendation and Rating Prediction
# =====================================================

import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    lower,
    regexp_replace,
    when,
    split
)

from pyspark.ml.feature import (
    Tokenizer,
    HashingTF,
    IDF
)

from pyspark.ml import Pipeline



# =====================================================
# CREATE SPARK SESSION
# =====================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Recipe Feature Engineering")

    # Memory Configuration
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.driver.maxResultSize", "4g")

    # Performance
    .config("spark.sql.shuffle.partitions", "100")
    .config("spark.default.parallelism", "100")

    # Parquet Memory Fix
    .config(
        "spark.sql.parquet.enableVectorizedReader",
        "false"
    )

    .config(
        "spark.sql.parquet.columnarReaderBatchSize",
        "512"
    )

    .getOrCreate()
)


spark.sparkContext.setLogLevel("ERROR")



# =====================================================
# PATHS
# =====================================================


input_path = "data/interim/recipe_interim"

output_path = "data/processed/recipe_features"



# =====================================================
# LOAD CLEAN DATA
# =====================================================


print("=" * 60)
print("Loading Clean Dataset")
print("=" * 60)


df = spark.read.parquet(input_path)



print("Available Columns:")
print(df.columns)


print("Total Records:")
print(df.count())



# =====================================================
# SELECT IMPORTANT COLUMNS
# =====================================================


required_columns = [

    "recipe_id",
    "name",
    "ingredients",
    "tags",
    "description",
    "steps",
    "nutrition",
    "rating",
    "review",
    "minutes",
    "n_ingredients"

]


available_columns = [

    c for c in required_columns

    if c in df.columns

]


df = df.select(available_columns)



# =====================================================
# HANDLE NULL VALUES
# =====================================================


print("Handling Missing Values...")


numeric_columns = [

    "rating",
    "minutes",
    "n_ingredients"

]


for c in numeric_columns:

    if c in df.columns:

        df = df.withColumn(

            c,

            when(
                col(c).isNull(),
                0
            )
            .otherwise(col(c))

        )



text_columns = [

    "name",
    "ingredients",
    "tags",
    "description",
    "steps",
    "nutrition",
    "review"

]


for c in text_columns:

    if c in df.columns:

        df = df.withColumn(

            c,

            when(
                col(c).isNull(),
                ""
            )
            .otherwise(col(c))

        )

# =====================================================
# NUTRITION FEATURE ENGINEERING
# =====================================================

print("Extracting Nutrition Features...")

df = df.withColumn(
    "nutrition_clean",
    regexp_replace(col("nutrition"), r"[\[\]]", "")
)

df = df.withColumn(
    "nutrition_array",
    split(col("nutrition_clean"), ",")
)

df = (
    df
    .withColumn("calories",
                col("nutrition_array")[0].cast("double"))

    .withColumn("total_fat",
                col("nutrition_array")[1].cast("double"))

    .withColumn("sugar",
                col("nutrition_array")[2].cast("double"))

    .withColumn("sodium",
                col("nutrition_array")[3].cast("double"))

    .withColumn("protein",
                col("nutrition_array")[4].cast("double"))

    .withColumn("saturated_fat",
                col("nutrition_array")[5].cast("double"))

    .withColumn("carbohydrates",
                col("nutrition_array")[6].cast("double"))
)

nutrition_columns = [
    "calories",
    "total_fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbohydrates"
]

for c in nutrition_columns:
    df = df.withColumn(
        c,
        when(col(c).isNull(), 0.0).otherwise(col(c))
    )

df = df.drop(
    "nutrition_clean",
    "nutrition_array"
)

# =====================================================
# TEXT CLEANING
# =====================================================


# =====================================================
# TEXT CLEANING FOR NLP ONLY
# =====================================================

print("Cleaning Text Data...")

if "ingredients" in df.columns:
    df = df.withColumn(
        "ingredients_clean",
        lower(
            regexp_replace(
                col("ingredients"),
                "[^a-zA-Z ]",
                " "
            )
        )
    )

if "tags" in df.columns:
    df = df.withColumn(
        "tags_clean",
        lower(
            regexp_replace(
                col("tags"),
                "[^a-zA-Z ]",
                " "
            )
        )
    )

if "description" in df.columns:
    df = df.withColumn(
        "description_clean",
        lower(
            regexp_replace(
                col("description"),
                "[^a-zA-Z ]",
                " "
            )
        )
    )

if "steps" in df.columns:
    df = df.withColumn(
        "steps_clean",
        lower(
            regexp_replace(
                col("steps"),
                "[^a-zA-Z ]",
                " "
            )
        )
    )



# =====================================================
# CREATE COMBINED RECIPE TEXT
# =====================================================


# =====================================================
# CREATE TEXT FOR TF-IDF
# =====================================================

print("Creating Recipe Text...")

text_columns = []

if "name" in df.columns:
    text_columns.append(col("name"))

if "ingredients_clean" in df.columns:
    text_columns.append(col("ingredients_clean"))

if "tags_clean" in df.columns:
    text_columns.append(col("tags_clean"))

if "description_clean" in df.columns:
    text_columns.append(col("description_clean"))

if "steps_clean" in df.columns:
    text_columns.append(col("steps_clean"))

df = df.withColumn(
    "recipe_text",
    concat_ws(" ", *text_columns)
)

# =====================================================
# TOKENIZATION
# =====================================================


print("Tokenizing Text...")


tokenizer = Tokenizer(

    inputCol="recipe_text",

    outputCol="words"

)



# =====================================================
# HASHING TF
# =====================================================


hashingTF = HashingTF(

    inputCol="words",

    outputCol="raw_features",

    numFeatures=5000

)



# =====================================================
# IDF
# =====================================================


idf = IDF(

    inputCol="raw_features",

    outputCol="tfidf_features"

)



# =====================================================
# PIPELINE
# =====================================================


pipeline = Pipeline(

    stages=[

        tokenizer,

        hashingTF,

        idf

    ]

)



# =====================================================
# GENERATE FEATURES
# =====================================================


print("Generating TF-IDF Features...")


model = pipeline.fit(df)


feature_df = model.transform(df)

# =====================================================
# SAVE FEATURE PIPELINE
# =====================================================

pipeline_path = "models/tfidf_pipeline"

if os.path.exists(pipeline_path):
    shutil.rmtree(pipeline_path)

model.write().save(pipeline_path)

print("TF-IDF Pipeline Saved Successfully")

# =====================================================
# FINAL DATASET
# =====================================================


final_columns = [

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

    "tfidf_features",

    "steps",
    "nutrition"
]

final_columns = [

    c for c in final_columns

    if c in feature_df.columns

]


final_df = feature_df.select(final_columns)



# =====================================================
# SAVE DATASET
# =====================================================


print("Saving Feature Dataset...")


final_df = final_df.coalesce(8)


(
    final_df.write
    .mode("overwrite")
    .parquet(output_path)
)



# =====================================================
# VALIDATION
# =====================================================


print("=" * 60)
print("Feature Engineering Completed Successfully")
print("=" * 60)


print("Output Path:")
print(output_path)


print("Total Records:")
print(final_df.count())


print("Sample Output:")


final_df.select(

    "recipe_id",
    "name",
    "rating"

).limit(5).show(
    truncate=False
)



# =====================================================
# STOP SPARK
# =====================================================


spark.stop()