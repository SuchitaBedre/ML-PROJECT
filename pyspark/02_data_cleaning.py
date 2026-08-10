import os
import sys

# =====================================================
# FORCE PYTHON VERSION
# =====================================================

PYTHON_PATH = r"C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH
os.environ["PYTHONHASHSEED"] = "0"

print("=" * 60)
print("Driver Python :", sys.executable)
print("Worker Python :", os.environ["PYSPARK_PYTHON"])
print("=" * 60)

# =====================================================
# CREATE SPARK SESSION
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Recipe Data Cleaning")

    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.driver.maxResultSize", "4g")

    .config("spark.default.parallelism", "8")
    .config("spark.sql.shuffle.partitions", "8")

    .config("spark.python.worker.reuse", "true")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.sql.adaptive.enabled", "true")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# =====================================================
# READ RAW DATA
# =====================================================

input_path = "data/raw/merged_food_dataset_cleaned.csv.gz"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv(input_path)
)

print("=" * 60)
print("Original Records :", df.count())
print("=" * 60)

# =====================================================
# DATA CLEANING
# =====================================================

# Remove duplicate rows
df = df.dropDuplicates()

# Remove rows where all columns are NULL
df = df.na.drop(how="all")

# Remove rows with missing important columns
df = df.filter(col("recipe_id").isNotNull())
df = df.filter(col("name").isNotNull())
df = df.filter(col("rating").isNotNull())

# Convert rating to numeric
df = df.withColumn("rating", col("rating").cast("double"))

# Keep only valid ratings (0-5)
df = df.filter((col("rating") >= 0) & (col("rating") <= 5))

print("After Cleaning :", df.count())

print("\nSchema")
df.printSchema()

print("\nSample Data")
df.select(
    "recipe_id",
    "name",
    "rating"
).show(10, False)

# =====================================================
# SAVE CLEAN DATA AS PARQUET
# =====================================================

output_path = "data/interim/recipe_interim"

(
    df.write
    .mode("overwrite")
    .parquet(output_path)
)

print("=" * 60)
print("Interim Dataset Saved Successfully")
print(output_path)
print("=" * 60)

spark.stop()