from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Recipe ETL Pipeline")
    .getOrCreate()
)

df = (
    spark.read
    .option("header", "true")
    .option("multiLine", "true")
    .option("escape", "\"")
    .option("quote", "\"")
    .option("inferSchema", "true")
    .csv("data/raw/merged_food_dataset_cleaned.csv.gz")
)

print("=" * 60)
print("Schema")
print("=" * 60)

df.printSchema()

print("=" * 60)
print("First 5 Records")
print("=" * 60)

df.show(5, truncate=False)

print("=" * 60)
print("Total Records:", df.count())

spark.stop()