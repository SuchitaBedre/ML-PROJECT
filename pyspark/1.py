from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Check Dataset")
    .getOrCreate()
)

df = spark.read.parquet("data/processed/final_recipe_dataset")

print("\n========== SCHEMA ==========\n")
df.printSchema()

print("\n========== COLUMNS ==========\n")
print(df.columns)

print("\n========== SAMPLE ==========\n")
df.show(5, truncate=False)

spark.stop()

