import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://recipe_user:recipe123@localhost:5432/recipe_db"
)

print("Loading Parquet dataset...")

df = pd.read_parquet("data/processed/final_recipe_dataset")

# Drop the TF-IDF vectors (they belong in ChromaDB/ML, not PostgreSQL)
df = df.drop(columns=["tfidf_features"])

print(df.head())
print("Total Records:", len(df))

print("Importing into PostgreSQL...")

df.to_sql(
    "recipes",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=1000
)

print("Import completed successfully!")