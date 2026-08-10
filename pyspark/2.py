import pandas as pd

df = pd.read_parquet(
"data/processed/recipe_sentiment_features.parquet"
)

print(df.columns)
print(df.head())