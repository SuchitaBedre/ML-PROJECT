import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer


# change this path according to your dataset
DATA_PATH = "../data/processed/final_recipe_dataset"


print("Loading dataset...")


df = pd.read_parquet(DATA_PATH)


print(df.columns)


# combine text columns
df["text"] = (
    df["ingredients"].fillna("").astype(str)
    + " "
    + df["tags"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
)


print("Creating TF-IDF...")


tfidf = TfidfVectorizer(
    max_features=5000
)


tfidf.fit(df["text"])


# save
os.makedirs(
    "../models",
    exist_ok=True
)


with open(
    "../models/tfidf_vectorizer.pkl",
    "wb"
) as f:
    pickle.dump(tfidf, f)


print("TF-IDF vectorizer saved successfully")