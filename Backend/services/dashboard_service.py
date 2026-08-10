import pandas as pd
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "final_recipe_dataset"

df = pd.read_parquet(DATA_PATH)

def get_embedding_count():

    try:
        client = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

        collection = client.get_collection(
            name="recipes"
        )

        return collection.count()

    except Exception as e:
        print("ChromaDB Error:", e)
        return 0


def get_dashboard_data():

    total_recipes = len(df)

    avg_rating = round(df["average_rating"].mean(), 2)

    total_reviews = int(df["total_reviews"].sum())

    avg_minutes = round(df["minutes"].mean(), 1)

    rating_distribution = (
        df["average_rating"]
        .round()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    cooking_distribution = (
        df["minutes"]
        .value_counts()
        .sort_index()
        .head(30)
        .to_dict()
    )

    top_recipes = (
        df.sort_values(
            by="average_rating",
            ascending=False
        )
        [["name", "average_rating"]]
        .head(10)
        .to_dict("records")
    )

    ingredient_counts = (
        df["n_ingredients"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {

        "total_recipes": total_recipes,

        "avg_rating": avg_rating,

        "total_reviews": total_reviews,

        "avg_minutes": avg_minutes,

        "embedding_count": get_embedding_count(),

        "rating_distribution": rating_distribution,

        "cooking_distribution": cooking_distribution,

        "ingredient_distribution": ingredient_counts,

        "top_recipes": top_recipes

    }