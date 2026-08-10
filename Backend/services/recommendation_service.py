# =====================================================
# recommendation_service.py
# =====================================================

import os
import pandas as pd

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

recipes = pd.read_parquet(
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "final_recipe_dataset"
    )
)


def recommend_by_ingredients(ingredients):

    ingredient_list = [
        x.strip().lower()
        for x in ingredients.split(",")
        if x.strip()
    ]

    df = recipes.copy()

    df["ingredients"] = (
        df["ingredients"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    result = df[
        df["ingredients"].apply(
            lambda x: all(item in x for item in ingredient_list)
        )
    ]

    if result.empty:

        return {
            "status": "not_found",
            "message": "No similar recipes found."
        }

    result = result.sort_values(
        by="average_rating",
        ascending=False
    ).head(3)

    recommendations = []

    for _, row in result.iterrows():

        recommendations.append({

            "recipe_name": row["name"],

            "ingredients": row["ingredients"],

            "rating": float(row["average_rating"]),

            "cooking_time": int(row["minutes"])

        })

    return {

        "status": "success",

        "recommendations": recommendations

    }