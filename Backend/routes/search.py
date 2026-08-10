# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import pandas as pd
from fastapi import APIRouter

from Backend.services.prediction_service import predict_rating

# =====================================================
# CREATE ROUTER
# =====================================================

router = APIRouter()

# =====================================================
# LOAD DATASET
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

try:

    recipes = pd.read_parquet(
        os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "recipe_features"
        )
    )

    recipes["name_lower"] = (
        recipes["name"]
        .fillna("")
        .str.lower()
    )

    print("===================================")
    print("Recipe Dataset Loaded Successfully")
    print(f"Total Recipes : {len(recipes)}")
    print("===================================")

except Exception as e:

    print(e)
    recipes = pd.DataFrame()


# =====================================================
# SEARCH RECIPE
# =====================================================

@router.get("/{recipe_name}")
def search_recipe(recipe_name: str):

    if recipes.empty:

        return {
            "status": "error",
            "message": "Recipe dataset not found."
        }

    recipe_name = recipe_name.lower().strip()

    result = recipes[
        recipes["name_lower"].str.contains(
            recipe_name,
            regex=False,
            na=False
        )
    ]

    if result.empty:

        return {
            "status": "not_found",
            "message": f"No recipe found for '{recipe_name}'."
        }

    recipe = result.iloc[0]

    # =====================================================
    # PREDICT RATING
    # =====================================================

    try:

        prediction = predict_rating(recipe["name"])

        predicted_rating = prediction["predicted_rating"]

    except Exception as e:

        print("Prediction Error:", e)

        predicted_rating = None

    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "status": "success",

        "recipe_id": int(recipe["recipe_id"]),

        "recipe_name": recipe["name"],

        "ingredients": recipe["ingredients"],

        "cooking_time_minutes": int(recipe["minutes"]),

        "description": recipe["description"],

        "predicted_rating": predicted_rating,

        "cooking_steps":
            recipe["steps"]
            if "steps" in recipes.columns
            else "Steps not available"

    }