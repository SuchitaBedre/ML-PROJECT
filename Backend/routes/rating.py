# =====================================================
# IMPORT LIBRARIES
# =====================================================

from fastapi import APIRouter, HTTPException
import pandas as pd
import os

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

recipes = pd.read_parquet(
    os.path.join(BASE_DIR, "data", "processed", "final_recipe_dataset")
)

# =====================================================
# GET STORED RATING
# =====================================================

@router.get("/{recipe_name}")
def get_rating(recipe_name: str):

    recipe = recipes[
        recipes["name"].fillna("").str.lower() == recipe_name.lower()
    ]

    if recipe.empty:
        raise HTTPException(
            status_code=404,
            detail="Recipe not found."
        )

    return {
        "status": "success",
        "recipe_id": int(recipe.iloc[0]["recipe_id"]),
        "recipe_name": recipe.iloc[0]["name"],
        "average_rating": float(recipe.iloc[0]["average_rating"]),
        "total_reviews": int(recipe.iloc[0]["total_reviews"])
    }