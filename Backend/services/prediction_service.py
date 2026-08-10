# =====================================================
# IMPORT LIBRARIES
# =====================================================

import numpy as np
import pandas as pd

from Backend.services.model_loader import regression_model


# =====================================================
# LOAD DATASET
# =====================================================

import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

recipes = pd.read_parquet(
    os.path.join(BASE_DIR, "data", "processed", "final_recipe_dataset")
)


# =====================================================
# CONVERT TF-IDF DICTIONARY TO NUMPY
# =====================================================

def vector_to_numpy(vector):

    arr = np.zeros(vector["size"])

    arr[vector["indices"]] = vector["values"]

    return arr


# =====================================================
# PREDICT RATING
# =====================================================

def predict_rating(recipe_name: str):

    recipe = recipes[
    recipes["name"].fillna("").str.lower() == recipe_name.lower()
    ]

    if recipe.empty:

        raise ValueError(
            f"Recipe '{recipe_name}' not found in the dataset."
        )

    feature = recipe.iloc[0]["tfidf_features"]

    X = vector_to_numpy(feature).reshape(1, -1)

    prediction = regression_model.predict(X)

    return {

        "recipe_id": int(recipe.iloc[0]["recipe_id"]),

        "recipe_name": recipe.iloc[0]["name"],

        "predicted_rating": round(float(prediction[0]), 2)

    }