# ============================================================
# merge_sentiment.py
# AI Powered Recipe Recommendation and Rating Prediction Platform
# Merge Sentiment Features with Final Recipe Dataset
# ============================================================

import os
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MAIN_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_recipe_dataset"
)

SENTIMENT_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recipe_sentiment_features.parquet"
)

OUTPUT_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_recipe_dataset_with_sentiment.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading recipe dataset...")

recipes = pd.read_parquet(MAIN_DATASET)

print("Recipe Dataset Shape:")
print(recipes.shape)


print("\nLoading sentiment dataset...")

sentiment = pd.read_parquet(SENTIMENT_DATASET)

print("Sentiment Dataset Shape:")
print(sentiment.shape)


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "recipe_id" not in recipes.columns:
    raise Exception(
        "recipe_id missing from recipe dataset"
    )


if "recipe_id" not in sentiment.columns:
    raise Exception(
        "recipe_id missing from sentiment dataset"
    )


# ============================================================
# REMOVE DUPLICATE SENTIMENT RECORDS
# ============================================================

print("\nChecking duplicate recipe IDs...")

sentiment = sentiment.drop_duplicates(
    subset=["recipe_id"]
)


# ============================================================
# MERGE
# ============================================================

print("\nMerging sentiment features...")

final_dataset = recipes.merge(
    sentiment,
    on="recipe_id",
    how="left"
)


# ============================================================
# VALIDATION
# ============================================================

print("\nMerged Dataset Shape:")
print(final_dataset.shape)


print("\nSentiment Columns Added:")

new_columns = [
    col for col in sentiment.columns
    if col != "recipe_id"
]

for col in new_columns:
    if col in final_dataset.columns:
        print("✔", col)


# Missing sentiment count

if "sentiment_label" in final_dataset.columns:

    missing = final_dataset["sentiment_label"].isna().sum()

    print(
        "\nRecipes without sentiment:",
        missing
    )


# ============================================================
# SAVE FINAL DATASET
# ============================================================

print("\nSaving final dataset...")

final_dataset.to_parquet(
    OUTPUT_DATASET,
    index=False
)


print("\n======================================")
print("MERGE COMPLETED SUCCESSFULLY")
print("Saved:")
print(OUTPUT_DATASET)
print("======================================")