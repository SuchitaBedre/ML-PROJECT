# =====================================================
# build_health_score.py
# Generate health scores for all recipes
# =====================================================

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

from Backend.services.health_score_service import (
    predict_health_score,
    get_health_category
)


load_dotenv()


# -------------------------------------
# PostgreSQL connection
# -------------------------------------

def get_connection():

    return psycopg2.connect(

        host="localhost",
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")

    )



# -------------------------------------
# Load Recipe Dataset
# -------------------------------------

DATA_PATH = os.path.join(
    "data",
    "processed",
    "final_recipe_dataset"
)


recipes = pd.read_parquet(DATA_PATH)


print(
    "Total Recipes:",
    len(recipes)
)



# -------------------------------------
# Insert into PostgreSQL
# -------------------------------------

conn = get_connection()

cursor = conn.cursor()



for _, row in tqdm(
    recipes.iterrows(),
    total=len(recipes)
):


    recipe_name = row["name"]

    ingredient = row["ingredients"]


    try:


        score = predict_health_score(
            recipe_name,
            ingredient
        )


        category = get_health_category(
            score
        )



        cursor.execute(
            """
            INSERT INTO nutrition_health
            (
                recipe_name,
                ingredient,
                health_score,
                health_category
            )

            VALUES
            (%s,%s,%s,%s)


            ON CONFLICT DO NOTHING

            """,

            (
                recipe_name,
                ingredient,
                score,
                category
            )
        )



    except Exception as e:

        print(
            "Error:",
            recipe_name,
            e
        )



conn.commit()


cursor.close()

conn.close()


print(
    "Health score insertion completed"
)