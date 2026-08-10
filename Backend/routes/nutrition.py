# =====================================================
# nutrition.py
# Nutrition + Health Score API
# Existing + Unseen Recipe Support
# =====================================================


import os
import ast
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


from Backend.services.health_score_service import (
    get_or_create_health_score
)



# =====================================================
# ROUTER
# =====================================================


router = APIRouter(

    prefix="/nutrition",

    tags=["Nutrition"]

)



# =====================================================
# REQUEST MODEL
# =====================================================


class NutritionRequest(BaseModel):

    recipe_name: str

    ingredient: str



# =====================================================
# LOAD DATASET
# =====================================================


BASE_DIR = os.path.abspath(

    os.path.join(

        os.path.dirname(__file__),

        "..",

        ".."

    )

)



DATA_PATH = os.path.join(

    BASE_DIR,

    "data",

    "processed",

    "final_recipe_dataset"

)



recipes = pd.read_parquet(

    DATA_PATH

)



print("="*60)

print("Nutrition Dataset Loaded")

print("Total Recipes:", len(recipes))

print("="*60)




# =====================================================
# API
# =====================================================


@router.post("/")
def nutrition(

    request: NutritionRequest

):


    recipe_name = request.recipe_name

    ingredient = request.ingredient



    # ---------------------------------------------
    # Search recipe in dataset
    # ---------------------------------------------


    recipe = recipes[

        recipes["name"]

        .fillna("")

        .str.lower()

        ==

        recipe_name.lower()

    ]



    # =================================================
    # EXISTING RECIPE
    # =================================================


    if not recipe.empty:


        recipe_row = recipe.iloc[0]



        nutrition_value = recipe_row["nutrition"]



        if isinstance(

            nutrition_value,

            str

        ):

            nutrition_value = ast.literal_eval(

                nutrition_value

            )



        health = get_or_create_health_score(

            recipe_name=recipe_row["name"],

            ingredient=recipe_row["ingredients"]

        )



        return {


            "status":"success",


            "type":"existing",


            "recipe_name":

                recipe_row["name"],



            "nutrition":{


                "calories":

                    nutrition_value[0],


                "total_fat_percent":

                    nutrition_value[1],


                "sugar_percent":

                    nutrition_value[2],


                "sodium_percent":

                    nutrition_value[3],


                "protein_percent":

                    nutrition_value[4],


                "saturated_fat_percent":

                    nutrition_value[5],


                "carbohydrates_percent":

                    nutrition_value[6]

            },


            "health_score":

                health["health_score"],



            "health_category":

                health["health_category"]

        }





    # =================================================
    # UNSEEN RECIPE
    # =================================================


    health = get_or_create_health_score(

        recipe_name=recipe_name,

        ingredient=ingredient

    )



    return {


        "status":"success",


        "type":"unseen",


        "recipe_name":

            recipe_name,


        "nutrition":{},



        "health_score":

            health["health_score"],



        "health_category":

            health["health_category"]

    }