# =====================================================
# 14_health_score_feature_engineering.py
# AI Powered Recipe Recommendation and Rating Prediction
# Health Score Feature Engineering
# =====================================================


import os
import ast
import re
import pandas as pd



# =====================================================
# PATH CONFIGURATION
# =====================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recipe_features_clean"
)



OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recipe_health_features.parquet"
)





# =====================================================
# HEALTH INGREDIENT RULES
# =====================================================


HEALTHY_WORDS = [

    "vegetable",
    "spinach",
    "broccoli",
    "carrot",
    "tomato",
    "cucumber",
    "lettuce",
    "fruit",
    "apple",
    "banana",
    "berry",
    "oat",
    "whole grain",
    "quinoa",
    "lentil",
    "bean",
    "chicken",
    "fish",
    "egg",
    "nut"

]



UNHEALTHY_WORDS = [

    "sugar",
    "cream",
    "butter",
    "fried",
    "deep fry",
    "processed",
    "bacon",
    "sausage",
    "mayonnaise",
    "oil",
    "chocolate"

]






# =====================================================
# LOAD DATA
# =====================================================


def load_data():


    print(
        "\nLoading Recipe Dataset..."
    )


    df = pd.read_parquet(
        INPUT_FILE
    )


    print(
        f"Records : {len(df)}"
    )


    print(
        df.columns.tolist()
    )


    return df






# =====================================================
# PARSE NUTRITION
# =====================================================


def parse_nutrition(value):


    nutrition = {

        "calories":0,

        "fat":0,

        "protein":0,

        "carbohydrates":0

    }



    try:


        if isinstance(value,str):

            value = ast.literal_eval(
                value
            )



        if isinstance(value,dict):


            for key in nutrition:


                if key in value:


                    nutrition[key] = float(
                        value[key]
                    )


    except Exception:


        pass



    return nutrition






# =====================================================
# CALORIE SCORE
# =====================================================


def calorie_score(calories):


    if calories <= 200:

        return 3


    elif calories <=400:

        return 2


    elif calories <=700:

        return 1


    else:

        return 0






# =====================================================
# FAT SCORE
# =====================================================


def fat_score(fat):


    if fat <=10:

        return 2


    elif fat <=20:

        return 1


    else:

        return 0






# =====================================================
# PROTEIN SCORE
# =====================================================


def protein_score(protein):


    if protein >=20:

        return 2


    elif protein >=10:

        return 1


    else:

        return 0






# =====================================================
# INGREDIENT SCORE
# =====================================================


def ingredient_score(text):


    text = str(text).lower()



    score = 0



    for item in HEALTHY_WORDS:


        if item in text:


            score += 0.2



    for item in UNHEALTHY_WORDS:


        if item in text:


            score -=0.2




    if score > 2:

        score = 2


    if score <0:

        score =0



    return round(
        score,
        2
    )







# =====================================================
# TIME SCORE
# =====================================================


def time_score(minutes):


    try:


        minutes=float(minutes)


    except:


        return 0



    if minutes <=30:

        return 1


    return 0






# =====================================================
# HEALTH SCORE GENERATION
# =====================================================


def create_health_score(df):


    print(
        "\nCreating Health Score..."
    )


    scores=[]



    for _,row in df.iterrows():


        total=0



        nutrition=parse_nutrition(

            row.get(
                "nutrition",
                {}
            )

        )



        total += calorie_score(

            nutrition["calories"]

        )



        total += fat_score(

            nutrition["fat"]

        )



        total += protein_score(

            nutrition["protein"]

        )



        ingredients = (

            str(
                row.get(
                    "ingredients",
                    ""
                )
            )

        )



        total += ingredient_score(

            ingredients

        )



        total += time_score(

            row.get(
                "minutes",
                999
            )

        )



        scores.append(

            round(

                min(
                    total,
                    10
                ),

                2

            )

        )



    df["health_score"]=scores



    return df







# =====================================================
# MAIN
# =====================================================


def main():


    df=load_data()



    df=create_health_score(

        df

    )



    print(

        "\nHealth Score Distribution"

    )


    print(

        df["health_score"].describe()

    )




    os.makedirs(

        os.path.dirname(
            OUTPUT_FILE
        ),

        exist_ok=True

    )



    df.to_parquet(

        OUTPUT_FILE,

        index=False

    )



    print(

        "\n================================"

    )


    print(

        "Health Feature Engineering Completed"

    )


    print(

        OUTPUT_FILE

    )


    print(

        "================================"

    )







if __name__=="__main__":

    main()