# =====================================================
# health_score_service.py
# Rule Based Health Score Calculation
# =====================================================

import os
import psycopg2
from dotenv import load_dotenv


load_dotenv()


# =====================================================
# PostgreSQL CONNECTION
# =====================================================

def get_connection():

    return psycopg2.connect(

        host="localhost",
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")

    )


# =====================================================
# HEALTH SCORE CALCULATION
# =====================================================

def calculate_health_score(ingredients):

    score = 10


    ingredients = str(ingredients).lower()


    # Negative ingredients

    if "sugar" in ingredients:

        score -= 1


    if "oil" in ingredients:

        score -= 1


    if "butter" in ingredients:

        score -= 1


    if "cream" in ingredients:

        score -= 1



    # Positive ingredients

    if "vegetable" in ingredients:

        score += 1


    if "protein" in ingredients:

        score += 1


    if "fruit" in ingredients:

        score += 1



    # Range 0-10

    score = max(
        0,
        min(
            10,
            score
        )
    )


    return round(
        score,
        2
    )



# =====================================================
# HEALTH CATEGORY
# =====================================================

def get_health_category(score):


    if score >= 8:

        return "Excellent"


    elif score >= 6:

        return "Good"


    elif score >= 4:

        return "Average"


    else:

        return "Low"



# =====================================================
# GET EXISTING OR CREATE NEW
# =====================================================

def get_or_create_health_score(
        recipe_name,
        ingredient
):


    conn = get_connection()

    cursor = conn.cursor()



    # Check existing recipe

    cursor.execute(

        """
        SELECT
        health_score,
        health_category

        FROM nutrition_health

        WHERE recipe_name=%s
        """,

        (
            recipe_name,
        )

    )


    result = cursor.fetchone()



    if result:


        cursor.close()

        conn.close()


        return {

            "health_score": result[0],

            "health_category": result[1]

        }



    # New recipe calculation

    score = calculate_health_score(
        ingredient
    )


    category = get_health_category(
        score
    )



    # Insert only new recipe

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

        """,

        (
            recipe_name,
            ingredient,
            score,
            category
        )

    )


    conn.commit()


    cursor.close()

    conn.close()



    return {

        "health_score": score,

        "health_category": category

    }