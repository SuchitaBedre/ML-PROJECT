# =====================================================
# sentiment.py
# Sentiment Analysis API Route
# Prediction + Save Review into PostgreSQL
# =====================================================

from fastapi import APIRouter
from pydantic import BaseModel

import psycopg2
import os

from dotenv import load_dotenv

from Backend.services.sentiment_service import predict_sentiment


load_dotenv()


router = APIRouter()


# -----------------------------
# Database Connection
# -----------------------------

def get_connection():

    return psycopg2.connect(

        host="localhost",

        port=os.getenv("POSTGRES_PORT"),

        database=os.getenv("POSTGRES_DB"),

        user=os.getenv("POSTGRES_USER"),

        password=os.getenv("POSTGRES_PASSWORD")

    )



# -----------------------------
# Request Model
# -----------------------------

class SentimentRequest(BaseModel):

    recipe_name: str

    review: str



# -----------------------------
# Sentiment API
# -----------------------------

@router.post("/sentiment")
def analyze_sentiment(
    request: SentimentRequest
):


    recipe_name = request.recipe_name

    review = request.review


    # -------------------------
    # Predict sentiment
    # -------------------------

    result = predict_sentiment(review)


    sentiment = result["sentiment"]

    confidence_score = result["confidence"]



    database_saved = False



    # -------------------------
    # Save into user_reviews
    # -------------------------

    try:

        conn = get_connection()

        cursor = conn.cursor()


        cursor.execute(
            """
            INSERT INTO user_reviews
            (
                recipe_name,
                review,
                sentiment,
                confidence_score
            )

            VALUES
            (
                %s,
                %s,
                %s,
                %s
            )
            """,

            (
                recipe_name,
                review,
                sentiment,
                confidence_score
            )
        )


        conn.commit()


        database_saved = True


        cursor.close()

        conn.close()



    except Exception as e:

        print(
            "Database Error:",
            e
        )



    return {


        "status": "success",


        "recipe_name": recipe_name,


        "review": review,


        "sentiment": sentiment,


        "confidence_score": confidence_score,


        "database_saved": database_saved

    }