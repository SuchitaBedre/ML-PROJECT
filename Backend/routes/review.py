# =====================================================
# review.py
# Recipe Review Sentiment Analysis API
# Prediction + PostgreSQL Storage
# =====================================================

from fastapi import APIRouter
from pydantic import BaseModel

import os
import psycopg2

from dotenv import load_dotenv

from Backend.services.sentiment_service import predict_sentiment


load_dotenv()


router = APIRouter(
    prefix="/review",
    tags=["Review"]
)


# =====================================================
# PostgreSQL Connection
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
# Request Model
# =====================================================

class ReviewRequest(BaseModel):

    recipe_name: str

    review: str



# =====================================================
# Analyze Review
# =====================================================

@router.post("/")
def analyze_review(
    request: ReviewRequest
):

    recipe_name = request.recipe_name

    review = request.review


    # ---------------------------------
    # Sentiment Prediction
    # ---------------------------------

    result = predict_sentiment(review)


    sentiment = result["sentiment"]

    confidence_score = result["confidence"]



    database_saved = False



    # ---------------------------------
    # Insert into user_reviews table
    # ---------------------------------

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
            "PostgreSQL Insert Error:",
            e
        )


        database_saved = False



    # ---------------------------------
    # Response for Streamlit
    # ---------------------------------

    return {

        "status": "success",

        "recipe_name": recipe_name,

        "review": review,

        "sentiment": sentiment,

        "confidence_score": confidence_score,

        "database_saved": database_saved

    }