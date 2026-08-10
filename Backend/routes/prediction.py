# =====================================================
# IMPORT LIBRARIES
# =====================================================

from fastapi import APIRouter, HTTPException

from Backend.services.prediction_service import predict_rating


# =====================================================
# CREATE ROUTER
# =====================================================

router = APIRouter()


# =====================================================
# PREDICT RECIPE RATING
# =====================================================

@router.get("/{recipe_name}")
def prediction(recipe_name: str):

    try:

        result = predict_rating(recipe_name)

        return {

            "status": "success",

            "recipe_id": result["recipe_id"],

            "recipe_name": result["recipe_name"],

            "predicted_rating": result["predicted_rating"]

        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )