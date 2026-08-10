# =====================================================
# health_score.py
# AI Powered Recipe Recommendation and Rating Prediction
# Health Score API (Rule Based)
# =====================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Backend.services.health_score_service import (
    get_or_create_health_score
)


# =====================================================
# ROUTER
# =====================================================

router = APIRouter(
    prefix="/health-score",
    tags=["Health Score"]
)


# =====================================================
# REQUEST MODEL
# =====================================================

class HealthScoreRequest(BaseModel):

    recipe_name: str

    ingredient: str



# =====================================================
# HEALTH SCORE API
# =====================================================

@router.post("/")
def calculate_health_score(
    request: HealthScoreRequest
):

    try:

        result = get_or_create_health_score(

            recipe_name=request.recipe_name,

            ingredient=request.ingredient

        )


        return {

            "status": "success",

            "recipe_name": request.recipe_name,

            "ingredient": request.ingredient,

            "health_score": result["health_score"],

            "health_category": result["health_category"]

        }


    except Exception as e:


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )