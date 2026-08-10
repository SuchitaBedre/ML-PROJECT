# =====================================================
# recommendation.py
# =====================================================

from fastapi import APIRouter

from Backend.services.recommendation_service import recommend_by_ingredients

router = APIRouter()


@router.get("/{ingredients}")
def recommendation(ingredients: str):

    return recommend_by_ingredients(ingredients)