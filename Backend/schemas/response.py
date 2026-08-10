# ==========================================================
# RESPONSE SCHEMAS
# ==========================================================

from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    predicted_rating: float


class RecipeResponse(BaseModel):
    recipe_id: int
    recipe_name: str
    rating: float


class RecommendationResponse(BaseModel):
    recommendations: List[RecipeResponse]