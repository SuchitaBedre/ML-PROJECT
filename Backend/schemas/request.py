# ==========================================================
# REQUEST SCHEMAS
# ==========================================================

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    name: str
    ingredients: str
    tags: str
    description: str


class RecommendationRequest(BaseModel):
    recipe_name: str
    top_n: int = 5


class SearchRequest(BaseModel):
    keyword: str


class RatingRequest(BaseModel):
    recipe_name: str