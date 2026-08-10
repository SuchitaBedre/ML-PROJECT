from fastapi import APIRouter

from Backend.services.recipe_service import search_recipes

router = APIRouter(
    prefix="/recipes",
    tags=["Recipes"]
)


@router.get("/search")
def recipe_search(q: str):

    return {

        "recipes": search_recipes(q)

    }