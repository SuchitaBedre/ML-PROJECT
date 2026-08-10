from fastapi import APIRouter
from agents.ai_assistant import ask_recipe_assistant

router = APIRouter()

@router.post("/assistant")
def assistant(question: str):

    answer = ask_recipe_assistant(question)

    return {
        "answer": answer
    }