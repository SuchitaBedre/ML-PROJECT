from agents.chroma_search import search_recipes
#from agents.gemini_client import generate_answer
from agents.prompt import create_prompt
from agents.groq_client import generate_answer


def ask_recipe_assistant(question):

    print("\n==============================")
    print("AI Recipe Assistant")
    print("==============================")

    print("Question:", question)


    # ==============================
    # Greeting Handling
    # ==============================

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "good morning",
        "good evening"
    ]


    if question.lower().strip() in greetings:

        return """
Hello 👋

I am your AI Recipe Assistant.

I can help you with:

🍽 Recipe suggestions
🥗 Nutrition information
⭐ Rating prediction
🔍 Similar recipes
👨‍🍳 Cooking instructions
🥘 Ingredients information

Ask me anything about recipes!
"""


    # ==============================
    # ChromaDB Search
    # ==============================

    recipes = search_recipes(
        question,
        top_k=3
    )


    print(
        f"Recipes Found : {len(recipes)}"
    )


    if len(recipes)==0:

        return (
            "Sorry, I couldn't find "
            "any matching recipes."
        )


    # ==============================
    # Create Prompt
    # ==============================

    prompt = create_prompt(
        question,
        recipes
    )


    print(
        "Prompt Created"
    )


    # ==============================
    # Gemini
    # ==============================

    answer = generate_answer(
        prompt
    )


    print(
        "Gemini Response Generated"
    )


    return answer