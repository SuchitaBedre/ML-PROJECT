def create_prompt(
        question,
        recipes
):


    context=""


    for r in recipes:

        context += "\n" + r



    prompt=f"""

You are an AI Recipe Assistant.

Answer user questions using only
the recipe information below.

Recipe Information:

{context}


User Question:

{question}


Give a clear answer with:

- Recipe name
- Ingredients
- Cooking time
- Rating
- Short explanation

"""


    return prompt