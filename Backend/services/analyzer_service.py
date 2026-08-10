# Backend/services/analyzer_service.py


from Backend.services.recipe_service import find_recipe

from Backend.services.review_service import save_review

from Backend.services.sentiment_service import predict_sentiment



def analyze_recipe_review(

        recipe_name,

        ingredients,

        review

):


    # ----------------------------------
    # Step 1:
    # Check recipe in main dataset
    # ----------------------------------

    recipe = find_recipe(recipe_name)


    recipe_id = None



    if recipe:

        print(
            "Recipe found in dataset"
        )

        recipe_id = recipe["recipe_id"]


    else:

        print(
            "New recipe"
        )



    # ----------------------------------
    # Step 2:
    # ALWAYS run RobustBERT
    # ----------------------------------

    sentiment_result = predict_sentiment(

        review

    )



    # ----------------------------------
    # Step 3:
    # Save every review
    # ----------------------------------

    save_review(

        recipe_id,

        recipe_name,

        ingredients,

        review,

        sentiment_result["sentiment"],

        sentiment_result["confidence"]

    )



    return {


        "recipe_name":recipe_name,


        "review":review,


        "sentiment":
            sentiment_result["sentiment"],


        "confidence":
            sentiment_result["confidence"],


        "source":
            "RobustBERT"


    }