from datetime import datetime



def save_review(
        recipe_name,
        review,
        sentiment,
        confidence
):


    review_data = {


        "recipe_name": recipe_name,

        "review_text": review,

        "sentiment": sentiment,

        "confidence_score": confidence,


    }



    # Replace this with DB insert

    print(review_data)


    return True