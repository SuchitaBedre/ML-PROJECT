# =====================================================
# 16_health_score_prediction.py
# AI Powered Recipe Recommendation and Rating Prediction
# Recipe Health Score Prediction
# =====================================================


import os
import joblib



# =====================================================
# PATH CONFIGURATION
# =====================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



MODEL_FILE = os.path.join(

    BASE_DIR,

    "models",

    "health_score_model.pkl"

)



VECTORIZER_FILE = os.path.join(

    BASE_DIR,

    "models",

    "health_score_vectorizer.pkl"

)





# =====================================================
# LOAD MODEL
# =====================================================


def load_model():


    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "Health score model not found. Run 15_health_score_model_training.py first."
        )


    model = joblib.load(
        MODEL_FILE
    )


    vectorizer = joblib.load(
        VECTORIZER_FILE
    )


    return model, vectorizer






# =====================================================
# TEXT CLEANING
# =====================================================


def clean_text(text):


    text = str(text)


    text = text.lower()


    return text






# =====================================================
# PREDICT HEALTH SCORE
# =====================================================


def predict_health_score(

        recipe_name,

        ingredients

):


    model,vectorizer = load_model()



    recipe_text = (

        recipe_name

        +

        " "

        +

        ingredients

    )



    recipe_text = clean_text(

        recipe_text

    )



    features = vectorizer.transform(

        [

            recipe_text

        ]

    )



    score = model.predict(

        features

    )[0]



    # keep score between 0-10

    score = max(

        0,

        min(

            10,

            score

        )

    )



    return round(

        float(score),

        2

    )







# =====================================================
# MAIN
# =====================================================


if __name__ == "__main__":


    print(

        "\n=============================="

    )

    print(

        "Recipe Health Score Prediction"

    )

    print(

        "=============================="

    )



    recipe_name = input(

        "\nEnter Recipe Name: "

    )



    ingredients = input(

        "\nEnter Ingredients: "

    )



    score = predict_health_score(

        recipe_name,

        ingredients

    )



    print(

        "\n------------------------------"

    )


    print(

        "Recipe:",

        recipe_name

    )


    print(

        "Predicted Health Score:",

        score,

        "/10"

    )


    print(

        "------------------------------"

    )