# =====================================================
# 15_health_score_model_training.py
# AI Powered Recipe Recommendation and Rating Prediction
# Health Score Prediction Model Training
# =====================================================


import os
import sys
import joblib
import warnings

import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.linear_model import LinearRegression


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



warnings.filterwarnings(
    "ignore"
)



# =====================================================
# PATH CONFIGURATION
# =====================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



INPUT_FILE = os.path.join(

    BASE_DIR,

    "data",

    "processed",

    "recipe_health_features.parquet"

)



MODEL_DIR = os.path.join(

    BASE_DIR,

    "models"

)



MODEL_FILE = os.path.join(

    MODEL_DIR,

    "health_score_model.pkl"

)



VECTORIZER_FILE = os.path.join(

    MODEL_DIR,

    "health_score_vectorizer.pkl"

)



METADATA_FILE = os.path.join(

    MODEL_DIR,

    "health_score_metadata.pkl"

)





# =====================================================
# CONFIGURATION
# =====================================================


MAX_FEATURES = 5000


TEST_SIZE = 0.2


RANDOM_STATE = 42





# =====================================================
# LOAD DATA
# =====================================================


def load_data():


    print(
        "\nLoading Health Dataset..."
    )



    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(

            f"""
            File not found:

            {INPUT_FILE}

            Run 14_health_score_feature_engineering.py first.
            """

        )



    df = pd.read_parquet(

        INPUT_FILE

    )



    print(

        f"Total Records : {len(df)}"

    )



    print(

        df.columns.tolist()

    )



    return df





# =====================================================
# TEXT PREPARATION
# =====================================================


def prepare_text(df):

    required_columns = [
        "name",
        "text_features",
        "health_score"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise Exception(
                f"Missing column: {column}"
            )

    df["recipe_text"] = (
        df["name"]
        .fillna("")
        .astype(str)
        + " "
        + df["text_features"]
        .fillna("")
        .astype(str)
    )

    df = df[
        [
            "recipe_text",
            "health_score"
        ]
    ]

    df = df.dropna()

    return df





# =====================================================
# TRAIN MODELS
# =====================================================


def train_models(

        X_train,

        X_test,

        y_train,

        y_test

):


    models = {


        "LinearRegression":

        LinearRegression(),



        "RandomForest":

        RandomForestRegressor(

            n_estimators=200,

            random_state=RANDOM_STATE,

            n_jobs=-1

        ),



        "GradientBoosting":

        GradientBoostingRegressor(

            random_state=RANDOM_STATE

        )


    }



    results = {}



    best_model = None

    best_score = -999



    for name,model in models.items():


        print(

            f"\nTraining {name}..."

        )



        model.fit(

            X_train,

            y_train

        )



        prediction = model.predict(

            X_test

        )



        rmse = np.sqrt(

            mean_squared_error(

                y_test,

                prediction

            )

        )



        mae = mean_absolute_error(

            y_test,

            prediction

        )



        r2 = r2_score(

            y_test,

            prediction

        )




        results[name]={


            "RMSE":

            rmse,


            "MAE":

            mae,


            "R2":

            r2


        }



        print(

            results[name]

        )




        if r2 > best_score:


            best_score = r2


            best_model = model





    return best_model,results






# =====================================================
# MAIN TRAINING
# =====================================================


def main():


    df = load_data()



    df = prepare_text(

        df

    )



    print(

        "\nDataset After Preparation"

    )


    print(

        df.head()

    )





    X_train_text, X_test_text, y_train, y_test = train_test_split(

        df["recipe_text"],

        df["health_score"],

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE

    )






    print(

        "\nCreating TF-IDF Features..."

    )



    vectorizer = TfidfVectorizer(

        max_features=MAX_FEATURES,

        stop_words="english"

    )



    X_train = vectorizer.fit_transform(

        X_train_text

    )



    X_test = vectorizer.transform(

        X_test_text

    )





    print(

        "Feature Shape:",

        X_train.shape

    )






    best_model,results=train_models(

        X_train,

        X_test,

        y_train,

        y_test

    )






    os.makedirs(

        MODEL_DIR,

        exist_ok=True

    )






    print(

        "\nSaving Model..."

    )



    joblib.dump(

        best_model,

        MODEL_FILE

    )



    joblib.dump(

        vectorizer,

        VECTORIZER_FILE

    )




    metadata={


        "best_model":

        type(best_model).__name__,


        "feature_count":

        MAX_FEATURES,


        "results":

        results


    }



    joblib.dump(

        metadata,

        METADATA_FILE

    )





    print(

        "\n================================"

    )


    print(

        "Health Score Model Training Completed"

    )


    print(

        f"Best Model : {type(best_model).__name__}"

    )


    print(

        "Saved Models:"

    )


    print(

        MODEL_FILE

    )



    print(

        "================================"

    )







if __name__ == "__main__":

    main()