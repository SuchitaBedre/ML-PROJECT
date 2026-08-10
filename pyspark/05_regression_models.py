# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd


# =====================================================
# PYTHON CONFIGURATION
# =====================================================

PYTHON_PATH = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH



# =====================================================
# SPARK IMPORTS
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark import StorageLevel



# =====================================================
# MACHINE LEARNING IMPORTS
# =====================================================

from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor
)


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)



# =====================================================
# CREATE SPARK SESSION
# =====================================================

spark = (

    SparkSession.builder

    .master("local[*]")

    .appName(
        "Recipe Rating Regression"
    )

    .config(
        "spark.driver.memory",
        "8g"
    )

    .config(
        "spark.sql.shuffle.partitions",
        "16"
    )

    .config(
        "spark.python.worker.reuse",
        "true"
    )

    .getOrCreate()

)


spark.sparkContext.setLogLevel("ERROR")



# =====================================================
# LOAD DATASET
# =====================================================


print("="*60)
print("Loading Dataset")
print("="*60)



df = spark.read.parquet(

    "data/processed/final_recipe_dataset"

)



df.printSchema()



# =====================================================
# SELECT REQUIRED COLUMNS
# =====================================================


df = df.select(

    "recipe_id",

    "average_rating",

    "tfidf_features"

)



# =====================================================
# REMOVE NULL VALUES
# =====================================================


df = df.filter(

    col("average_rating").isNotNull()

    &

    col("tfidf_features").isNotNull()

)



# =====================================================
# CACHE DATA
# =====================================================


df.persist(
    StorageLevel.MEMORY_AND_DISK
)


total = df.count()


print(
    "Total Records:",
    total
)



# =====================================================
# CONVERT SPARK TO PANDAS
# =====================================================


print(
    "Converting Spark DataFrame to Pandas..."
)

df = (
    df
    .sample(False, 0.20, seed=42)
    .limit(50000)
)


pdf = df.toPandas()



print(
    pdf.head()
)



# =====================================================
# TF-IDF VECTOR CONVERSION
# =====================================================


X = np.vstack(

    pdf["tfidf_features"]

    .apply(
        lambda x: x.toArray()
    )

)


y = pdf["average_rating"]



print(
    "Feature Shape:",
    X.shape
)



# =====================================================
# TRAIN TEST SPLIT
# =====================================================


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



print(
    "Training Data:",
    X_train.shape
)


print(
    "Testing Data:",
    X_test.shape
)



# =====================================================
# DEFINE REGRESSION MODELS
# =====================================================


models = {


    "Decision_Tree":

        DecisionTreeRegressor(

            max_depth=8,

            random_state=42

        ),



    "Random_Forest":

        RandomForestRegressor(

            n_estimators=100,

            max_depth=10,

            random_state=42,

            n_jobs=-1

        ),

}



# =====================================================
# TRAIN MODELS
# =====================================================


results=[]


best_model = None

best_model_name = None

best_rmse = float("inf")



for name, model in models.items():


    print("\n")
    print("="*60)
    print("Training:",name)
    print("="*60)



    model.fit(

        X_train,

        y_train

    )

# =====================================================
# MODEL EVALUATION
# =====================================================

    prediction = model.predict(

        X_test

    )

    mse = mean_squared_error(

        y_test,

        prediction

    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(

        y_test,

        prediction

    )

    r2 = r2_score(

        y_test,

        prediction

    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "R2:",
        round(r2, 4)
    )


    results.append(

        {

            "Model":name,

            "RMSE":rmse,

            "MAE":mae,

            "R2":r2

        }

    )



    # BEST MODEL CHECK


    if rmse < best_rmse:


        best_rmse = rmse

        best_model = model

        best_model_name = name





# =====================================================
# SAVE BEST MODEL AS PKL
# =====================================================


os.makedirs(

    "models",

    exist_ok=True

)



joblib.dump(

    best_model,

    "models/best_regression_model.pkl"

)

# =====================================================
# SAVE BEST MODEL AS PKL
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "models/best_regression_model.pkl"
)


# =====================================================
# SAVE MODEL METADATA
# =====================================================

metadata = {

    "regression_model":
    best_model_name,

    "rmse":
    best_rmse,

    "features":
    [
        "tfidf_features"
    ]

}


joblib.dump(
    metadata,
    "models/metadata.pkl"
)


print(
    "Saved: models/metadata.pkl"
)

print("\n")

print("="*60)

print(
    "BEST MODEL:",
    best_model_name
)

print(
    "BEST RMSE:",
    best_rmse
)

print("="*60)




# =====================================================
# SAVE RESULTS
# =====================================================


results_df = pd.DataFrame(

    results

)



os.makedirs(

    "data/output",

    exist_ok=True

)



results_df.to_csv(

    "data/output/regression_results.csv",

    index=False

)



print(

"Regression Results Saved"

)



# =====================================================
# CLEANUP
# =====================================================


df.unpersist()


spark.stop()



print(

"Spark Session Closed"

)