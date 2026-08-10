# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


# =====================================================
# MACHINE LEARNING IMPORTS
# =====================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =====================================================
# EVALUATION METRICS
# =====================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)



# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# =====================================================
# SPARK SESSION AND DATA LOADING
# =====================================================

spark = (
    SparkSession.builder
    .appName("Recipe Classification Models")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nLoading Dataset")
df = spark.read.parquet("data/processed/final_recipe_dataset")

df = df.withColumn(
    "label",
    when(col("average_rating") < 3, 0).otherwise(1)
)

df = df.select("tfidf_features", "label")

# Take a sample
df = df.sample(False, 0.10, seed=42)

print("Sampling completed.")
print("Converting Spark DataFrame to Pandas...")
pdf = df.toPandas()
print("Conversion completed.")
print("Sample size:", len(pdf))

from scipy.sparse import csr_matrix
from pyspark.ml.linalg import SparseVector, DenseVector

if len(pdf) == 0:
    raise ValueError("Sampled dataset is empty.")

rows = []
cols = []
vals = []

for i, vec in enumerate(pdf["tfidf_features"]):

    if isinstance(vec, SparseVector):

        rows.extend([i] * len(vec.indices))
        cols.extend(vec.indices.tolist())
        vals.extend(vec.values.tolist())

    elif isinstance(vec, DenseVector):

        arr = vec.toArray()

        nz = np.nonzero(arr)[0]

        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(arr[nz].tolist())

    else:

        arr = np.asarray(vec)

        nz = np.nonzero(arr)[0]

        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(arr[nz].tolist())

num_features = pdf["tfidf_features"].iloc[0].size

X = csr_matrix(
    (vals, (rows, cols)),
    shape=(len(pdf), num_features)
)

y = pdf["label"].values

print("Feature matrix shape:", X.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])

# =====================================================
# HANDLE CLASS IMBALANCE FOR XGBOOST
# =====================================================

negative = np.sum(y_train == 0)
positive = np.sum(y_train == 1)

if positive == 0:
    scale_pos_weight = 1
else:
    scale_pos_weight = negative / positive


# =====================================================
# DEFINE MODELS
# =====================================================


models = {


    "Logistic Regression":

        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ),



    "Random Forest":

        RandomForestClassifier(

            n_estimators=200,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1

        ),



    "XGBoost":

        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )

}



# =====================================================
# STORE RESULTS
# =====================================================


results = []



# =====================================================
# TRAIN AND EVALUATE MODELS
# =====================================================


for name, model in models.items():


    print("\n")
    print("="*70)
    print(name)
    print("="*70)



    # -----------------------------
    # TRAIN MODEL
    # -----------------------------

    model.fit(

        X_train,

        y_train

    )



    # -----------------------------
    # PREDICTION
    # -----------------------------

    y_pred = model.predict(

        X_test

    )


    y_prob = model.predict_proba(

        X_test

    )[:,1]



    # -----------------------------
    # METRICS
    # -----------------------------


    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        y_pred,

        zero_division=0

    )


    roc = roc_auc_score(

        y_test,

        y_prob

    )



    # -----------------------------
    # PRINT METRICS
    # -----------------------------


    print("\nPerformance Metrics")

    print("-------------------------")

    print(
        "Accuracy :",
        round(accuracy,4)
    )


    print(
        "Precision:",
        round(precision,4)
    )


    print(
        "Recall   :",
        round(recall,4)
    )


    print(
        "F1 Score :",
        round(f1,4)
    )


    print(
        "ROC-AUC  :",
        round(roc,4)
    )




    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------


    cm = confusion_matrix(

        y_test,

        y_pred

    )


    print("\nConfusion Matrix")

    print(cm)



    # Save confusion matrix image


    plt.figure(figsize=(5,4))


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )


    plt.title(

        name + " Confusion Matrix"

    )


    plt.xlabel(

        "Predicted"

    )


    plt.ylabel(

        "Actual"

    )


    plt.tight_layout()


    plt.savefig(

        "results/"
        +
        name.lower().replace(" ","_")
        +
        "_confusion_matrix.png"

    )


    plt.close()



    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------


    print("\nClassification Report")


    print(

        classification_report(

            y_test,

            y_pred

        )

    )



    # -----------------------------
    # SAVE MODEL
    # -----------------------------


    model_filename = (

        "models/"
        +
        name.lower().replace(" ","_")
        +
        ".pkl"

    )


    joblib.dump(

        model,

        model_filename

    )


    print(

        "\nModel Saved:",

        model_filename

    )




    # -----------------------------
    # STORE RESULT
    # -----------------------------


    results.append(

        [

            name,

            accuracy,

            precision,

            recall,

            f1,

            roc

        ]

    )





# =====================================================
# CREATE RESULT DATAFRAME
# =====================================================


result_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ]

)



# =====================================================
# SORT MODELS BY F1 SCORE
# =====================================================


result_df = result_df.sort_values(

    by="F1 Score",

    ascending=False

)



print("\n")

print("="*90)

print("MODEL COMPARISON")

print("="*90)


print(result_df)


print("="*90)




# =====================================================
# SAVE RESULTS CSV
# =====================================================


result_df.to_csv(

    "results/classification_results.csv",

    index=False

)


print(

    "\nResults Saved Successfully"

)


print(

    "Location : results/classification_results.csv"

)




# =====================================================
# BEST MODEL SELECTION
# =====================================================


best_model_name = (

    result_df.iloc[0]["Model"]

)



best_model = models[best_model_name]



# Save best model


joblib.dump(

    best_model,

    "models/best_classification_model.pkl"

)



print("\n")

print("="*60)

print("BEST CLASSIFICATION MODEL")

print("="*60)



print(

    "Model      :",

    best_model_name

)


print(

    "Accuracy   :",

    round(result_df.iloc[0]["Accuracy"],4)

)


print(

    "Precision  :",

    round(result_df.iloc[0]["Precision"],4)

)


print(

    "Recall     :",

    round(result_df.iloc[0]["Recall"],4)

)


print(

    "F1 Score   :",

    round(result_df.iloc[0]["F1 Score"],4)

)


print(

    "ROC-AUC    :",

    round(result_df.iloc[0]["ROC-AUC"],4)

)


print(

    "Saved      : models/best_classification_model.pkl"

)


print("="*60)

