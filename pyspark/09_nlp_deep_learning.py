from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Recipe NLP Deep Learning") \
    .getOrCreate()

df = spark.read.parquet("data/processed/final_recipe_dataset")
import pandas as pd
from pyspark.sql.functions import concat_ws, lower, regexp_replace, col
from tensorflow.keras.models import Sequential


df = df.fillna("")

df = df.withColumn(
    "text",
    concat_ws(
        " ",
        col("name"),
        col("ingredients"),
        col("description"),
        col("review"),
        col("tags")
    )
)


df = df.withColumn(
    "text",
    lower(
        regexp_replace(col("text"), "[^a-zA-Z ]", " ")
    )
)

df = df.sample(False, 0.10, seed=42)


pdf = df.select(
    "text",
    "average_rating"
).toPandas()

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(
    num_words=10000
)

tokenizer.fit_on_texts(pdf["text"])

sequences = tokenizer.texts_to_sequences(pdf["text"])

from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(
    sequences,
    maxlen=300
)
y = (
    pdf["average_rating"] >= 3
).astype(int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    GRU,
    Dense,
    Dropout,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D
)

def build_lstm():

    model = Sequential()

    model.add(
        Embedding(
            input_dim=10000,
            output_dim=128,
            input_length=300
        )
    )

    model.add(
        LSTM(128)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(64, activation="relu")
    )

    model.add(
        Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def build_bilstm():

    model = Sequential()

    model.add(
        Embedding(
            10000,
            128,
            input_length=300
        )
    )

    model.add(
        Bidirectional(
            LSTM(128)
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(64, activation="relu")
    )

    model.add(
        Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def build_gru():

    model = Sequential()

    model.add(
        Embedding(
            10000,
            128,
            input_length=300
        )
    )

    model.add(
        GRU(128)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(64, activation="relu")
    )

    model.add(
        Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def build_cnn():

    model = Sequential()

    model.add(
        Embedding(
            10000,
            128,
            input_length=300
        )
    )

    model.add(
        Conv1D(
            filters=128,
            kernel_size=5,
            activation="relu"
        )
    )

    model.add(
        MaxPooling1D(pool_size=2)
    )

    model.add(
        GlobalMaxPooling1D()
    )

    model.add(
        Dense(64, activation="relu")
    )

    model.add(
        Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

models = {

    "LSTM": build_lstm(),

    "BiLSTM": build_bilstm(),

    "GRU": build_gru(),

    "CNN": build_cnn()

}

# =====================================================
# TRAIN, EVALUATE AND COMPARE ALL MODELS
# =====================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

best_model = None
best_model_name = ""
best_f1 = -1

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Prediction
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc
    ])

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# =====================================================
# CREATE MODEL COMPARISON TABLE
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

result_df = result_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n")
print("=" * 70)
print("NLP MODEL COMPARISON")
print("=" * 70)
print(result_df)
print("=" * 70)

import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

result_df.to_csv(
    "results/nlp_model_comparison.csv",
    index=False
)


# =====================================================
# SAVE BEST MODEL
# =====================================================



best_model.save(
    "models/best_nlp_model.keras"
)


# =====================================================
# SAVE TOKENIZER
# =====================================================

import joblib

joblib.dump(
    tokenizer,
    "models/tokenizer.pkl"
)


# =====================================================
# PRINT BEST MODEL
# =====================================================

print("\n")
print("=" * 60)
print("BEST NLP MODEL")
print("=" * 60)
print("Model      :", best_model_name)
print("F1 Score   :", round(best_f1, 4))
print("Saved Model: models/best_nlp_model.keras")
print("Tokenizer  : models/tokenizer.pkl")
print("Results    : results/nlp_model_comparison.csv")
print("=" * 60)

spark.stop()