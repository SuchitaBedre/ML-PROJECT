# =====================================================
# 12_sentiment_model.py
# AI Powered Recipe Recommendation and Rating Prediction
# RoBERTa Sentiment Model Fine-Tuning
# =====================================================

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

warnings.filterwarnings("ignore")

# =====================================================
# PYTHON CONFIGURATION
# =====================================================

PYTHON_PATH = sys.executable

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("=" * 60)

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

OUTPUT_DIR = os.path.join(
    MODEL_DIR,
    "robustbert_sentiment_model"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =====================================================
# INPUT DATASET
# =====================================================

INPUT_FILE = os.path.join(
    DATA_DIR,
    "recipe_sentiment_features.parquet"
)

# =====================================================
# MODEL CONFIGURATION
# =====================================================

BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

MAX_LENGTH = 256

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

NUM_LABELS = 3

# =====================================================
# DATASET COLUMNS
# =====================================================

TEXT_COLUMN = "sentiment_text"

LABEL_COLUMN = "sentiment_label"

print("\nDataset Path")
print(INPUT_FILE)

# =====================================================
# LOAD DATASET
# =====================================================

def load_data():

    print("\n" + "=" * 60)
    print("Loading Sentiment Dataset...")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_FILE}\n"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nTotal Records : {len(df):,}")
    print(f"Total Columns : {len(df.columns)}")

    print("\nColumns")

    for col in df.columns:
        print(f"✓ {col}")

    return df


# =====================================================
# PREPARE DATASET
# =====================================================

def prepare_dataset(df):

    print("\nChecking required columns...")

    required_columns = [
        TEXT_COLUMN,
        LABEL_COLUMN
    ]

    missing_columns = []

    for col in required_columns:

        if col not in df.columns:
            missing_columns.append(col)

    if len(missing_columns) > 0:

        raise Exception(
            f"\nMissing Columns : {missing_columns}"
        )

    print("All required columns found.")

    df = df[required_columns].copy()

    before = len(df)

    df = df.dropna()

    after = len(df)

    print(f"\nRemoved Missing Records : {before-after:,}")

    df = df.rename(
        columns={
            TEXT_COLUMN: "text",
            LABEL_COLUMN: "label"
        }
    )

    df["text"] = df["text"].astype(str)

    df["label"] = df["label"].astype(int)

    print("\nDataset Preview")

    print(df.head())

    print("\nSentiment Distribution")

    print(df["label"].value_counts().sort_index())

    print("\nClass Percentages")

    print(
        (
            df["label"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    unique_labels = sorted(df["label"].unique())

    print("\nUnique Labels :", unique_labels)

    if len(unique_labels) != NUM_LABELS:

        raise Exception(
            f"\nExpected {NUM_LABELS} classes "
            f"but found {len(unique_labels)}"
        )

    return df


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

def create_train_test(df):

    print("\n" + "=" * 60)
    print("Creating Train/Test Split")
    print("=" * 60)

    train_df, test_df = train_test_split(

        df,

        test_size=0.20,

        random_state=SEED,

        shuffle=True,

        stratify=df["label"]

    )

    print(f"\nTraining Samples : {len(train_df):,}")

    print(f"Testing Samples  : {len(test_df):,}")

    train_dataset = Dataset.from_pandas(
        train_df,
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False
    )

    return train_dataset, test_dataset


    # =====================================================
# LOAD TOKENIZER
# =====================================================

def load_tokenizer():

    print("\n" + "=" * 60)
    print("Loading RoBERTa Tokenizer...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    return tokenizer


# =====================================================
# TOKENIZATION FUNCTION
# =====================================================

def tokenize_function(batch, tokenizer):

    return tokenizer(

        batch["text"],

        padding="max_length",

        truncation=True,

        max_length=MAX_LENGTH

    )


# =====================================================
# TOKENIZE DATASETS
# =====================================================

def tokenize_datasets(

    train_dataset,

    test_dataset,

    tokenizer

):

    print("\nTokenizing Training Dataset...")

    train_dataset = train_dataset.map(

        lambda batch: tokenize_function(
            batch,
            tokenizer
        ),

        batched=True

    )

    print("\nTokenizing Testing Dataset...")

    test_dataset = test_dataset.map(

        lambda batch: tokenize_function(
            batch,
            tokenizer
        ),

        batched=True

    )

    train_dataset = train_dataset.rename_column(
        "label",
        "labels"
    )

    test_dataset = test_dataset.rename_column(
        "label",
        "labels"
    )

    columns = [

        "input_ids",

        "attention_mask",

        "labels"

    ]

    train_dataset.set_format(

        type="torch",

        columns=columns

    )

    test_dataset.set_format(

        type="torch",

        columns=columns

    )

    return train_dataset, test_dataset


# =====================================================
# COMPUTE METRICS
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(

        labels,

        predictions,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        labels,

        predictions,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        labels,

        predictions,

        average="weighted",

        zero_division=0

    )

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)

    print(

        classification_report(

            labels,

            predictions,

            digits=4

        )

    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "confusion_matrix.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1

    }

    # =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    print("\n" + "=" * 60)
    print("Loading RoBERTa Model...")
    print("=" * 60)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS
    )

    model.to(DEVICE)

    print(f"Model Loaded on : {DEVICE}")

    return model


# =====================================================
# TRAINING ARGUMENTS
# =====================================================

def get_training_arguments():

    training_args = TrainingArguments(

        output_dir=CHECKPOINT_DIR,

        overwrite_output_dir=False,

        num_train_epochs=EPOCHS,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        evaluation_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="epoch",

        save_total_limit=3,

        load_best_model_at_end=True,

        metric_for_best_model="eval_f1_score",

        greater_is_better=True,

        fp16=torch.cuda.is_available(),

        logging_dir=LOG_DIR,

        report_to="none",

        seed=SEED

    )

    return training_args

# =====================================================
# CREATE TRAINER
# =====================================================

def create_trainer(

    model,

    tokenizer,

    train_dataset,

    test_dataset

):

    print("\n" + "=" * 60)
    print("Creating Trainer...")
    print("=" * 60)

    trainer = Trainer(

        model=model,

        args=get_training_arguments(),

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        tokenizer=tokenizer,

        compute_metrics=compute_metrics,

        callbacks=[

            EarlyStoppingCallback(

                early_stopping_patience=2

            )

        ]

    )

    return trainer


# =====================================================
# RESUME CHECKPOINT
# =====================================================

def find_latest_checkpoint():

    if not os.path.exists(CHECKPOINT_DIR):
        return None

    checkpoints = []

    for folder in os.listdir(CHECKPOINT_DIR):

        path = os.path.join(CHECKPOINT_DIR, folder)

        if folder.startswith("checkpoint-") and os.path.isdir(path):
            checkpoints.append(path)

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda x: int(x.split("-")[-1])
    )

    return checkpoints[-1]

    # =====================================================
# TRAIN MODEL
# =====================================================

def train_model():

    # ----------------------------
    # Load Dataset
    # ----------------------------
    df = load_data()

    df = prepare_dataset(df)

    train_dataset, test_dataset = create_train_test(df)

    # ----------------------------
    # Load Tokenizer
    # ----------------------------
    tokenizer = load_tokenizer()

    train_dataset, test_dataset = tokenize_datasets(
        train_dataset,
        test_dataset,
        tokenizer
    )

    # ----------------------------
    # Load Model
    # ----------------------------
    model = load_model()

    # ----------------------------
    # Create Trainer
    # ----------------------------
    trainer = create_trainer(
        model,
        tokenizer,
        train_dataset,
        test_dataset
    )

    # ----------------------------
    # Resume Training
    # ----------------------------
    checkpoint = find_latest_checkpoint()

    if checkpoint is not None:

        print("\n" + "=" * 60)
        print("Resuming from checkpoint")
        print(checkpoint)
        print("=" * 60)

        trainer.train(resume_from_checkpoint=checkpoint)

    else:

        print("\n" + "=" * 60)
        print("Starting Fresh Training")
        print("=" * 60)

        trainer.train()

    # ----------------------------
    # Final Evaluation
    # ----------------------------
    print("\nEvaluating Model...")

    metrics = trainer.evaluate()

    print("\nEvaluation Metrics")

    for key, value in metrics.items():
        print(f"{key} : {value}")

    # ----------------------------
    # Save Metrics
    # ----------------------------
    metrics_df = pd.DataFrame([metrics])

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("\nMetrics Saved")

    print(metrics_path)

    # ----------------------------
    # Save Model
    # ----------------------------
    print("\nSaving Model...")

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Model Saved Successfully")

    print(OUTPUT_DIR)

    return trainer


    # =====================================================
# SAVE TRAINING HISTORY
# =====================================================

def save_training_history(trainer):

    print("\nSaving Training History...")

    history = trainer.state.log_history

    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("Training history saved:")
    print(history_path)

    return history_df


# =====================================================
# PLOT TRAINING HISTORY
# =====================================================

def plot_training_history(history_df):

    if "loss" not in history_df.columns:
        print("Training loss not found.")
        return

    # --------------------------
    # Training Loss
    # --------------------------

    train_df = history_df.dropna(subset=["loss"])

    if len(train_df) > 0:

        plt.figure(figsize=(8,5))

        plt.plot(
            train_df["step"],
            train_df["loss"],
            marker="o"
        )

        plt.title("Training Loss")

        plt.xlabel("Steps")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.savefig(
            os.path.join(
                PLOT_DIR,
                "training_loss.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------
    # Validation Loss
    # --------------------------

    if "eval_loss" in history_df.columns:

        eval_df = history_df.dropna(
            subset=["eval_loss"]
        )

        if len(eval_df) > 0:

            plt.figure(figsize=(8,5))

            plt.plot(
                eval_df["epoch"],
                eval_df["eval_loss"],
                marker="o"
            )

            plt.title("Validation Loss")

            plt.xlabel("Epoch")

            plt.ylabel("Loss")

            plt.grid(True)

            plt.savefig(
                os.path.join(
                    PLOT_DIR,
                    "validation_loss.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    # --------------------------
    # Validation Accuracy
    # --------------------------

    if "eval_accuracy" in history_df.columns:

        eval_df = history_df.dropna(
            subset=["eval_accuracy"]
        )

        if len(eval_df) > 0:

            plt.figure(figsize=(8,5))

            plt.plot(
                eval_df["epoch"],
                eval_df["eval_accuracy"],
                marker="o"
            )

            plt.title("Validation Accuracy")

            plt.xlabel("Epoch")

            plt.ylabel("Accuracy")

            plt.grid(True)

            plt.savefig(
                os.path.join(
                    PLOT_DIR,
                    "validation_accuracy.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    print("Training graphs saved.")


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 70)
    print("AI Powered Recipe Recommendation")
    print("RoBERTa Sentiment Model Training")
    print("=" * 70)

    trainer = train_model()

    history_df = save_training_history(trainer)

    plot_training_history(history_df)

    print("\n" + "=" * 70)
    print("Training Completed Successfully")
    print("=" * 70)

    print("\nModel Directory")
    print(OUTPUT_DIR)

    print("\nGenerated Files")

    print("model.safetensors")
    print("config.json")
    print("tokenizer.json")
    print("tokenizer_config.json")
    print("special_tokens_map.json")
    print("evaluation_metrics.csv")
    print("training_history.csv")
    print("confusion_matrix.png")
    print("training_loss.png")
    print("validation_loss.png")
    print("validation_accuracy.png")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()

