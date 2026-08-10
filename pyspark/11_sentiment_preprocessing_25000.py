# =====================================================
# 11_sentiment_preprocessing.py
# AI Powered Recipe Recommendation and Rating Prediction
# RoBERTa Sentiment Feature Generation
# Version with Checkpoint + Resume Support
# =====================================================

import os
import re
import time
import pickle
import torch
import pandas as pd

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

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
    "data"
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

CHECKPOINT_DIR = os.path.join(
    DATA_DIR,
    "checkpoints"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================
# OUTPUT FILES
# =====================================================

OUTPUT_CSV = os.path.join(
    PROCESSED_DIR,
    "recipe_sentiment_features.csv"
)

OUTPUT_PARQUET = os.path.join(
    PROCESSED_DIR,
    "recipe_sentiment_features.parquet"
)

CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "sentiment_checkpoint.pkl"
)

PARTIAL_FILE = os.path.join(
    PROCESSED_DIR,
    "recipe_sentiment_partial.parquet"
)

# =====================================================
# MODEL CONFIG
# =====================================================

MODEL_NAME = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_LENGTH = 256

BATCH_SIZE = (
    128
    if DEVICE == "cuda"
    else 32
)

# Number of reviews to process
SAMPLE_SIZE = 25000

# Save every N batches
SAVE_EVERY = 20

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# =====================================================
# CHECKPOINT FUNCTIONS
# =====================================================

def save_checkpoint(
        batch_index,
        labels,
        scores
):

    checkpoint = {

        "batch_index": batch_index,

        "data_size": len(labels),

        "labels": labels,

        "scores": scores

    }

    with open(
            CHECKPOINT_FILE,
            "wb"
    ) as f:

        pickle.dump(
            checkpoint,
            f
        )

def load_checkpoint():

    if not os.path.exists(
            CHECKPOINT_FILE
    ):

        return 0, [], []

    with open(
            CHECKPOINT_FILE,
            "rb"
    ) as f:

        checkpoint = pickle.load(
            f
        )

    print("\nCheckpoint Found")

    print(
        f"Resuming from batch {checkpoint['batch_index']}"
    )

    print(
    f"Processed Reviews : {checkpoint['data_size']}"
)

    return (
        checkpoint["batch_index"],
        checkpoint["labels"],
        checkpoint["scores"]
    )

# =====================================================
# FIND DATASET
# =====================================================

TEXT_COLUMNS = [

    "review",

    "reviews",

    "description",

    "comments",

    "text"

    

]



# =====================================================
# FIND DATASET
# =====================================================

def find_dataset():

    dataset_path = os.path.join(
        PROCESSED_DIR,
        "final_recipe_dataset.parquet"
    )

    if not os.path.exists(dataset_path):

        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    print("\nLoading Dataset...\n")

    print(dataset_path)

    return dataset_path
# =====================================================
# LOAD DATA
# =====================================================

# =====================================================
# LOAD DATASET
# =====================================================

def load_data(path):

    print("\nLoading Dataset...")

    if path.endswith(".csv"):

        df = pd.read_csv(path)

    else:

        df = pd.read_parquet(path)

    print(f"Original Records : {len(df):,}")

    return df
# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    if isinstance(text, list):
        text = " ".join(map(str, text))

    text = str(text)

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9\s.,!?]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =====================================================
# FIND TEXT COLUMN
# =====================================================

def get_text_column(df):

    for col in df.columns:

        if col.lower() in TEXT_COLUMNS:

            return col

    raise Exception(
        "No review/text column found."
    )



# =====================================================
# LOAD TOKENIZER + MODEL
# =====================================================

def load_model():

    print("\nLoading Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(

        MODEL_NAME

    )

    print("Loading RoBERTa Model...")

    model = AutoModelForSequenceClassification.from_pretrained(

        MODEL_NAME

    )

    model.to(

        DEVICE

    )

    model.eval()

    print(

        f"Running on : {DEVICE}"

    )

    print(

        f"Batch Size : {BATCH_SIZE}"

    )

    return tokenizer, model


# =====================================================
# INITIALIZE PROCESS
# =====================================================

def prepare():

    path = find_dataset()

    df = load_data(path)

    print(f"\nOriginal Records : {len(df):,}")

    # Find review column
    text_column = get_text_column(df)

    # Clean review text
    df["sentiment_text"] = (
        df[text_column]
        .fillna("")
        .apply(clean_text)
    )

    # Remove empty reviews
    df = df[
        df["sentiment_text"].str.len() > 0
    ].reset_index(drop=True)

    print(f"Available Reviews : {len(df):,}")

    # Take only 25,000 reviews
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:

        print(f"\nSampling {SAMPLE_SIZE:,} Reviews...")

        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=42
        ).reset_index(drop=True)

    print(f"Final Reviews : {len(df):,}")

    # Save sampled dataset
    sample_file = os.path.join(
        PROCESSED_DIR,
        "sampled_sentiment_dataset.parquet"
    )

    df.to_parquet(
        sample_file,
        index=False
    )

    print(f"Sample Saved : {sample_file}")

    # Load RoBERTa
    tokenizer, model = load_model()

    # Load checkpoint if available
    start_batch, labels, scores = load_checkpoint()

    return (
        df,
        tokenizer,
        model,
        start_batch,
        labels,
        scores
    )

    # =====================================================
# GENERATE SENTIMENT
# =====================================================

def generate_sentiment(
        df,
        tokenizer,
        model,
        start_batch,
        labels,
        scores
):

    total_batches = (

        len(df) + BATCH_SIZE - 1

    ) // BATCH_SIZE

    print("\nStarting Prediction...\n")

    print(f"Total Reviews : {len(df):,}")
    print(f"Total Batches : {total_batches}")

    start_time = time.time()

    progress = tqdm(

        range(start_batch, total_batches),

        desc="Generating Sentiment",

        unit="batch"

    )

    for batch_idx in progress:

        start = batch_idx * BATCH_SIZE

        end = min(

            start + BATCH_SIZE,

            len(df)

        )

        batch = (

            df["sentiment_text"]

            .iloc[start:end]

            .tolist()

        )

        tokens = tokenizer(

            batch,

            padding=True,

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"

        )

        tokens = {

            k: v.to(DEVICE)

            for k, v in tokens.items()

        }

        with torch.no_grad():

            output = model(**tokens)

            probs = torch.softmax(

                output.logits,

                dim=1

            )

        confidence, prediction = torch.max(

            probs,

            dim=1

        )

        labels.extend(

            LABEL_MAP[int(x)]

            for x in prediction.cpu().numpy()

        )

        scores.extend(

            confidence.cpu().numpy().tolist()

        )

        elapsed = (

            time.time() - start_time

        )

        processed = batch_idx + 1

        avg_time = elapsed / processed

        remaining = (

            total_batches - processed

        ) * avg_time

        progress.set_postfix(

            ETA=f"{remaining/60:.1f} min"

        )

        # ============================================
        # SAVE EVERY N BATCHES
        # ============================================

        if (

            (batch_idx + 1) % SAVE_EVERY == 0

            or

            batch_idx == total_batches - 1

        ):

            save_checkpoint(

                batch_idx + 1,

                labels,

                scores

            )

            temp = df.iloc[

                :len(labels)

            ].copy()

            temp["sentiment_label"] = labels

            temp["sentiment_score"] = scores

            temp.to_parquet(

                PARTIAL_FILE,

                index=False

            )

            print(

                f"\nCheckpoint Saved -> Batch {batch_idx+1}/{total_batches}"

            )

    return labels, scores


    # =====================================================
# SAVE FINAL OUTPUT
# =====================================================

def save_results(
        df,
        labels,
        scores
):

    print("\nPreparing Final Dataset...")

    # Safety check
    if len(labels) != len(df):
        raise Exception(
            f"Label count ({len(labels)}) "
            f"does not match dataframe size ({len(df)})"
        )

    df = df.copy()

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    print("\nSaving CSV...")

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("Saved :", OUTPUT_CSV)

    print("\nSaving Parquet...")

    df.to_parquet(
        OUTPUT_PARQUET,
        index=False
    )

    print("Saved :", OUTPUT_PARQUET)

    # Remove checkpoint after successful completion
    if os.path.exists(CHECKPOINT_FILE):

        os.remove(
            CHECKPOINT_FILE
        )

        print(
            "\nCheckpoint Removed."
        )

    # Remove partial parquet
    if os.path.exists(PARTIAL_FILE):

        os.remove(
            PARTIAL_FILE
        )

        print(
            "Partial File Removed."
        )

    return df


# =====================================================
# SUMMARY
# =====================================================

def print_summary(df):

    print("\n" + "=" * 60)

    print("SENTIMENT GENERATION COMPLETED")

    print("=" * 60)

    print(
        f"Total Reviews : {len(df):,}"
    )

    print(
        "\nSentiment Distribution:\n"
    )

    print(

        df["sentiment_label"]

        .value_counts()

    )

    print("\nAverage Confidence")

    print(

        round(

            df["sentiment_score"]

            .mean(),

            4

        )

    )

    print("\nFiles Generated")

    print("-" * 60)

    print(OUTPUT_CSV)

    print(OUTPUT_PARQUET)

    print("=" * 60)


    # =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    overall_start = time.time()

    try:

        print("=" * 70)
        print("AI Powered Recipe Recommendation and Rating Prediction")
        print("RoBERTa Sentiment Feature Generation")
        print("=" * 70)

        (
            df,
            tokenizer,
            model,
            start_batch,
            labels,
            scores
        ) = prepare()

        print("\nDataset Ready")
        print(f"Reviews Selected : {len(df):,}")

        labels, scores = generate_sentiment(
            df=df,
            tokenizer=tokenizer,
            model=model,
            start_batch=start_batch,
            labels=labels,
            scores=scores
        )

        df = save_results(
            df,
            labels,
            scores
        )

        print_summary(df)

        total_time = time.time() - overall_start

        print("\nCompleted Successfully")
        print(f"Execution Time : {total_time/60:.2f} Minutes")

    except KeyboardInterrupt:

        print("\n")
        print("=" * 70)
        print("Process Interrupted By User")
        print("Checkpoint Saved")
        print("Run the script again to resume.")
        print("=" * 70)

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(str(e))
        print("=" * 70)


if __name__ == "__main__":
    main()