# ============================================================
# sentiment_service.py
# AI Powered Recipe Recommendation and Rating Prediction
# Real-Time Sentiment Analysis Service
# Uses Pre-trained Hugging Face Model
# ============================================================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Pre-trained sentiment model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = None
model = None


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

def load_model():

    global tokenizer, model


    if model is None:

        print("Loading sentiment model...")


        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )


        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )


        model.to(DEVICE)

        model.eval()


        print(
            f"Sentiment model loaded on {DEVICE}"
        )


    return tokenizer, model



# ============================================================
# PREDICT SENTIMENT
# ============================================================

def predict_sentiment(text):


    tokenizer, model = load_model()


    # Tokenization

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )


    # Move tensors to CPU/GPU

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    # Prediction

    with torch.no_grad():

        output = model(**inputs)


        probabilities = torch.softmax(
            output.logits,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    # Model labels
    # CardiffNLP RoBERTa:
    # 0 = negative
    # 1 = neutral
    # 2 = positive

    labels = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }


    result = {

        "text": text,

        "sentiment": labels[
            int(prediction.item())
        ],

        "confidence": round(
            float(confidence.item()),
            4
        )

    }


    return result