# =====================================================
# embeddings.py
# =====================================================

import os

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load .env
load_dotenv()


def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )