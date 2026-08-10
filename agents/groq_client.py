from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"


def generate_answer(prompt: str) -> str:
    try:
        print(f"Using Groq Model: {MODEL}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert recipe assistant. Answer only using the provided recipe context. If the answer is not in the context, say so."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        print("Groq Error:", e)

        return (
            "I found suitable recipes from the recipe database.\n\n"
            "However, the AI service is temporarily unavailable.\n"
            "Please try again later."
        )