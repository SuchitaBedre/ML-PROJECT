from pathlib import Path
from dotenv import load_dotenv
from google import genai
import os
import time
import traceback

# =====================================================
# LOAD ENVIRONMENT
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

# Use one API key
api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not api_key:
    raise ValueError("Google Gemini API Key not found.")

client = genai.Client(api_key=api_key)


# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(prompt):

    MODEL = "gemini-2.5-flash"

    print(f"\nUsing Gemini Model : {MODEL}")
    print(f"Prompt Length : {len(prompt)} characters")

    retries = 3

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            if response.text:
                return response.text

            return "No response generated."

        except Exception as e:

            print("\n" + "=" * 70)
            print(f"Gemini Attempt {attempt + 1} Failed")
            print(type(e).__name__)
            print(e)
            print("=" * 70)

            if attempt < retries - 1:
                print("Retrying in 5 seconds...\n")
                time.sleep(5)
            else:
                traceback.print_exc()

    return (
        "⚠️ I found matching recipes in the recipe database, "
        "but Google's Gemini service is temporarily unavailable.\n\n"
        "Please try again after a few minutes."
    )