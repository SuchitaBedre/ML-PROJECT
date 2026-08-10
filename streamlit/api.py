import requests
from config import BACKEND_URL


# ==========================================
# SEARCH
# ==========================================

def search_recipe(recipe_name):

    url = f"{BACKEND_URL}/search/{recipe_name}"

    print("=" * 60)
    print("URL :", url)

    response = requests.get(url)

    print("STATUS :", response.status_code)
    print("TEXT :", response.text)

    try:
        data = response.json()
        print("JSON :", data)
        return data

    except Exception:
        return {
            "status": "error",
            "message": response.text
        }

# ==========================================
# NUTRITION
# ==========================================


def nutrition(recipe_name, ingredient):

    try:

        url = f"{BACKEND_URL}/nutrition/"

        print("CALLING:", url)

        response = requests.post(

            url,

            json={

                "recipe_name": recipe_name,

                "ingredient": ingredient

            }

        )


        print("STATUS:", response.status_code)

        print("RESPONSE:", response.text)


        return response.json()


    except Exception as e:


        return {

            "status":"error",

            "message":str(e)

        }
# ==========================================
# PREDICTION
# ==========================================

def prediction(recipe_name):

    try:

        response = requests.get(
            f"{BACKEND_URL}/prediction/{recipe_name}",
            timeout=30
        )

        return response.json()

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ==========================================
# AI ASSISTANT
# ==========================================

def ai_assistant(question):

    try:

        response = requests.post(
            f"{BACKEND_URL}/assistant",
            params={
                "question": question
            },
            timeout=180
        )

        return response.json()

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

# ==========================================
# REVIEW SENTIMENT ANALYSIS
# ==========================================

def analyze_review(recipe_name, review):

    try:

        response = requests.post(
            f"{BACKEND_URL}/review/",
            json={
                "recipe_name": recipe_name,
                "review": review
            },
            timeout=120
        )

        return response.json()


    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

# ==========================================
# RECIPE LIST
# ==========================================

def get_recipe_names():

    try:

        response = requests.get(
            f"{BACKEND_URL}/recipes/",
            timeout=30
        )

        return response.json()["recipes"]

    except Exception as e:

        print(e)
        return []

# ==========================================
# RECIPE AUTO SEARCH
# ==========================================

from requests import get


def search_recipe_names(keyword):

    try:

        response = requests.get(

            f"{BACKEND_URL}/recipes/search",

            params={
                "q": keyword
            },

            timeout=30

        )

        return response.json().get(
            "recipes",
            []
        )


    except Exception:

        return []

# ==========================================
# RECOMMENDATION
# ==========================================

def recommendation(recipe_name):

    try:

        response = requests.get(

            f"{BACKEND_URL}/recommendation/{recipe_name}",

            timeout=120

        )


        return response.json()


    except Exception as e:


        return {

            "status": "error",

            "message": str(e)

        }