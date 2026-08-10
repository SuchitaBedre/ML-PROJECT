import streamlit as st
from streamlit_searchbox import st_searchbox

from api import (
    analyze_review,
    search_recipe_names
)

from components.sidebar import sidebar

from styles import PAGE_STYLE



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title=" Recipe Review Analyzer",

    page_icon="🍲",

    layout="wide"

)



# =====================================================
# CSS

# =====================================================

st.markdown(

    PAGE_STYLE,

    unsafe_allow_html=True

)



# =====================================================
# SIDEBAR

# =====================================================

sidebar()



# =====================================================
# HEADER

# =====================================================

st.markdown(

"""
<div style="
background:linear-gradient(135deg,#ff6b35,#ff914d);
padding:40px;
border-radius:25px;
color:white;
">

<h1 style="
color:white;
font-size:42px;
font-weight:800;
">

 🍲 Recipe Review Analyzer
</h1>
<p style="
font-size:20px;
">
Search recipes, reviews and analyze sentiment of reviews
</p>


</div>

""",

unsafe_allow_html=True

)



st.write("")


# =====================================================
# Search Function
# =====================================================

def search_recipes(searchterm: str):

    if not searchterm:
        return []

    return search_recipe_names(searchterm)


# =====================================================
# Recipe Search
# =====================================================

selected_recipe = st_searchbox(

    search_function=search_recipes,

    placeholder="Type recipe name...",

    label="Recipe Name",

    key="recipe_search"

)


# If recipe not selected, allow manual input
manual_recipe = st.text_input(

    "Or enter a new recipe name",

    placeholder="Example: My Special Pasta"

)


# Decide final recipe
recipe_name = selected_recipe if selected_recipe else manual_recipe


# =====================================================
# Review
# =====================================================

review = st.text_area(

    "Enter Your Review"

)


# =====================================================
# Analyze Button
# =====================================================

if st.button("Analyze Review"):

    if recipe_name.strip() == "":

        st.warning("Please enter a recipe name.")

        st.stop()

    if review.strip() == "":

        st.warning("Please enter your review.")

        st.stop()

    response = analyze_review(

        recipe_name,

        review

    )

    if response.get("status") == "error":

        st.error(response["message"])

    else:

        st.success("Review analyzed successfully")

        st.subheader("🍲 Recipe Details")

        st.write("**Recipe Name:**", response["recipe_name"])

        st.write("**Review:**", response["review"])

        sentiment = response["sentiment"]

        confidence = response["confidence_score"]

        if sentiment.lower() == "positive":

            st.success(f"😊 Sentiment : {sentiment.upper()}")

        elif sentiment.lower() == "negative":

            st.error(f"😞 Sentiment : {sentiment.upper()}")

        else:

            st.warning(f"😐 Sentiment : {sentiment.upper()}")

        st.metric(

            "Confidence Score",

            f"{confidence*100:.2f}%"

        )

        if response["database_saved"]:

            st.success("✅ Review saved into PostgreSQL")

        else:

            st.error("❌ Database save failed")