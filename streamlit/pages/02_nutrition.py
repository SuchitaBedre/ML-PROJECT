# =====================================================
# 02_nutrition.py
# Nutrition Analysis + Health Score
# Auto Recipe Search
# =====================================================


import streamlit as st

from streamlit_searchbox import st_searchbox


from api import (

    nutrition,

    search_recipe_names

)


from components.sidebar import sidebar

from styles import PAGE_STYLE



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="Nutrition Analysis",

    page_icon="🥗",

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

🥗 Recipe Nutrition Intelligence

</h1>


<p style="
font-size:20px;
">

Search recipes, analyze nutrition and calculate AI health score.

</p>


</div>

""",

unsafe_allow_html=True

)



st.write("")



# =====================================================
# RECIPE SEARCH

# =====================================================


def search_recipes(searchterm):


    if not searchterm:

        return []


    return search_recipe_names(

        searchterm

    )




selected_recipe = st_searchbox(

    search_function=search_recipes,

    placeholder="Type recipe name...",

    label="🍽️ Search Recipe",

    key="nutrition_recipe_search"

)



# Manual input for unseen recipes

manual_recipe = st.text_input(

    "Or enter new recipe name",

    placeholder="Example: Vegetable Pasta"

)



recipe_name = (

    selected_recipe

    if selected_recipe

    else manual_recipe

)



# =====================================================
# INGREDIENT

# =====================================================


ingredient = st.text_input(

    "🥕 Enter Ingredient",

    placeholder="Example: pasta,tomato,vegetables"

)



# =====================================================
# BUTTON

# =====================================================


if st.button("🥗 Analyze Nutrition"):



    if recipe_name.strip()=="":


        st.warning(

            "Please enter recipe name"

        )

        st.stop()



    if ingredient.strip()=="":


        st.warning(

            "Please enter ingredient"

        )

        st.stop()



    with st.spinner(

        "Analyzing recipe..."

    ):



        data = nutrition(

            recipe_name,

            ingredient

        )



    # ==========================================
    # ERROR HANDLING
    # ==========================================


    if "detail" in data:


        st.error(

            data["detail"]

        )

        st.stop()



    if data.get("status")=="error":


        st.error(

            data.get(

                "message",

                "Error"

            )

        )

        st.stop()



    # ==========================================
    # SUCCESS

    # ==========================================


    st.success(

        "✅ Analysis Completed"

    )



    st.subheader(

        "🍲 Recipe Details"

    )



    st.write(

        "**Recipe Name:**",

        data.get(

            "recipe_name",

            recipe_name

        )

    )



    if data.get("type")=="unseen":


        st.info(

            "🆕 New recipe added to health database"

        )


    else:


        st.info(

            "📚 Existing recipe"

        )



    st.divider()



    # ==========================================
    # HEALTH SCORE

    # ==========================================


    st.subheader(

        "❤️ Health Score"

    )



    col1,col2 = st.columns(2)



    with col1:


        st.metric(

            "Health Score",

            f"{data.get('health_score',0)}/10"

        )



    with col2:


        st.metric(

            "Category",

            data.get(

                "health_category",

                "Unknown"

            )

        )



    st.divider()



    # ==========================================
    # NUTRITION

    # ==========================================


    nutrition_data = data.get(

        "nutrition",

        {}

    )



    if "calories" in nutrition_data:



        st.subheader(

            "📊 Nutrition Breakdown"

        )



        col1,col2,col3,col4 = st.columns(4)



        with col1:


            st.metric(

                "🔥 Calories",

                nutrition_data["calories"]

            )



        with col2:


            st.metric(

                "💪 Protein",

                f"{nutrition_data['protein_percent']}%"

            )



        with col3:


            st.metric(

                "🥑 Fat",

                f"{nutrition_data['total_fat_percent']}%"

            )



        with col4:


            st.metric(

                "🍚 Carbs",

                f"{nutrition_data['carbohydrates_percent']}%"

            )



        st.divider()



        st.subheader(

            "Detailed Nutrition"

        )



        col1,col2 = st.columns(2)



        with col1:


            st.write(

                "💪 Protein"

            )


            st.progress(

                min(

                    nutrition_data["protein_percent"]/100,

                    1.0

                )

            )



            st.write(

                f"{nutrition_data['protein_percent']}%"

            )



            st.write(

                "🥑 Fat"

            )


            st.progress(

                min(

                    nutrition_data["total_fat_percent"]/100,

                    1.0

                )

            )



            st.write(

                f"{nutrition_data['total_fat_percent']}%"

            )



        with col2:


            st.write(

                "🍬 Sugar"

            )


            st.progress(

                min(

                    nutrition_data["sugar_percent"]/100,

                    1.0

                )

            )



            st.write(

                f"{nutrition_data['sugar_percent']}%"

            )



            st.write(

                "🧂 Sodium"

            )


            st.progress(

                min(

                    nutrition_data["sodium_percent"]/100,

                    1.0

                )

            )


            st.write(

                f"{nutrition_data['sodium_percent']}%"

            )



    else:


        st.info(

            "Nutrition values are unavailable for new recipes."

        )