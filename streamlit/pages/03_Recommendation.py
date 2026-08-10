import streamlit as st
import ast

from api import recommendation
from components.sidebar import sidebar
from styles import PAGE_STYLE



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Recipe Recommendation",
    page_icon="🍽️",
    layout="wide"
)



# =====================================================
# STYLE
# =====================================================

st.markdown(
    PAGE_STYLE,
    unsafe_allow_html=True
)



sidebar()



# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
"""
<div style="
background:linear-gradient(135deg,#ff6b35,#ff914d);
padding:40px;
border-radius:25px;
box-shadow:0px 8px 25px rgba(0,0,0,0.15);
color:white;
">

<h1 style="
color:white;
font-size:42px;
font-weight:800;
">

<h1>
    🍽 AI Recipe Recommendation Engine
</h1>


<p style="font-size:18px;color:#666">

Discover similar recipes using AI embeddings
and ingredient-based recommendation.

</p>


</div>

""",
unsafe_allow_html=True
)



st.write("")



# =====================================================
# INPUT
# =====================================================

ingredients = st.text_input(

    "🥕 Enter Ingredients",

    placeholder="Example: chicken, garlic, onion"

)



# =====================================================
# BUTTON
# =====================================================

if st.button(
    "🔍 Find Recipes"
):


    if ingredients.strip()=="":


        st.warning(
            "Please enter ingredients."
        )


    else:


        with st.spinner(
            "Finding best recipes..."
        ):


            data = recommendation(
                ingredients
            )



        if data.get("status")=="success":



            st.success(
                "✅ Recommendations Found"
            )



            recipes = data["recommendations"]



            st.subheader(
                f"🍴 Top {len(recipes)} Recommended Recipes"
            )



            st.divider()



            # =====================================================
            # RECIPE CARDS
            # =====================================================


            for i,recipe in enumerate(recipes,1):


                st.markdown(
                f"""
                <div class="recipe-card">


                <h2 class="recipe-title">

                #{i} 🍽️ {recipe['recipe_name']}

                </h2>


                </div>

                """,
                unsafe_allow_html=True
                )



                col1,col2,col3 = st.columns(3)



                with col1:

                    st.metric(
                        "⭐ Rating",
                        recipe["rating"]
                    )


                with col2:

                    st.metric(
                        "⏱ Cooking Time",
                        f"{recipe['cooking_time']} min"
                    )


                with col3:

                    st.metric(
                        "🍴 Category",
                        "Recipe"
                    )



                # ------------------------------
                # Ingredients formatting
                # ------------------------------


                ingredient_text = str(
                    recipe["ingredients"]
                )


                try:

                    ingredient_list = ast.literal_eval(
                        ingredient_text
                    )


                    if isinstance(
                        ingredient_list,
                        list
                    ):


                        ingredient_text=", ".join(

                            str(x).strip()

                            for x in ingredient_list

                        )


                except Exception:


                    ingredient_text=(

                        ingredient_text

                        .replace("[","")

                        .replace("]","")

                        .replace("'","")

                        .replace('"',"")
                    )



                st.markdown(
                    "### 🥕 Ingredients"
                )


                st.info(
                    ingredient_text
                )



                # Rating stars


                rating = int(
                    recipe["rating"]
                )


                stars = "⭐" * rating



                st.markdown(

                    f"""
                    <div class="card">

                    <h3>
                    {stars}
                    </h3>

                    User Rating:
                    {recipe['rating']}

                    </div>
                    """,

                    unsafe_allow_html=True

                )



                st.divider()



        else:


            st.error(

                data.get(
                    "message",
                    "No recommendations found."
                )

            )