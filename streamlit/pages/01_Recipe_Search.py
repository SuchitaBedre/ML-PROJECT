import streamlit as st

from api import search_recipe
from components.sidebar import sidebar
from styles import PAGE_STYLE


st.set_page_config(
    page_title="AI Recipe Search",
    page_icon="🔍",
    layout="wide"
)


st.markdown(
    PAGE_STYLE,
    unsafe_allow_html=True
)


sidebar()



# -----------------------------
# Header
# -----------------------------

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


🔍 AI Recipe Search Engine
</h1>

<p>
Search recipes using AI powered NLP.
Get ingredients, cooking steps and recipe details instantly.
</p>

</div>
""",
unsafe_allow_html=True
)



st.write("")



# -----------------------------
# Search Box
# -----------------------------


recipe_name = st.text_input(

    "🍽 Enter Recipe Name",

    placeholder="Example: chicken biryani"

)



search_btn = st.button(
    "🔎 Search Recipe"
)



# -----------------------------
# API CALL
# -----------------------------


if search_btn:


    if recipe_name.strip()=="":


        st.warning(
            "Please enter recipe name"
        )


    else:


        with st.spinner(
            "🤖 AI searching recipe..."
        ):


            data = search_recipe(
                recipe_name
            )



        if data and data.get("status")=="success":



            st.success(
                "Recipe Found Successfully 🎉"
            )



            st.divider()



            # Recipe Title


            st.markdown(

            f"""

            <div class="recipe-card">


            <h1 class="recipe-title">

            🍽 {data['recipe_name']}

            </h1>


            <p>

            {data.get('description','')}

            </p>


            </div>

            """,

            unsafe_allow_html=True

            )



            st.write("")



            # -----------------------------
            # Metrics
            # -----------------------------


            col1,col2,col3=st.columns(3)



            with col1:


                st.metric(

                    "⏱ Cooking Time",

                    f"{data.get('cooking_time_minutes','N/A')} min"

                )



            with col2:

                predicted = data.get("predicted_rating")

                st.metric(
                    "🤖 Predicted Rating",
                    f"{predicted:.2f}/5"
                if predicted is not None
                else "Not Available"
                )



            with col3:


                st.metric(

                    "🍴 Type",

                    "Recipe"

                )




            st.divider()


            import ast

            # -----------------------------
            # Ingredients
            # -----------------------------

            import ast

            st.subheader("🥘 Ingredients")

            ingredients = data.get("ingredients", [])

            if isinstance(ingredients, str):
                try:
                    ingredients = ast.literal_eval(ingredients)
                except:
                    ingredients = [ingredients]

            col1, col2 = st.columns(2)

            for i, ingredient in enumerate(ingredients):

                if i % 2 == 0:
                    with col1:
                        st.success(f"✅ {ingredient}")

                else:
                    with col2:
                        st.success(f"✅ {ingredient}")

            st.divider()

            # -----------------------------
            # Cooking Steps
            # -----------------------------

            st.subheader("👨‍🍳 Cooking Instructions")

            steps = data.get("cooking_steps", [])

            if isinstance(steps, str):
                try:
                    steps = ast.literal_eval(steps)
                except:
                    steps = [steps]

            for i, step in enumerate(steps, start=1):

                st.markdown(
                    f"""
<div style="
background:white;
padding:18px;
margin-bottom:15px;
border-radius:15px;
border-left:6px solid #ff6b35;
box-shadow:0px 4px 15px rgba(0,0,0,0.1);
">

<h4 style="color:#ff6b35;">
🍴 Step {i}
</h4>

<p style="font-size:17px;">
{step.capitalize()}
</p>

</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.error("Recipe not found")