# =====================================================
# Home.py
# AI Recipe Intelligence Platform
# =====================================================

import streamlit as st

from components.sidebar import sidebar
from styles import PAGE_STYLE


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Recipe Intelligence",
    page_icon="🍽️",
    layout="wide"
)


# =====================================================
# GLOBAL STYLE
# =====================================================

st.markdown(
    PAGE_STYLE + """
<style>

h1, h2, h3, h4, p {
    text-align: center;
}

div.stButton > button {
    width: 100%;
    height: 50px;
    background-color: #ff6b35;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    padding: 0;
}

div.stButton > button:hover {
    background-color: #ff914d;
    color: white;
}

</style>
""",
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

sidebar()


# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
    """
<div style="
    background: linear-gradient(135deg, #ff6b35, #ff914d);
    padding: 60px;
    border-radius: 25px;
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
">

<h1 style="
    color: white;
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    margin: 0;
">
🍽️ AI Recipe Intelligence Platform
</h1>

</div>
""",
    unsafe_allow_html=True
)


st.write("")


# =====================================================
# QUICK START
# =====================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)


# =====================================================
# RECIPE SEARCH
# =====================================================

with c1:
    if st.button(
        "🍲 Recipe Search",
        use_container_width=True
    ):
        st.switch_page(
            "pages/01_Recipe_Search.py"
        )


# =====================================================
# NUTRITION
# =====================================================

with c2:
    if st.button(
        "🥗 Nutrition",
        use_container_width=True
    ):
        st.switch_page(
            "pages/02_nutrition.py"
        )


# =====================================================
# REVIEW ANALYZER
# =====================================================

with c3:
    if st.button(
        "😊 Review Analyzer",
        use_container_width=True
    ):
        st.switch_page(
            "pages/04_Review_Analyzer.py"
        )


# =====================================================
# AI ASSISTANT
# =====================================================

with c4:
    if st.button(
        "🤖 AI Assistant",
        use_container_width=True
    ):
        st.switch_page(
            "pages/05_AI_Assistant.py"
        )


# =====================================================
# RECOMMENDATION
# =====================================================

with c5:
    if st.button(
        "🔥 Recommendation",
        use_container_width=True
    ):
        st.switch_page(
            "pages/03_Recommendation.py"
        )


# =====================================================
# TABLEAU DASHBOARD
# =====================================================

with c6:

    st.markdown(
        """
        <a href="https://public.tableau.com/views/ML_Recipe_Project_v2026_1/RecipeInsightsSentimentDashboard?:showVizHome=no"
           target="_blank"
           style="
               display:flex;
               align-items:center;
               justify-content:center;
               width:100%;
               height:50px;
               background-color:#ff6b35;
               color:white;
               text-decoration:none;
               border-radius:10px;
               font-size:14px;
               font-weight:600;
               box-sizing:border-box;
               margin:0;
               padding:0;
           ">
            📊 Tableau
        </a>
        """,
        unsafe_allow_html=True
    )