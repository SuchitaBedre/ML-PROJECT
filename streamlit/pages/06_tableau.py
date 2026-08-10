import streamlit as st

st.set_page_config(
    page_title="Tableau Dashboard",
    page_icon="📊",
    layout="wide"
)

TABLEAU_URL = "https://public.tableau.com/views/ML_Recipe_Project/RecipeInsightsSentimentDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"

st.markdown(
    f"""
    <meta http-equiv="refresh" content="0; url={TABLEAU_URL}">
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    # 📊 Tableau Dashboard

    Your Tableau dashboard is opening...

    **[Open Tableau Dashboard]({TABLEAU_URL})**
    """,
    unsafe_allow_html=True
)