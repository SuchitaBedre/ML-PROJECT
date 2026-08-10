import streamlit as st

st.set_page_config(
    page_title="AI Recipe Platform",
    page_icon="🍲",
    layout="wide"
)

st.title("🍲 AI Powered Recipe Recommendation & Rating Prediction")

st.write("""
Welcome to AI Recipe Platform.

Features:
- ⭐ Recipe Rating Prediction
- 🔍 Recipe Search
- 🥗 Similar Recipe Recommendation
- 🤖 AI Recipe Assistant
- 📊 Recipe Analytics Dashboard
""")

st.info("Select a page from the sidebar.")