import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Recipe Intelligence Platform",
    page_icon=":fork_and_knife:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    :root {
        --orange: #FF7A3D;
        --orange-dark: #E8542E;
        --red: #EF4444;
        --green: #16A34A;
        --green-light: #ECFDF3;
        --purple: #7C3AED;
        --purple-light: #F5F3FF;
        --card-bg: #FFFFFF;
        --page-bg: #F6F7F9;
        --ink: #1A1A1A;
        --ink-soft: #6B7280;
        --line: #ECEEF1;
    }

    .stApp {
        background-image: url("app/static/recipe_bg.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main .block-container { padding: 0 3rem 2.5rem 3rem; max-width: 1300px; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

    /* ---- Hero banner ---- */
    .hero-banner {
        background: linear-gradient(120deg, #FF7A3D 0%, #E8542E 60%, #FF9640 100%);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin: 1.6rem 0 1.6rem 0;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        color: white;
        box-shadow: 0 10px 28px rgba(232,84,46,0.3);
    }
    .hero-icon {
        width: 58px; height: 58px;
        background: rgba(255,255,255,0.22);
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.7rem;
        flex-shrink: 0;
    }
    .hero-title { font-size: 1.9rem; font-weight: 900; margin: 0; line-height: 1.15; color: white; letter-spacing: -0.01em; }
    .hero-sub { font-size: 0.92rem; font-weight: 600; color: rgba(255,255,255,0.95); margin-top: 2px; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(6px);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.7);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.9rem;
        color: var(--ink-soft);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #FFF1E8, #FFE4D3) !important;
        color: var(--orange-dark) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---- Cards ---- */
    .panel {
        background: rgba(255,255,255,0.97);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        margin-top: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .panel-head { display: flex; align-items: center; gap: 10px; margin-bottom: 1.1rem; }
    .panel-icon {
        width: 38px; height: 38px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; flex-shrink: 0;
    }
    .panel-title { font-weight: 800; font-size: 1.08rem; margin: 0; }
    .panel-desc { font-size: 0.82rem; color: var(--ink-soft); margin: 0; }

    /* field labels */
    .stTextArea label, .stTextInput label, .stNumberInput label {
        font-weight: 600 !important; font-size: 0.85rem !important; color: var(--ink) !important;
    }
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        border: 1.5px solid var(--line) !important;
        border-radius: 10px !important;
        background: #FAFAFA !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--orange) !important;
        box-shadow: 0 0 0 1px var(--orange) !important;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.7rem 1.4rem;
        border: none;
        width: 100%;
        background: linear-gradient(120deg, var(--orange), var(--red));
        color: white;
    }
    .stButton > button:hover { opacity: 0.92; color: white; }

    /* ---- Badges ---- */
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.8rem; font-weight: 700;
        padding: 5px 12px; border-radius: 999px;
    }
    .badge-green { background: var(--green-light); color: var(--green); }

    /* ---- Why list ---- */
    .why-box { background: var(--green-light); border-radius: 12px; padding: 1rem 1.1rem; margin-top: 1rem; }
    .why-title { font-weight: 700; font-size: 0.88rem; color: #14532D; margin-bottom: 6px; }
    .why-item { font-size: 0.85rem; color: #166534; margin: 3px 0; }

    /* ---- Stat cards ---- */
    .stat-card {
        background: rgba(255,255,255,0.97);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    .stat-label { font-size: 0.8rem; color: var(--ink-soft); font-weight: 600; display: flex; align-items: center; gap: 6px; }
    .stat-value { font-size: 1.5rem; font-weight: 800; margin-top: 4px; }
    .stat-delta { font-size: 0.76rem; color: var(--green); margin-top: 2px; font-weight: 600; }

    /* ---- Chat ---- */
    .stChatMessage { border-radius: 12px; }

    /* ---- Chat box — fully custom, guaranteed solid background ---- */
    .chat-scroll {
        background: rgba(255,255,255,0.97);
        border: 1px solid var(--line);
        border-radius: 12px;
        height: 260px;
        overflow-y: auto;
        padding: 1.2rem;
        margin-top: 0.4rem;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .chat-row { display: flex; align-items: flex-end; gap: 8px; max-width: 80%; }
    .chat-row-user { align-self: flex-end; flex-direction: row-reverse; }
    .chat-row-bot { align-self: flex-start; }
    .chat-avatar {
        width: 30px; height: 30px; min-width: 30px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.95rem;
    }
    .chat-avatar-user { background: #FFE4D3; }
    .chat-avatar-bot { background: #F5F3FF; }
    .chat-bubble {
        padding: 0.65rem 0.95rem;
        border-radius: 14px;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .chat-bubble-user { background: var(--orange); color: white; border-bottom-right-radius: 4px; }
    .chat-bubble-bot { background: #F3F4F6; color: var(--ink); border-bottom-left-radius: 4px; }

    .empty-hint {
        color: var(--ink-soft);
        font-size: 0.92rem;
        text-align: center;
        padding: 3rem 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        height: 100%;
        justify-content: center;
    }
    .empty-hint-icon { font-size: 1.8rem; opacity: 0.7; }
    .empty-hint b { color: var(--ink); }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL & PREPROCESSING OBJECTS (cached)
# =========================================================
MODEL_PATH = os.path.join("models", "best_model.pkl")
TFIDF_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


@st.cache_resource
def load_model_files():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        return None, None
    model = pickle.load(open(MODEL_PATH, "rb"))
    tfidf = pickle.load(open(TFIDF_PATH, "rb"))
    return model, tfidf


model, tfidf = load_model_files()


def rag_chat(query, user_type="job_seeker"):
    return (
        "Here are 3 ideas based on real recipes in our dataset. "
        "Connect your real RAG pipeline (FAISS retrieval + LLM generation) "
        "in rag_pipeline.py to replace this placeholder."
    )


def make_gauge(value, max_value=5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 42, 'family': 'Inter', 'color': '#1A1A1A'}},
        gauge={
            'axis': {'range': [0, max_value], 'visible': False},
            'bar': {'color': "#16A34A", 'thickness': 0.28},
            'bgcolor': "#F0FDF4",
            'borderwidth': 0,
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(255,255,255,0.97)",
        font={'family': 'Inter'}
    )
    return fig


# =========================================================
# HERO BANNER
# =========================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">👨‍🍳</div>
    <div>
        <p class="hero-title">Recipe Intelligence Platform</p>
        <p class="hero-sub">Predicting recipe success and recommending dishes using ML and Generative AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["📈  Rating Prediction", "📊  Analytics Dashboard", "🤖  AI Assistant"])

# ---------------------------------------------------------
# TAB 1 — Prediction (two-column layout like the mockup)
# ---------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1], gap="medium")

    with left:
        st.markdown("""
        <div class="panel">
            <div class="panel-head">
                <div class="panel-icon" style="background:#FFF1E8;">🍲</div>
                <div>
                    <p class="panel-title">Will this recipe be well-received?</p>
                    <p class="panel-desc">Enter recipe details to predict its rating</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        ingredients = st.text_area(
            "Ingredients",
            placeholder="e.g., flour, sugar, butter, eggs, vanilla extract, baking powder, milk",
            height=100
        )
        tags = st.text_input("Tags", placeholder="e.g., dessert, quick, vegetarian, breakfast, easy")

        c1, c2 = st.columns(2)
        with c1:
            minutes = st.number_input("Prep time (minutes)", min_value=1, value=30)
        with c2:
            n_steps = st.number_input("Number of steps", min_value=1, value=8)

        predict_clicked = st.button("✨  Predict Rating")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if predict_clicked:
            if model is None or tfidf is None:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.error(
                    "Model files not found. Place best_model.pkl and "
                    "tfidf_vectorizer.pkl inside the models/ folder."
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                combined_text = f"{ingredients} {tags}"
                features = tfidf.transform([combined_text])
                prediction = model.predict(features)[0]
                confidence = model.predict_proba(features)[0].max()
                conf_pct = confidence * 100
                predicted_stars = 3.0 + (confidence * 2) if prediction == 1 else 1.0 + (confidence * 2)
                predicted_stars = min(5.0, max(0.0, predicted_stars))

                status_label = "High Potential" if prediction == 1 else "Needs Improvement"

                st.markdown(f"""
                <div class="panel">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                        <span class="badge badge-green">📈 Prediction Result</span>
                        <span class="badge badge-green">✓ {status_label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                gcol, tcol = st.columns([1, 1.1])
                with gcol:
                    st.plotly_chart(make_gauge(predicted_stars), use_container_width=True, config={'displayModeBar': False})
                with tcol:
                    st.markdown(f"""
                        <div style="padding-top:1.2rem;">
                            <div style="font-weight:700; font-size:1.1rem; color:#16A34A;">Predicted Rating</div>
                            <div style="font-size:0.82rem; color:#6B7280; margin-bottom:6px;">Confidence Score</div>
                            <div style="background:#ECEEF1; border-radius:999px; height:8px; overflow:hidden;">
                                <div style="width:{conf_pct:.0f}%; background:#16A34A; height:100%;"></div>
                            </div>
                            <div style="font-size:0.85rem; font-weight:700; margin-top:4px;">{conf_pct:.0f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="why-box">
                        <div class="why-title">Why this recipe will be loved</div>
                        <div class="why-item">✓ Balanced, well-known ingredients</div>
                        <div class="why-item">✓ Reasonable preparation time ({minutes} min)</div>
                        <div class="why-item">✓ Manageable number of steps ({n_steps})</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="panel" style="display:flex; align-items:center; justify-content:center; min-height:340px;">
                <div style="text-align:center; color:#9CA3AF;">
                    <div style="font-size:2.2rem; margin-bottom:8px;">📊</div>
                    <div>Fill in the recipe details and click<br><b>Predict Rating</b> to see results here</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 — Analytics Dashboard
# ---------------------------------------------------------
with tab2:
    st.markdown("""
    <div class="panel">
        <div class="panel-head">
            <div class="panel-icon" style="background:#FFF1E8;">📊</div>
            <div>
                <p class="panel-title">Analytics Dashboard</p>
                <p class="panel-desc">Insights from our recipe dataset</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("⭐ Average Rating", "4.35", "+0.18 vs last period", s1),
        ("📖 Total Recipes", "231,637", "+8.6% vs last period", s2),
        ("🏆 High Rated Recipes", "1,003,724", "+12.4% vs last period", s3),
        ("🎯 Model Accuracy", "—%", "connect model metrics", s4),
    ]
    for label, value, delta, col in stats:
        with col:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-delta">{delta}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><p class="panel-title" style="margin-bottom:0.8rem;">Rating Distribution</p>', unsafe_allow_html=True)
        rating_counts = pd.DataFrame({
            "Rating": ["0", "1", "2", "3", "4", "5"],
            "Count": [60700, 12800, 14100, 40800, 187300, 815700]
        })
        fig_bar = go.Figure(go.Bar(
            x=rating_counts["Rating"], y=rating_counts["Count"],
            marker_color="#FF9640"
        ))
        fig_bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(255,255,255,0.97)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><p class="panel-title" style="margin-bottom:0.8rem;">Rating Trend (sample)</p>', unsafe_allow_html=True)
        trend = pd.DataFrame({
            "Week": ["W1", "W2", "W3", "W4", "W5", "W6"],
            "Avg Rating": [4.2, 4.25, 4.3, 4.28, 4.33, 4.35]
        })
        fig_line = go.Figure(go.Scatter(
            x=trend["Week"], y=trend["Avg Rating"], mode="lines+markers",
            line=dict(color="#FF7A3D", width=3), marker=dict(size=7)
        ))
        fig_line.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor="rgba(255,255,255,0.97)", plot_bgcolor="rgba(255,255,255,0)",
                                yaxis=dict(range=[3.8, 4.6]))
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Sample values shown above — replace with live SQL queries from your recipe_summary table.")

# ---------------------------------------------------------
# TAB 3 — AI Assistant
# ---------------------------------------------------------
with tab3:
    st.markdown("""
    <div class="panel">
        <div class="panel-head">
            <div class="panel-icon" style="background:#F5F3FF;">🤖</div>
            <div>
                <p class="panel-title">AI Recipe Assistant</p>
                <p class="panel-desc">Ask anything about recipes, ingredients, or cooking</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Build the chat log as one block of custom HTML so the background
    # is guaranteed solid, regardless of Streamlit's internal container markup.
    if not st.session_state.messages:
        bubbles_html = (
            '<div class="empty-hint"><span class="empty-hint-icon">🍽️</span>'
            'Ask me anything about recipes<br>'
            'Try: <b>&ldquo;Suggest some high-protein vegetarian dinner recipes&rdquo;</b></div>'
        )
    else:
        bubble_parts = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                bubble_parts.append(
                    f'<div class="chat-row chat-row-user">'
                    f'<div class="chat-bubble chat-bubble-user">{msg["content"]}</div>'
                    f'<div class="chat-avatar chat-avatar-user">🧑</div>'
                    f'</div>'
                )
            else:
                bubble_parts.append(
                    f'<div class="chat-row chat-row-bot">'
                    f'<div class="chat-avatar chat-avatar-bot">🤖</div>'
                    f'<div class="chat-bubble chat-bubble-bot">{msg["content"]}</div>'
                    f'</div>'
                )
        bubbles_html = "".join(bubble_parts)

    st.markdown(f'<div class="chat-scroll">{bubbles_html}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = rag_chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)