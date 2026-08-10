# =====================================================
# 05_AI_Assistant.py
# AI Recipe Assistant
# =====================================================


import streamlit as st


from api import ai_assistant

from components.sidebar import sidebar

from styles import PAGE_STYLE



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="AI Recipe Assistant",

    page_icon="🤖",

    layout="wide"

)



# =====================================================
# STYLE
# =====================================================

st.markdown(

PAGE_STYLE +

"""
<style>


h1,h2,h3,h4,p {

    text-align:center;

}



div.stButton > button {

    width:100%;

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
text-align:center;
">

🤖 AI Recipe Assistant

</h1>



<p style="
font-size:20px;
color:white;
text-align:center;
">

Ask anything about recipes, cooking methods,
nutrition, ingredients and recommendations
using AI powered RAG.

</p>

</div>

""",

unsafe_allow_html=True

)



st.write("")



# =====================================================
# SESSION STATE
# =====================================================


if "messages" not in st.session_state:

    st.session_state.messages = []



# =====================================================
# QUICK QUESTIONS
# =====================================================


st.subheader(

    "💡 Quick Questions"

)



q1,q2,q3,q4,q5 = st.columns(5)



with q1:


    if st.button(

        "🍗 Chicken",

        use_container_width=True

    ):


        st.session_state.quick_question = (

            "Suggest the best chicken recipes."

        )




with q2:


    if st.button(

        "🥗 Healthy",

        use_container_width=True

    ):


        st.session_state.quick_question = (

            "Suggest healthy recipes."

        )




with q3:


    if st.button(

        "🍰 Dessert",

        use_container_width=True

    ):


        st.session_state.quick_question = (

            "Suggest dessert recipes."

        )




with q4:


    if st.button(

        "🥘 Indian Food",

        use_container_width=True

    ):


        st.session_state.quick_question = (

            "Suggest popular Indian recipes."

        )




with q5:


    if st.button(

        "🔥 Trending",

        use_container_width=True

    ):


        st.session_state.quick_question = (

            "Suggest trending recipes."

        )



st.write("")



# =====================================================
# CLEAR CHAT
# =====================================================


if st.button(

    "🗑 Clear Chat",

    use_container_width=True

):


    st.session_state.messages = []


    st.rerun()



st.divider()



# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================


for message in st.session_state.messages:


    with st.chat_message(

        message["role"]

    ):


        if message["role"] == "assistant":


            st.markdown(

            f"""

            <div class="recipe-card">


            <h3 style="
            color:#ff6b35;
            text-align:center;
            ">

            👨‍🍳 AI Chef Response

            </h3>


            <p style="
            font-size:17px;
            line-height:1.8;
            text-align:left;
            ">

            {message["content"].replace(chr(10),"<br>")}

            </p>


            </div>

            """,

            unsafe_allow_html=True

            )


        else:


            st.write(

                message["content"]

            )



# =====================================================
# USER INPUT
# =====================================================


question = st.chat_input(

    "Ask anything about recipes..."

)



if not question and "quick_question" in st.session_state:


    question = st.session_state.quick_question


    del st.session_state.quick_question




# =====================================================
# SEND REQUEST
# =====================================================


if question:


    st.session_state.messages.append(

        {

            "role":"user",

            "content":question

        }

    )



    with st.chat_message(

        "user"

    ):


        st.write(question)



    with st.chat_message(

        "assistant"

    ):


        with st.spinner(

            "🤖 Searching ChromaDB + Gemini..."

        ):


            try:


                response = ai_assistant(

                    question

                )



                if response.get(

                    "answer"

                ):


                    answer = response["answer"]



                else:


                    answer = (

                        "❌ Backend Error\n\n"

                        +

                        response.get(

                            "message",

                            "Unknown error"

                        )

                    )



            except Exception as e:


                answer = (

                    "❌ Cannot connect to backend\n\n"

                    +

                    str(e)

                )



        st.markdown(

        f"""

        <div class="recipe-card">


        <h3 style="
        color:#ff6b35;
        text-align:center;
        ">

        👨‍🍳 AI Chef Recommendation

        </h3>


        <p style="
        font-size:17px;
        line-height:1.8;
        text-align:left;
        ">

        {answer.replace(chr(10),"<br>")}

        </p>


        </div>


        """,

        unsafe_allow_html=True

        )



    st.session_state.messages.append(

        {

            "role":"assistant",

            "content":answer

        }

    )