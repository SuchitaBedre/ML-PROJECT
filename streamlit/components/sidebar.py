import os
import streamlit as st


def sidebar():

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    logo_path = os.path.join(
        base_dir,
        "assets",
        "logo.png"
    )

    with st.sidebar:

        # Logo
        if os.path.exists(logo_path):

            st.image(
                logo_path,
                use_container_width=True
            )