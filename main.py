# app.py

import streamlit as st

from bandit_utils import init_state

st.set_page_config(
    page_title="Bayesian Carousel Bandits",
    page_icon="",
    layout="wide",
)

init_state()

navigation = st.navigation(
    [
        st.Page("views/Create_Carousels.py", title="Create Carousels", default=True),
        st.Page("views/Interaction.py", title="Interaction"),
        st.Page("views/Analytics.py", title="Analytics"),
        st.Page("views/Gaussian_Bandit.py", title="Gaussian Bandit"),
    ]
)

navigation.run()
