import streamlit as st
import pandas as pd
import numpy as np


st.title('NBA MVP predict')
st.markdown("""
This app performs a simple machine learning model to predict the NBA MVP for the 2022 season,
* **  Python libraries:** Requests, Pandas, Sklearn...,
* ** Data source:** [basketball-reference.com] (https://www.basketball-reference.com/)
""")
    

st.sidebar.header('Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2022))))

