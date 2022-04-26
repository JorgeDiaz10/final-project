import pandas as pd
import numpy as np
import requests
import os
import shutil
from bs4 import BeautifulSoup
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

import streamlit as st



#set title and description
st.title('NBA MVP predictor')
st.markdown("""
This app performs a simple machine learning model to predict the NBA MVP for the 2022 season
* **Python libraries:** Requests, Pandas, Sklearn...
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

#Create sidebar
sidebar =st.sidebar
dataset = st.sidebar.header('User input features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1995, 2022))))

# Extract final NBA players stats
def final_stats():
    stats_combined = pd.read_csv('data/player_stats.csv')
    stats_combined = stats_combined.drop(columns = ['Unnamed: 0'])
    stats_combined.fillna(0, inplace=True)
    return stats_combined


if __name__ == "__main__":
    stats_combined = final_stats()


st.dataframe(stats_combined)





