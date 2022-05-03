import pandas as pd
import numpy as np
import requests
import os
import shutil
from bs4 import BeautifulSoup

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import Ridge

import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridUpdateMode, JsCode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


import chart_studio.plotly as py
import plotly.graph_objects as go

from PIL import Image

# Extract final NBA players stats
@st.cache
def final_stats():
    stats_combined = pd.read_csv('data/player_stats.csv')
    stats_combined = stats_combined.drop(columns = ['Unnamed: 0'])
    stats_combined = stats_combined[['Player', 'Pos', 'Age', 'Tm','Year', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 
       'Pts Won', 'Pts Max', 'Share', 'Team', 'W', 'L', 'W/L%', 'GB', 'PS/G',
       'PA/G', 'SRS']]
    stats_combined.fillna(0, inplace=True)
    stat_ratios = stats_combined[["PTS", "AST", "STL", "BLK", "3P", "Year"]].groupby("Year").apply(lambda x: x/x.mean())
    stats_combined[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = stat_ratios[["PTS", "AST", "STL", "BLK", "3P"]]
    return stats_combined

# Get the NBA MVP for the 2021-22 season
@st.cache
def predictions():
    features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA',
            '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 
            'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS',
            'PTS_R', 'AST_R', 'STL_R', 'BLK_R', '3P_R']
    target = ['Share']
    df_train = stats_combined[(stats_combined["Year"] < 2022)]
    df_test = stats_combined[stats_combined["Year"] == 2022]
    model = Ridge()
    model.fit(df_train[features],df_train["Share"])
    pred = model.predict(df_test[features])
    pred = pd.DataFrame(pred, columns=["predictions"], index=df_test.index)
    compared = pd.concat([df_test[["Player", "Pos", "Age", "Tm", "G", "GS"]], pred], axis=1)
    mvp_prediction = compared.sort_values("predictions", ascending=False).round({'predictions': 3}).head(10)
    mvp_prediction = mvp_prediction.reset_index(drop=True)
    mvp_prediction.insert(1, "Ranking", [1,2,3,4,5,6,7,8,9,10])
    mvp_prediction.set_index("Player",inplace = True)
    return mvp_prediction

@st.cache
def players_df():
    players_stats = stats_combined.to_csv().encode('utf-8')
    return players_stats

@st.cache
def mvp_df():
    mvp_csv = mvp_prediction.to_csv().encode('utf-8')
    return mvp_csv

def points():
    df_actual_year = stats_combined[stats_combined['Year'] == 2022]
    df_max_points = df_actual_year[df_actual_year['PTS'] > 27]
    return df_max_points

def assists():
    df_actual_year = stats_combined[stats_combined['Year'] == 2022]
    df_max_assists = df_actual_year[df_actual_year['AST'] > 7.5]
    return df_max_assists

def rebounds():
    df_actual_year = stats_combined[stats_combined['Year'] == 2022]
    df_max_rebounds = df_actual_year[df_actual_year['TRB'] > 11]
    return df_max_rebounds

def plot_points():
    fig = go.Figure(data=[go.Bar(x=df_max_points.Player,
            y=df_max_points.PTS)],layout_title_text="PTS per game")
    return fig

def plot_assists():
    fig_2 = go.Figure(data=[go.Bar(x=df_max_assists.Player,
            y=df_max_assists.AST)],layout_title_text="AST per game")
    return fig_2

def plot_rebounds():
    fig_3 = go.Figure(data=[go.Bar(x=df_max_rebounds.Player,
            y=df_max_rebounds.TRB)],layout_title_text="TRB per game")
    return fig_3

if __name__ == "__main__":
    stats_combined = final_stats()
    mvp_prediction = predictions()
    players_stats = players_df()
    mvp_csv = mvp_df()
    df_max_points = points()
    df_max_assists = assists()
    df_max_rebounds = rebounds()
    fig = plot_points()
    fig_2 = plot_assists()
    fig_3 = plot_rebounds()


#image = Image.open('images/nbalogo.jpg')
st.title("🏀 NBA Stats and MVP")
#set title and description
page_names = ['Players Stats', 'MVP predictor']
page = st.radio('Navigate to', page_names)
if page == "Players Stats":
        st.title('***Individual Player Stats***')
        st.subheader('Welcome to the Players Stats page')
        years_list = 'Year', stats_combined['Year'].unique()
        filtered_df = stats_combined[stats_combined['Year'].isin(years_list)]

        _funct = st.sidebar.radio(label='Select Input Parameters', options=['Display', 'Highlight'])
        st.sidebar.markdown("""*Made by [Jorge Díaz](https://github.com/JorgeDiaz10). Code on [GitHub](https://github.com/JorgeDiaz10/final-project).*""")
        gd = GridOptionsBuilder.from_dataframe(stats_combined)
        gd.configure_pagination(enabled=True)
        gd.configure_default_column(editable=True, groupable=True)

        if _funct == 'Display':
            sel_mode = st.radio('Selection Type', options = ['single', 'multiple'])
            gd.configure_selection(selection_mode=sel_mode, use_checkbox=True)
            gridoptions = gd.build()
            grid_table = AgGrid(stats_combined, gridOptions=gridoptions, 
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height= 500, allow_unsafe_jscode=True, theme='streamlit')
            sel_row = grid_table["selected_rows"]
            st.subheader('*Complete Stats*')
            st.write(sel_row)
        if _funct == 'Highlight':
            col_opt = st.selectbox(label='Select column', options = stats_combined.columns)
            cellstyle_jscode = JsCode("""
                function(params){
                    if (params.value == 'Giannis Antetokounmpo') {
                        return {
                            'color': 'black',
                            'backgroundColor' : 'orange'
                    }
                    }
                    if (params.value == 'Nikola Jokić') {
                        return{
                            'color'  : 'black',
                            'backgroundColor' : 'red'
                        }
                    }
                    if (params.value == 'Joel Embiid') {
                        return{
                            'color'  : 'black',
                            'backgroundColor' : 'red'
                        }
                    }
                    else{
                        return{
                            'color': 'black',
                            'backgroundColor': 'lightpink'
                        }
                    }
            
            };
            """)
            gd.configure_columns(col_opt, cellStyle=cellstyle_jscode)
            gridOptions = gd.build()
            grid_table = AgGrid(stats_combined, 
            gridOptions = gridOptions, 
            height = 500,
            theme = "streamlit",
            update_mode = GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            )
        st.download_button(
            label="Download dataframe as CSV",
            data=players_stats,
            file_name='PlayersStats.csv',
            mime='text/csv',)
        st.subheader('*Check more daily information*')
        components.iframe("https://www.nbamaniacs.com/", height= 1000, width= 1000, scrolling = True)
        #st.balloons()
else:
    st.title('*NBA MVP Predictor*')
    st.markdown("""
        This app performs a simple machine learning model to predict the NBA MVP for the 2022 season
        * **Python libraries:** Requests, Pandas, Sklearn, Streamlit, Chart_studio.plotly...
        * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
        """)
    check = st.checkbox('Show data')
    if check:
            st.header("Current year predictions")
            col1 = st.columns(1)
            st.subheader("Predicted top 3")
            top_3 = mvp_prediction["Ranking"][:3].to_dict()
            emojis = ["🥇", "🥈", "🥉"]

            for n, player_name in enumerate(top_3):
                title_level = "###" + n * "#"
                st.markdown(f"""##### {emojis[n]} *{player_name}*""")
            st.dataframe(data=mvp_prediction, width=None, height=None)
            st.download_button(
            label="Download table as CSV",
            data=mvp_csv,
            file_name='MVP.csv',
            mime='text/csv',)

    with st.expander('Points per game'):
            st.plotly_chart(fig, use_container_width=True)
    with st.expander('Assists per game'):
            st.plotly_chart(fig_2, use_container_width=True)
    with st.expander('Rebounds per game'):
            st.plotly_chart(fig_3, use_container_width=True)
    #st.balloons()



        