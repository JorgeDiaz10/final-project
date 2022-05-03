<p align="centre"><img src="https://www.pngplay.com/wp-content/uploads/13/Logo-NBA-Free-PNG.png"></p>

# **NBA MVP Predictor**

Ironhack Madrid - Data Analytics Part Time - May 2022 - Final Project

---

##  *Introduction*

This repository will be focusing on the statistics revolving around NBA from 1995 and it aims at predicting the player who will win the 2022 NBA MVP award, by modelling the voting panel.

---

## :computer: *Installation*

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries.

```bash
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
```

---

## :file_folder: *Folder structure*

```
└── final-project
    ├── Players
    │   └── players(1995-2022).html
    ├── Standings
    │   └── standings(1995-2022).html
    ├── mvp
    │   └── mvp(1995-2021).html
    ├── data
        ├── mvp.csv
        ├── nicknames.txt
        ├── player_stats.csv
        ├── players.csv
        └── teams.csv
    ├── Functions
        ├── acquisition_functions.ipynb
        ├── cleaning_functions.ipynb
        └── machine_learning_functions.ipynb
    ├── images
        └── nbalogo.jpg
    ├── .gitignore
    ├── main.py
    └── README.md
```
---

## *Code*

- Get 3 DataFrames Scrapping the web Basketaball-reference.
- Once we have scrapped, we start cleaning each of the DataFrames. I have used Jupyter Notebook.
- We got everything clean!! We do have to merge the 3 Dataframes into one to get the DataFrame with all the stats
  from 1995 to today.
- Since we have the data prepare, we can now try to predict the Player who is going to be the MVP.
- I have used a simple machine learning linear model (ridge) selecting all the columns but 'Share' as features. We will use
  the column 'Share' as the target.
- After the prediction is done, we do some plots to compare between the top player in the 2021-22.
- To show the job done, we run Streamlit in Terminal.
- The Streamlit app shows 2 buttons:
    - In the first one, you can check every player stat and the webpage nbamaniacs to see daily news.
    - In the second one, you see the DataFrame with the top 10 NBA MVP predictions with some plots to contrast the model 
      used.
- The command used in Terminal: 

```
                streamlit run main.py
```

---

## *To do*



---

## *License*
- [Basketball-reference](https://www.basketball-reference.com/)

---

## *Project Main Stack*

- [Requests](https://requests.readthedocs.io/)

- [Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

- [Streamlit](https://docs.streamlit.io/)

- [AgGrid](https://pypi.org/project/streamlit-aggrid/)

- [Sklearn](https://scikit-learn.org/stable/modules/classes.html)

- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

- [Plotly](https://plotly.com/python/pandas-backend/)











 


 

