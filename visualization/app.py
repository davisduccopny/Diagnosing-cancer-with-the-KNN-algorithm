import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
import base64
import datetime
import os
from PIL import Image
from datetime import date
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import os
import numpy as np
import math
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, explained_variance_score, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from pickle import TRUE
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

st.set_page_config(layout='wide',page_title="Stock Prediction", initial_sidebar_state='expanded')
from LIBRARIES import MAINCLASS
app = MAINCLASS()
# Tải giao diện
app.run()
# Chạy hàm chọn cổ phiếu
app.select_stock_options()
# Chạy hàm tải dữ liệu từ cổ phiếu
data = app.download_data(app.option_stock_name,app.start_date,app.end_date)
# Tiến hành làm sạch dữ liệu
data = app.clean_dataframe(dataframe=data)
# Đặt biến self.data trong MAINCLASS với dữ liệu mới để thực thi các hàm trong đó
app.data = data
# Thực thi hàm main
if __name__ == '__main__':
    app.main()
app.footer_info()