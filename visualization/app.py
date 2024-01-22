from streamlit_option_menu import option_menu
import streamlit as st
import base64
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
sns.set(style='darkgrid', font_scale=1.4)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import time
import optuna
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions
# Import thư viện sklearn nhằm mục đích so sánh với mô hình tự xây dựng
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(layout='wide',page_title="Dianosis prediction",page_icon= '🖥️' ,initial_sidebar_state='expanded')
class DESCRIPTIVE_STATISTICS:
    def __init__(self, df):
        self.df = df
        self.df_statistic = self.df[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']]
        self.df_statistic['diagnosis'] = (self.df_statistic['diagnosis'] =='M').astype('int')
    def describe(self):
        describe = self.df.describe()
        return describe
    def subplot_histograms(self):
        fig = sp.make_subplots(rows=3, cols=4, subplot_titles=self.df_statistic.columns[0:])

        for i, feature in enumerate(self.df_statistic.columns[0:]):
            fig.add_trace(go.Histogram(x=self.df_statistic[feature]), row=(i // 4) + 1, col=(i % 4) + 1)
        fig.update_layout(title_text='Histogram Subplots', showlegend=False,
        width=13*80,
        height=10*80,
        title_x=0.4,
        bargap=0.04)
        return fig
    def pairplot(self):
        fig = px.scatter_matrix(self.df_statistic, dimensions=self.df_statistic.columns[1:], color='diagnosis')
        fig.update_layout(title_text='Mối quan hệ giữa các biến',
        width=13*80,
        height=12*80,
        title_x=0.4,)
        return fig
    def heatmap(self):
        correlation_matrix = self.df_statistic.corr()

        fig = px.imshow(
            correlation_matrix,
            x=self.df_statistic.columns,
            y=self.df_statistic.columns,
            color_continuous_scale='RdYlGn',
            labels=dict(color='Correlation'),
            title='Heatmap'
        )

        fig.update_layout( xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0),
        width=13*80,
        height=12*80,
        title_x=0.4,)
        return fig
class KNN_MODELS():
    # Khởi tạo các biến trong hàm init
    def __init__(self, k=3, metric='euclidean', p=None):
        self.k = k
        self.metric = metric
        self.p = p
    
    # Xây dựng độ đo euclidenan
    def euclidean(self, v1, v2):
        return np.sqrt(np.sum((v1-v2)**2))
    
    # Xây dựng độ đo manhattan
    def manhattan(self, v1, v2):
        return np.sum(np.abs(v1-v2))
    
    # Xây dựng độ đo minkowski
    def minkowski(self, v1, v2, p=2):
        return np.sum(np.abs(v1-v2)**p)**(1/p)
        
    # Xây dựng hàm fit
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    # Xây dựng hàm Predict
    def predict(self, X_test):
        preds = []
        for test_row in X_test:
            nearest_neighbours = self.get_neighbours(test_row)
            majority = Counter(nearest_neighbours).most_common(1)[0][0]
            preds.append(majority)
        return np.array(preds)
        

    
    # Xây dựng hàm lấy điểm gần nhất
    def get_neighbours(self, test_row):
        distances = list()
        
        # Tính khoảng cách tất cả các điểm trong tập train
        for (train_row, train_class) in zip(self.X_train, self.y_train):
            if self.metric=='euclidean':
                dist = self.euclidean(train_row, test_row)
            elif self.metric=='manhattan':
                dist = self.manhattan(train_row, test_row)
            elif self.metric=='minkowski':
                dist = self.minkowski(train_row, test_row, self.p)
            else:
                raise NameError('Supported metrics are euclidean, manhattan and minkowski')
            distances.append((dist, train_class))
            
        # sắp xếp lại 
        distances.sort(key=lambda x: x[0])
        
        # Xác định k
        neighbours = list()
        for i in range(self.k):
            neighbours.append(distances[i][1])
            
        return neighbours
class TRAIN_MODELS:
    def __init__(self, df):
        self.df = df
        self.slideroption = 5
        self.slideroption_p = 2.5
        self.option_distance  = 'euclidean'
        
    def train_test_split(self):
        self.df.reset_index(drop=True)
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        y = (y=='M').astype('int')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
        return X_train, X_test, y_train, y_test
    
    def accuracy(self,preds, y_test):
        return 100 * (preds == y_test).mean()
    
    def run_model_option(self,metric,k,p):
        X_train, X_test, y_train, y_test = self.train_test_split()
        clf = KNN_MODELS(k=k, metric=metric, p=p)
        clf.fit(X_train.values, y_train.values)
        preds = clf.predict(X_test.values)
        return preds,y_test
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cnf_matrix = metrics.confusion_matrix(y_true, y_pred)

        fig = go.Figure()
        cell_values = [[cnf_matrix[1, 1], cnf_matrix[0, 1]],
                       [cnf_matrix[1, 0], cnf_matrix[0, 0]]]
        fig.add_trace(go.Heatmap(z=cell_values,
                                 x=['Predicted Positive', 'Predicted Negative'],
                                 y=['Actual Positive', 'Actual Negative'],
                                 colorscale="YlGnBu",
                                 showscale=True,
                                 colorbar=dict(tickvals=[val for sublist in cell_values for val in sublist],
                                               ticktext=[str(val) for sublist in cell_values for val in sublist])))
        fig.update_layout(title='Confusion Matrix',
                          xaxis=dict(title='Predicted label'),
                          yaxis=dict(title='Actual label'),
                          width=13*80,
                          height=6*80,
                          title_x=0.45)

        return fig
    
    def run_knn(self):
        col1,col2 = st.columns(2)
        with col1:
            self.option_distance =  st.radio('Chọn độ đo', [
                                 'euclidean', 'manhattan', 'minkowski'])
        with col2:
            self.slideroption = st.number_input('Chọn giá trị k',value=self.slideroption)
            if self.option_distance =='minkowski':
                self.slideroption_p = st.number_input('Chọn giá trị p', value=self.slideroption_p)
        preds,y_test = self.run_model_option(metric=self.option_distance,k=self.slideroption,p=self.slideroption_p)
        accuracy = self.accuracy(preds,y_test)
        if st.button('Predict'):
            st.success(f"Độ chính xác: {accuracy}%")
            st.plotly_chart(self.plot_confusion_matrix(y_true=y_test, y_pred=preds))
            st.subheader("classification report")
            report = classification_report(y_test, preds, output_dict=True)
            st.table(pd.DataFrame(report))
            datapred = pd.DataFrame({'True Labels': y_test, 'Predicted Labels': preds})
            with st.expander("📈 Data Predict 📈"):
                st.dataframe(datapred)
def run():
    st.markdown("<h1 style='text-align: center;'>Diagnosing cancer with the KNN algorithm</h1>", unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <h1 style='font-size:35px;text-align:center'>TEAM 1</h1>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
        <div style="display: flex; justify-content: center;margin-bottom:0">
            <img src='https://scontent.fdad3-5.fna.fbcdn.net/v/t39.30808-6/242489593_405101811147345_1733417058228090429_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=173fa1&_nc_eui2=AeFuaEaf18BbprUuxRa5lYL8wUu8nFmqFBHBS7ycWaoUETQ21uzivfHWo-qW4uimAiH9d-O-sZIVXWkJaRHH_YCo&_nc_ohc=WJ6z-ROQw4MAX_p73ht&_nc_ht=scontent.fdad3-5.fna&oh=00_AfDtQxN0MnTcpVsJ6dt-qeO9js0zxW0LQLoNzkQ8k19NwA&oe=65AFF6AE' alt='Ten_Hinh_Anh' width='60%' style='border-radius:50%;margin-bottom:12%;'>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.info('❤🌤️Welcome to project🌤️❤️')
    st.sidebar.markdown("---")
run()
def streamlit_menu():
    # 1. as sidebar menu
    with st.sidebar:
        selected = option_menu(
            menu_title="🏚️Main Menu",  # required
            options=["Overview", "Descriptive statistics","Model", "Contact"],  # required
            icons=["house", "book", "envelope","phone"],  # optional
            menu_icon=None,  # optional
            default_index=0,  # optional
        )
    return selected
@st.cache_resource
def upload_data():
    file_path = '../data/data.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_file_path = os.path.join(current_dir, file_path)
    
    if os.path.exists(full_file_path):
        data = pd.read_csv(full_file_path, encoding='utf-8')
        return data
    else:
        st.warning(f"File '{full_file_path}' không tồn tại.")
        return None
data = upload_data()

def embed_image( file_path,width,height):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_file_path = os.path.join(current_dir, file_path)
    with open(full_file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    html_code = f"""
    <div style="display: flex; justify-content: center;">
        <img src='data:image/jpeg;base64,{encoded_image}' alt='Ten_Hinh_Anh' width='{width}%' height='{height}' style='border-radius:10%; margin-bottom:5%;'>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
def display_file_content( file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_file_path = os.path.join(current_dir, file_path)

    if os.path.exists(full_file_path):
        with open(full_file_path, "r", encoding="utf-8") as file:
            try:
                lines = file.readlines()
                content = "\n".join(lines).strip()
                return content
            except UnicodeDecodeError:
                st.error(
                    f"Tệp tin '{full_file_path}' không thể đọc với encoding utf-8.")
    else:
        st.error(f"Tệp tin '{full_file_path}' không tồn tại.")
def introduction():
    total_rows = data.shape[0]
    total_columns = data.shape[1]
    dimension_data = data.shape
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Total Rows:**\n\n{total_rows} rows")
    with col2:
        st.success(f"**Total Columns:**\n\n{total_columns} columns")

    with col3:
        st.warning(f"**Dimension:**\n\n{dimension_data} ")
    st.subheader("Overview")
    st.dataframe(data.head(),width=13*80)
    file_path = "../asset/image/knn_overview.png"
    embed_image(file_path=file_path,width=60,height='auto')
    st.header("Về thuật toán")
    st.info(display_file_content('../info/KNN_kn.txt'))
    col4,col5 = st.columns(2)
    with col4:
        st.subheader("Nguyên lý hoạt động cơ bản")
        st.info(display_file_content('../info/KNN_nlhdcb.txt'))
        
    with col5:
        st.subheader("Bài toán giải quyết")
        st.info(display_file_content('../info/KNN_vdgq.txt'))
    st.subheader("Giải thích nguyên lý")
    st.info(display_file_content('../info/KNN_nlhdct.txt')) 
    embed_image('../asset/image/nguyenlihoatdong_knn.png',width=40,height='auto')   

def main(selected):
    if selected == "Overview":
        st.header("K-Nearest Neighbors")
        introduction()
    if selected == "Descriptive statistics":
        st.header("Thống kê mô tả")
        descriptive = DESCRIPTIVE_STATISTICS(data)
        st.dataframe(descriptive.describe(),width=13*80)
        st.subheader('Histogram')
        st.plotly_chart(descriptive.subplot_histograms())
        st.subheader("Pairplot")
        st.plotly_chart(descriptive.pairplot())
        st.subheader("Heatmap")
        st.plotly_chart(descriptive.heatmap())
    if selected == "Model":
        st.header("Mô hình chẩn đoán ")
        train_model = TRAIN_MODELS(data)
        train_model.run_knn()
    if selected == "Contact":
        st.header("Liên hệ")
        col_info1,col_info2,col_info3 = st.columns(3)
        with col_info1:
            embed_image('../asset/image/quoc_info.jpg',width=100,height=225)
            st.info("🔥**Hoàng Xuân Quốc - 2156210125**🔥💯")
            st.code("email: 2156210125@hcmussh.edu.vn")
        with col_info2:
            embed_image('../asset/image/chien_info.jpg',width=100,height=225)
            st.info("🔥**Đặng Hoàng Chiến - 2156210095**🔥💯")
            st.code("email: 2156210095@hcmussh.edu.vn")
        with col_info3:
            embed_image('../asset/image/duc_info.png',width=100,height=225)
            st.info("🔥**Nguyễn Viết Đức - 2156210100**🔥💯")
            st.code("email: 2156210100@hcmussh.edu.vn")
        st.info("Created and designed by [Team Data Science - QuocChienDuc](https://github.com/davisduccopny/Diagnosing-cancer-with-the-KNN-algorithm)") 
if __name__ == '__main__':
    selected = streamlit_menu()
    main(selected)
st.sidebar.markdown(  """
            ---
            """)
st.sidebar.info(
            "Created and designed by [Team Data Science - QuocChienDuc](https://github.com/davisduccopny/Stock-Prediction-with-Python-project/)")
