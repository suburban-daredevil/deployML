'''
StreamLit based Python UI creation application
StreamLit is a easily codable and deployable Python framework to code and deploy ML models to the internet
'''

import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Classifying Iris Flowers')
st.markdown('Web application to classify the iris flowers')

st.header('Plant Features')

col1, col2 = st.columns(2)
with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text('Petal characteristics')
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

if st.button('Predict', key = 1):# key here is just a unique value to identify multiple buttons in streamlit
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])