import streamlit as st
from io import StringIO
import pandas as pd
import train_model

st.write("""
# AI Cancer Prediction App
""")
 
uploaded_file = st.file_uploader("Upload a file")

dataframe = None
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
if (st.button("Start Training", disabled=(dataframe is None))):
    st.write("RMSE = " + str(train_model.train(train_model.loadData(dataframe))))