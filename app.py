import streamlit as st
from streamlit import session_state
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import datetime
import os

global select_df, filename, todaydate, current_time,df

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

def upload_data():
    st.title("Dataset Viewer")
    st.info("Currently supports datasets up to 2GB")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        filename = file.name
        todaydate = datetime.date.today().strftime("%d%m%y")
        current_time = datetime.datetime.now().strftime("%H_%M")
        df.to_csv(f"{filename[:8]}_{todaydate}at{current_time}.csv", index=None)
        print(f"Data uploaded from {file}")
        st.dataframe(df)

def profiling():
    file = st.file_uploader("Upload Your Dataset to be Profiled")
    if file:
        df = pd.read_csv(file, index_col=None)
    else:
        st.info("Select the dataset to be profiled")
    if df is not None:
        profile_df = ProfileReport(df)
        st_profile_report(profile_df)
        export = profile_df.to_html()
        todaydate = datetime.date.today().strftime("%d_%m_%y")
        current_time = datetime.datetime.now().strftime("%H_%M")
        st.download_button(label="Export Report (HTML)", data=export, file_name=f"REPORT_{todaydate}at{current_time}.html")
    else:
        st.error("Please upload a dataset first.")

def modelling():
    file = st.file_uploader("Upload Your Dataset to be Profiled")
    if file:
        df = pd.read_csv(file, index_col=None)
    if df is not None:
        st.title("Model Generation")
        st.info("Generates ML models with user-selected features.")
        chosen_target = st.selectbox('Choose Target Column', df.columns)
        numeric_features = df.select_dtypes(include=['number']).columns
        df_numeric = df[numeric_features]

        if st.button('Run Modelling'):
            typed_df = df_numeric.dtypes.astype(str)
            setup(df_numeric, target=chosen_target, verbose=False, use_gpu=True)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
    else:
        st.error("Please upload a dataset and select a target column first.")

def export_model():
    if os.path.exists('best_model.pkl'):
        st.title("Export Generated Model")
        st.info("Ensure you don't overwrite existing models.")
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Save Model', f, file_name="best_model.pkl")
    else:
        st.title("Export Generated Model")
        st.info("Please see to it that generated models are not overwritten!")
        st.error("No model saved yet. Please train a model first.")

with st.sidebar:
    st.set_page_config(page_title="EAML")
    st.title("Elasticized AutoML System")
    st.divider()
    choice = st.radio("Sections", ["Dataset Viewer", "Profiling", "Modelling", "Export Model"])
    st.info("Kindly upload your CSV file and then proceed to 'Profiling' for analysis.")
    st.divider()
    st.caption(" Made by :violet[Amirtha Krishnan], :blue[Sachin] & :green[Yekanthavasan] - since 2023")

if choice == "Dataset Viewer":
    upload_data()
if choice == "Profiling":
    profiling()
if choice == "Modelling":
    modelling()
if choice == "Export Model":
    export_model()
