import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model, predict_model
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os 

def preprocess_data(df):
    df = df.dropna()
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
        
    return df
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.set_page_config(page_title="EAML Regression")
    st.title("Elasticized AutoML (Regression) System")
    st.divider()
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Model Analysis and Export","Elasticized Auto-Modelling"])
    st.divider()
    st.caption(" Made by :violet[Amirtha Krishnan], :blue[Sachin] & :green[Yekanthavasan] - since September 2023")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = df.to_csv()
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        df_preprocessed = preprocess_data(df)
        
        if len(df_preprocessed) > 1:
            setup(df_preprocessed, target=chosen_target, train_size=min(0.7, len(df_preprocessed) - 1), use_gpu = True)
            setup_df = pull()
            st.dataframe(setup_df)

            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
                
            save_model(best_model, 'best_regression_model')
        else:
            st.error("Not enough samples in the dataset to perform modeling.")
        

if choice == "Model Analysis and Export":
    model = load_model('best_regression_model')
    st.write("Model Information:")
    st.write(model)

    # model parameters
    st.write("Model Parameters:")
    params = model.get_params()
    st.write(params)

    # feature importances (if applicable)
    if hasattr(model, 'feature_importances_'):
        st.write("Feature Importances:")
        feature_importances = model.feature_importances_
        st.write(feature_importances)

    with open('best_regression_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_regression_model.pkl")

if choice == "Elasticized Auto-Modelling":
    model = load_model('best_regression_model')
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
        df_test_processed = preprocess_data(df)
        dx=predict_model(model, data=df_test_processed)
        st.dataframe(dx)
    else:
        st.warning("No changes detected in test data. Selecting a column will retraining the model with existing trained data...")
        # Trigger model training
        df_preprocessed = preprocess_data(df)
        if len(df_preprocessed) > 1:
            setup(df_preprocessed, target=chosen_target, train_size=min(0.7, len(df_preprocessed) - 1), use_gpu=True)
            setup_df = pull()
            st.dataframe(setup_df)
                
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
                
            save_model(best_model, 'best_regression_model')
        else:
            st.error("Not enough samples in the dataset to perform modelling.")
