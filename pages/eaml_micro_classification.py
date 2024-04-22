
import streamlit as st
import plotly.express as px
import pycaret.classification as pycl
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
    st.set_page_config(page_title="EAML Classification")
    st.title("Elasticized AutoML (Classification) System")
    st.divider()
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Model Analysis","Model Export","Elasticized Auto-Modelling"])
    st.divider()
    st.caption(" Made by :violet[Amirtha Krishnan], :blue[Sachin] & :green[Yekanthavasan] - since September 2023")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
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
        # Preprocess DataFrame
        df_preprocessed = preprocess_data(df)

        # Ensure sufficient samples
        if len(df_preprocessed) > 1:
            # Perform modeling setup
            pycl.setup(df_preprocessed, target=chosen_target,
                       train_size=min(0.7, len(df_preprocessed) - 1))
            setup_df = pycl.pull()
            st.dataframe(setup_df)

            # Compare models
            best_model = pycl.compare_models()
            compare_df = pycl.pull()
            st.dataframe(compare_df)

            # Save the best model
            pycl.save_model(best_model, 'best_classification_model')
        else:
            st.error("Not enough samples in the dataset to perform modeling.")

if choice == "Model Analysis":
    classmodel = pycl.load_model('best_classification_model')
    st.write("Model Information:")
    st.write(classmodel)

    # Example: Display classmodel parameters
    st.write("classmodel Parameters:")
    params = classmodel.get_params()
    st.write(params)

    # Example: Display feature importances (if applicable)
    if hasattr(classmodel, 'feature_importances_'):
        st.write("Feature Importances:")
        feature_importances = classmodel.feature_importances_
        st.write(feature_importances)

if choice == "Model Export":
    with open('best_classification_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_classification_model.pkl")

if choice == "Elasticized Auto-Modelling":
    model = pycl.load_model('best_classification_model')
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
        df_test_processed = preprocess_data(df)
        dx = pycl.predict_model(model, data=df_test_processed)
        st.dataframe(dx)
    else:
        st.warning("No test data uploaded. Retraining the model...")
        # Trigger model training
        df_preprocessed = preprocess_data(df)
        if len(df_preprocessed) > 1:
            pycl.setup(df_preprocessed, target=chosen_target,
                       train_size=min(0.7, len(df_preprocessed) - 1))
            setup_df = pycl.pull()
            st.dataframe(setup_df)

            best_model = pycl.compare_models()
            compare_df = pycl.pull()
            st.dataframe(compare_df)

            pycl.save_model(best_model, 'best_classification_model')
        else:
            st.error("Not enough samples in the dataset to perform modeling.")
