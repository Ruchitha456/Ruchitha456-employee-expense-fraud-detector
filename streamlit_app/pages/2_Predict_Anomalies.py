# streamlit_app/pages/2_Predict_Anomalies.py
import streamlit as st
import pandas as pd
import joblib
import os
from io import StringIO

st.set_page_config(page_title="Predict Anomalies", layout="wide")
st.title("Predict Anomalies")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "Data","demo", "isolation_forest_demo.pkl")
PROCESSED_PATH = os.path.join(ROOT, "Data","demo", "processed_demo.csv")

# Load model
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
else:
    st.error("Model not found. Please run training first.")
    st.stop()

st.markdown("Upload a CSV with the same feature columns as `processed.csv`. Or use a sample from the processed data.")

use_sample = st.checkbox("Use sample from processed.csv", value=True)
if use_sample:
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH).head(1000)  # small preview
        st.write("Preview of sample data:")
        st.dataframe(df.head())
        if st.button("Predict on sample"):
            X = df.drop("Class", axis=1)
            df["anomaly"] = clf.predict(X)
            st.success("Predictions added to the table (anomaly = -1 -> anomaly, 1 -> normal)")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions.csv", data=csv, file_name="predictions_sample.csv")
    else:
        st.warning("processed.csv not found.")
else:
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())
        if st.button("Predict on uploaded file"):
            if "Class" in df.columns:
                X = df.drop("Class", axis=1)
            else:
                X = df.copy()
            df["anomaly"] = clf.predict(X)
            st.success("Predictions added to the table.")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions.csv", data=csv, file_name="predictions.csv")
