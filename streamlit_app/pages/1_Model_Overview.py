import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import joblib

st.set_page_config(page_title="Model Overview", layout="wide")
st.title("Model Overview")

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT,"Data" ,"demo", "processed_demo.csv")
MODEL_PATH = os.path.join(ROOT, "Data","demo", "isolation_forest_demo.pkl")

# Dataset
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.markdown("### Processed Dataset Preview (first 5 rows)")
    st.dataframe(df.head())
    st.markdown(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
else:
    st.warning("Processed dataset not found. Please run preprocessing first.")

st.write("---")

# Model status
if os.path.exists(MODEL_PATH):
    st.success("✅ Trained Isolation Forest model is ready for predictions.")
    # Load model details
    clf = joblib.load(MODEL_PATH)
    st.markdown("### Model Details")
    st.write(clf)
else:
    st.warning("⚠️ Isolation Forest model file not found. Please train the model first.")
