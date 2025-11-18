import streamlit as st
import os
from PIL import Image
import pandas as pd

st.set_page_config(page_title="SHAP Global Insights", layout="wide")
st.title("SHAP: Global Insights")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
SHAP_SUM_PNG = os.path.join(RESULTS_DIR, "shap_summary.png")
SHAP_BAR_PNG = os.path.join(RESULTS_DIR, "shap_bar.png")
SHAP_CSV = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")

st.markdown("### Global SHAP explanations")

# Load and resize images
def load_and_resize(path, width=550, height=400):
    if os.path.exists(path):
        img = Image.open(path)
        img = img.resize((width, height))
        return img
    return None

col1, col2 = st.columns(2)

with col1:
    img1 = load_and_resize(SHAP_SUM_PNG)
    if img1:
        st.image(img1, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.warning("shap_summary.png not found.")

with col2:
    img2 = load_and_resize(SHAP_BAR_PNG)
    if img2:
        st.image(img2, caption="SHAP Feature Importance Bar Plot", use_container_width=True)
    else:
        st.warning("shap_bar.png not found.")

st.write("---")

# Show top 20 features in a clean table
if os.path.exists(SHAP_CSV):
    df_fi = pd.read_csv(SHAP_CSV)
    df_top = df_fi.sort_values("mean_abs_shap", ascending=False).head(20)
    st.markdown("### Top 20 Features by Mean(|SHAP|)")
    st.table(df_top)
else:
    st.info("shap_feature_importance.csv not found.")
