import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Single Row", layout="wide")
st.title("SHAP: Explain Single Transaction")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_PATH = os.path.join(ROOT, "data", "processed.csv")
MODEL_PATH = os.path.join(ROOT, "data", "isolation_forest_model.pkl")
RESULTS_DIR = os.path.join(ROOT, "results")
FORCE_PNG = os.path.join(RESULTS_DIR, "shap_force_example.png")

# Check required files
if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSED_PATH):
    st.error("Required files missing (model or processed data).")
    st.stop()

df = pd.read_csv(PROCESSED_PATH)
X = df.drop("Class", axis=1)

# Row selection
st.markdown("### Select a transaction to explain (Row Index)")
row_idx = st.number_input("Row index", min_value=0, max_value=len(X)-1, value=0, step=1)
sample = X.iloc[[row_idx]]

# Prediction
st.markdown("### Model Prediction")
clf = joblib.load(MODEL_PATH)
pred = clf.predict(sample)[0]
st.write(f"Prediction: **{pred}**  (-1 → anomaly, 1 → normal)")

st.markdown("### SHAP Explanation for Selected Row")
with st.spinner("Calculating SHAP values..."):
    try:
        explainer = shap.Explainer(clf, X)
        single_exp = explainer(sample)

        # Show top 10 features contributing to this row
        feature_names = sample.columns
        shap_values = np.abs(single_exp.values).flatten()
        df_local = pd.DataFrame({
            "Feature": feature_names,
            "Absolute SHAP Value": shap_values
        }).sort_values("Absolute SHAP Value", ascending=False).head(10)

        st.markdown("**Top 10 Contributing Features**")
        st.table(df_local)

        # Force plot (matplotlib fallback)
        try:
            shap.plots.force(single_exp, matplotlib=True, show=False)
            fig = plt.gcf()
            st.pyplot(fig)  # explicit figure
            plt.clf()
        except Exception:
            if os.path.exists(FORCE_PNG):
                st.image(FORCE_PNG, use_container_width=True)
            else:
                st.info("Interactive force plot may not render. Use results/shap_force_example.png as fallback.")

    except Exception as e:
        st.error("Error computing SHAP for this row: " + str(e))
        if os.path.exists(FORCE_PNG):
            st.image(FORCE_PNG, use_container_width=True)
