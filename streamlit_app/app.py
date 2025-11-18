import streamlit as st

st.set_page_config(page_title="Employee Expense Fraud Detector", layout="wide")
st.title("Employee Expense Fraud Detector")

st.markdown("""
This project demonstrates an **end-to-end anomaly detection system**:

- **Model:** Isolation Forest  
- **Explainability:** SHAP (feature-level and local explanations)  
- **UI:** Streamlit multi-page app  
- **Outputs:** Predictions + SHAP visuals  

Use the left sidebar (Pages) to navigate through the app:
- **Model Overview**  
- **Predict Anomalies** (upload CSV or use sample)  
- **SHAP: Global Insights**  
- **SHAP: Explain Single Row**  

""")

st.markdown("---")

st.markdown("### Quick Info / Highlights")
st.markdown("""
- **Global SHAP Insights:** Shows overall feature importance across all transactions.  
- **Explain Single Transaction:** Explore which features contributed to the model prediction for a single transaction.  
- **Deployment Ready:** The project can be deployed to **GitHub** and **Streamlit Cloud** for a live demo.
""")

st.markdown("---")

st.markdown("**Tip:** Use the *Predict Anomalies* page to upload a CSV (same schema as `processed.csv`) and get anomaly flags with SHAP explanations.")
