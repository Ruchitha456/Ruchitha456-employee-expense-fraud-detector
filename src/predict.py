"""
src/predict.py

Use the trained Isolation Forest model to predict anomalies.
- Loads processed data
- Loads trained model
- Predicts anomalies
- Saves results to 'data/predictions.csv'
"""

import os
import pandas as pd
import joblib

# ------------------------------
# File Paths
# ------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.csv")
MODEL_PATH = os.path.join(DATA_DIR, "isolation_forest_model.pkl")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv(PROCESSED_PATH)
print(f"[INFO] Loaded processed data: {df.shape[0]} rows, {df.shape[1]} cols")

# ------------------------------
# Load Model
# ------------------------------
clf = joblib.load(MODEL_PATH)
print(f"[INFO] Loaded Isolation Forest model")

# ------------------------------
# Predict anomalies
# ------------------------------
X = df.drop("Class", axis=1)   # features only
df["anomaly"] = clf.predict(X)  # 1 = normal, -1 = anomaly

# ------------------------------
# Save predictions
# ------------------------------
df.to_csv(PREDICTIONS_PATH, index=False)
print(f"[INFO] Predictions saved at: {PREDICTIONS_PATH}")
