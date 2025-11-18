"""
src/train.py

Train an Isolation Forest model on the processed Employee Expense dataset.
- Loads processed data
- Splits data into features (X) and target (y)
- Trains Isolation Forest for anomaly detection
- Saves the trained model for later use
"""

import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.csv")
MODEL_PATH = os.path.join(DATA_DIR, "isolation_forest_model.pkl")

# Load processed data
df = pd.read_csv(PROCESSED_PATH)
print(f"[INFO] Loaded processed data: {df.shape[0]} rows, {df.shape[1]} cols")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train Isolation Forest
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
clf.fit(X)
print(f"[INFO] Isolation Forest model trained.")

# Save the model
joblib.dump(clf, MODEL_PATH)
print(f"[INFO] Isolation Forest model saved at: {MODEL_PATH}")
