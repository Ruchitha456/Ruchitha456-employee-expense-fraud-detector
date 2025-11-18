from sklearn.ensemble import IsolationForest
import joblib
import pandas as pd

# Load demo CSV
df_demo = pd.read_csv("Data/demo/processed_demo.csv")

# Drop the target column if exists
if "Class" in df_demo.columns:
    X_demo = df_demo.drop("Class", axis=1)
else:
    X_demo = df_demo.copy()

# Train a smaller Isolation Forest for demo
clf_demo = IsolationForest(
    n_estimators=50,  # smaller for demo
    contamination=0.01,
    random_state=42
)
clf_demo.fit(X_demo)

# Save the demo model
joblib.dump(clf_demo, "Data/demo/isolation_forest_demo.pkl")
print("Demo model saved: isolation_forest_demo.pkl")
