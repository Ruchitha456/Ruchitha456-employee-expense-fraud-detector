"""
src/shap_explain.py

SHAP explanation for Isolation Forest (learning/sample-first version).

What this script does (fast, safe version):
- Loads processed data and trained IsolationForest model
- Takes a random sample (default 2000 rows) for fast SHAP computation
- Uses SHAP TreeExplainer (fast & appropriate for tree ensembles)
- Computes SHAP values and produces:
    - summary plot (global feature importance)
    - bar plot of mean(|SHAP|)
    - force plot for one example (saved as HTML)
- Saves a CSV with mean absolute SHAP per feature
- Saves plots under 'results/shap_*'

How to run:
    python src/shap_explain.py
"""

import os
import random
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Configuration (change if needed)
# ------------------------------
SAMPLE_SIZE = 2000         # number of rows to use for SHAP (keep small for learning)
RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.csv")
MODEL_PATH = os.path.join(DATA_DIR, "isolation_forest_model.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SUMMARY_PLOT_PATH = os.path.join(RESULTS_DIR, "shap_summary.png")
BAR_PLOT_PATH = os.path.join(RESULTS_DIR, "shap_bar.png")
FORCE_HTML_PATH = os.path.join(RESULTS_DIR, "shap_force_example.html")
SHAP_CSV_PATH = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------
# Load data & model
# ------------------------------
print("[INFO] Loading processed data...")
df = pd.read_csv(PROCESSED_PATH)
print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

print("[INFO] Loading trained model...")
clf = joblib.load(MODEL_PATH)
print("[INFO] Model loaded.")

# ------------------------------
# Prepare sample for SHAP
# ------------------------------
# We drop the target 'Class' before explaining features
if "Class" in df.columns:
    X_all = df.drop("Class", axis=1)
else:
    X_all = df.copy()

# If dataset is small enough, we could use entire X_all; for learning we sample.
n = min(SAMPLE_SIZE, X_all.shape[0])
print(f"[INFO] Sampling {n} rows from the processed data for SHAP (seed={RANDOM_STATE})")
X_sample = X_all.sample(n=n, random_state=RANDOM_STATE)

# Convert to numpy if needed (SHAP accepts pandas too)
# Keep feature names for plots
feature_names = X_sample.columns.tolist()

# ------------------------------
# Create SHAP explainer (TreeExplainer)
# ------------------------------
print("[INFO] Creating TreeExplainer (fast for tree-based models)...")
# Use shap.Explainer -- it will pick the appropriate internal explainer (TreeExplainer for sklearn ensembles)
explainer = shap.Explainer(clf, X_sample, feature_names=feature_names)

print("[INFO] Computing SHAP values (this may take a short while)...")
shap_values = explainer(X_sample)   # returns an Explanation object

# ------------------------------
# Global summary plot (beeswarm)
# ------------------------------
print("[INFO] Creating summary plot (global view)...")
plt.figure(figsize=(10, 6))
# shap.summary_plot accepts the .values in an Explanation object
shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(SUMMARY_PLOT_PATH, dpi=150)
plt.close()
print(f"[INFO] Saved SHAP summary plot to: {SUMMARY_PLOT_PATH}")

# ------------------------------
# Bar plot of mean(|SHAP|)
# ------------------------------
print("[INFO] Creating bar plot of mean(|SHAP|) per feature...")
# shap_values.values is (n_samples, n_features)
mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
fi = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

# Save CSV of feature importance
fi.to_csv(SHAP_CSV_PATH, index=False)
print(f"[INFO] Saved SHAP feature importances to: {SHAP_CSV_PATH}")

# Bar plot
plt.figure(figsize=(10, 6))
plt.barh(fi["feature"].iloc[::-1], fi["mean_abs_shap"].iloc[::-1])
plt.xlabel("mean(|SHAP value|)")
plt.title("Feature importance by mean absolute SHAP")
plt.tight_layout()
plt.savefig(BAR_PLOT_PATH, dpi=150)
plt.close()
print(f"[INFO] Saved SHAP bar plot to: {BAR_PLOT_PATH}")

# ------------------------------
# Force plot for a single sample (interactive)
# ------------------------------
# Save an HTML of a single instance explanation (pick first row of the sample)
print("[INFO] Creating force plot for one example and saving as HTML...")
idx = 0
single_X = X_sample.iloc[[idx]]
single_exp = explainer(single_X)   # compute explanation for single row
# shap.plots.force(single_exp)  # interactive in notebook, but we will save HTML
html = shap.plots.force(single_exp, matplotlib=False, show=False)
# shap.plots.force returns an IPython object; to save we leverage the explanation object .html() if available
try:
    # If Explanation has an HTML representation, use it
    force_html = single_exp.html()
    with open(FORCE_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(force_html)
    print(f"[INFO] Saved SHAP force plot HTML to: {FORCE_HTML_PATH}")
except Exception:
    # fallback: create a static image of the force plot (best-effort)
    try:
        shap.plots.force(single_exp, matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "shap_force_example.png"), dpi=150)
        plt.close()
        print("[INFO] Saved fallback SHAP force plot PNG to: results/shap_force_example.png")
    except Exception as e:
        print("[WARN] Could not save force plot interactively:", str(e))

print("[INFO] SHAP explanation (sample) completed.")
