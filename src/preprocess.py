"""
src/preprocess.py

Simple preprocessing for the Employee Expense Fraud Detector project.
- Loads CSV file (expects a 'data/raw.csv' path by default)
- Basic cleaning, feature selection, scaling
- Saves processed dataframe to 'data/processed.csv'
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "raw.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.csv")

# Function to load CSV data
def load_data(path=RAW_PATH):
    """Load CSV into pandas DataFrame"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at: {path}. Please download and place it there.")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df

# Function to preprocess data
def preprocess_data(df):
    """Basic preprocessing: remove duplicates, handle missing, scale features"""
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    print(f"[INFO] Removed duplicates: {df.shape[0]} rows remaining")
    
    # 2. Handle missing values (if any)
    missing_cols = df.columns[df.isnull().any()]
    for col in missing_cols:
        df[col].fillna(df[col].median(), inplace=True)
    print(f"[INFO] Handled missing values for columns: {list(missing_cols)}")
    
    # 3. Scale numeric features
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"[INFO] Scaled numeric features: {list(numeric_cols)}")
    
    return df

# Main execution
if __name__ == "__main__":
    # Step 1: Load raw data
    df = load_data()
    
    # Step 2: Preprocess the data
    df_processed = preprocess_data(df)
    
    # Step 3: Save processed data
    df_processed.to_csv(PROCESSED_PATH, index=False)
    print(f"[INFO] Processed data saved at: {PROCESSED_PATH}") 