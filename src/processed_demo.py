import pandas as pd

# Load your full processed data
df = pd.read_csv("Data/processed.csv")

# Take a sample for demo (e.g., 100 rows)
df_demo = df.sample(100, random_state=42)

# Save as demo CSV
df_demo.to_csv("Data/demo/processed_demo.csv", index=False)
print("Demo CSV created: processed_demo.csv")
