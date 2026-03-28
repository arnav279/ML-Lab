import pandas as pd

# Experiment 7: info(), shape, and missing values
df = pd.read_csv('your_file.csv')

print("Dataset Info:")
df.info()

print(f"\nNumber of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

print("\nMising Values in Each Column:")
print(df.isnull().sum())