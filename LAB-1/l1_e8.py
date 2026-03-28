import pandas as pd

# Experiment 8: describe() for statistical summary
df = pd.read_csv('your_file.csv')

print("Statistical Summary:")
print(df.describe())