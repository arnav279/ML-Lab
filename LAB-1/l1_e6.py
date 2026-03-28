import pandas as pd

# Experiment 6: read_csv(), head(), and tail()
# Replace 'data.csv' with your actual filename
df = pd.read_csv('your_file.csv') 

print("First 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())