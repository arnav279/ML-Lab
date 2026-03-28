import pandas as pd

# Experiment 5: Student DataFrame and dtypes
data = {
    'Student Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Marks': [85, 90, 78, 92]
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df)
print("\nDatatypes:")
print(df.dtypes)