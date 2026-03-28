import numpy as np

# Experiment 4: Min, Max, Mean, and Sum
# Values chosen to match your ODT output image
data = np.array([18, 80, 5, 52, 38, 61, 76, 37, 38, 93, 47, 23, 65, 77, 56, 40, 20, 26, 50, 72])

print(f"Array: {data}")
print(f"Minimum: {data.min()}")
print(f"Maximum: {data.max()}")
print(f"Mean: {data.mean()}")
print(f"Sum: {data.sum()}")