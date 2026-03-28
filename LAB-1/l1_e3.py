import numpy as np

# Experiment 3: Create 1-12 array and reshape to 3x4
print("Original Array:")
original = np.arange(1, 13)
print(original)
print(f"Shape: {original.shape}\n")

print("Reshaped Array (3x4):")
reshaped = original.reshape(3, 4)
print(reshaped)
print(f"Shape: {reshaped.shape}")