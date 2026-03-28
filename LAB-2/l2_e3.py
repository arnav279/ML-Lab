from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# a) Splitting the dataset into Training and Test set (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

# b) Feature Scaling
sc = StandardScaler()
# Scaling only the numerical features (Age and Salary, which are now at the end of the array)
X_train[:, -2:] = sc.fit_transform(X_train[:, -2:])
X_test[:, -2:] = sc.transform(X_test[:, -2:])

print("\nScaled X_train (Numerical Columns):\n", X_train)