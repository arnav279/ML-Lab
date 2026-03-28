from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# a) Encoding the Independent Variable (Country column)
# This creates dummy variables for the countries
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("Encoded Independent Variables (X):\n", X)

# b) Encoding the Dependent Variable (Purchased column)
# Converts 'Yes'/'No' to 1/0
le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoded Dependent Variable (y):\n", y)