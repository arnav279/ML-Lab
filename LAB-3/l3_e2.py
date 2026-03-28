import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# a) Load Data
url = "https://raw.githubusercontent.com/yashpalsingh-parihar/Machine-Learning/master/50_Startups.csv"
df = pd.read_csv(url)

# b) Correlation Heatmap
plt.figure(figsize=(10, 8))
# Selecting only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn')
plt.title('Feature Correlation Heatmap')
plt.show()

# c) Outliers using Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_df)
plt.title('Identifying Outliers')
plt.show()

# d) Relationship Visualization
sns.pairplot(df)
plt.show()

# e) Train Model (80:20)
X = numeric_df.drop('Profit', axis=1)
y = numeric_df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# f) Visualizing True vs Predicted
y_pred = model.predict(X_test)
plt.plot(range(len(y_test)), y_test.values, label='Actual Profit', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Profit', marker='x')
plt.legend()
plt.title('Actual vs Predicted Profit')
plt.show()

# g) Metrics
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")