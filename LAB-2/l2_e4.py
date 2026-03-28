import seaborn as sns

# Reloading data for visualization purposes
df = pd.read_csv('Data.csv')

# a) Checking for Outliers in Salary
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Salary'])
plt.title('Box Plot of Salary to Identify Outliers')
plt.show()

# b) Distribution of Age vs Salary
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Salary', hue='Purchased', data=df)
plt.title('Scatter Plot: Age vs Salary')
plt.grid(True)
plt.show()