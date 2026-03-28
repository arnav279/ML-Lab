import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# a) Load Data
url = "https://raw.githubusercontent.com/yashpalsingh-parihar/Machine-Learning/master/Social_Network_Ads.csv"
df = pd.read_csv(url)

# b) Preprocessing (Dropping User ID)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# c) Correlation Heatmap
sns.heatmap(X.corr(), annot=True)
plt.title('Feature Correlation')
plt.show()

# d) Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling is mandatory for Logistic Regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# f) Classification Report
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# g) Confusion Matrix & ROC Curve
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# h) Visualizing Actual vs Predicted
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', marker='x')
plt.title('Actual vs Predicted Classes')
plt.legend()
plt.show()