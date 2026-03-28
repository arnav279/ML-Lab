import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve

# 1. Load Dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 2. Build and Train Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 3. Predictions
y_pred = nb_model.predict(X_test)
y_probs = nb_model.predict_proba(X_test)[:, 1]

# 4. Evaluation Output
print("--- Experiment 1: Naïve Bayes Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Plotting Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Naïve Bayes Precision-Recall Curve')
plt.grid(True)
plt.show()
plt.savefig('plot_nb.png')