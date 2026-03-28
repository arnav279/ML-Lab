import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 1. Setup Data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# 2. Train both models
nb = GaussianNB().fit(X_train, y_train)
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# --- 1. Bar Chart: Training vs Testing Accuracy ---
plt.figure(figsize=(10, 6))
labels = ['Naïve Bayes', 'Decision Tree']
train_acc = [nb.score(X_train, y_train), dt.score(X_train, y_train)]
test_acc = [nb.score(X_test, y_test), dt.score(X_test, y_test)]

x = np.arange(len(labels))
plt.bar(x - 0.2, train_acc, 0.4, label='Training Accuracy', color='#3498db')
plt.bar(x + 0.2, test_acc, 0.4, label='Testing Accuracy', color='#e74c3c')

plt.xticks(x, labels)
plt.ylabel('Accuracy Score')
plt.title('Training vs Testing Accuracy Comparison')
plt.legend()
plt.savefig('ex3_accuracy_comparison.png')
print("Saved: ex3_accuracy_comparison.png")

# --- 2. ROC Curve Comparison ---
plt.figure(figsize=(10, 6))
# Naive Bayes ROC
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test)[:, 1])
plt.plot(fpr_nb, tpr_nb,
         label=f'Naïve Bayes (AUC = {auc(fpr_nb, tpr_nb):.2f})', color='blue')

# Decision Tree ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])
plt.plot(fpr_dt, tpr_dt,
         label=f'Decision Tree (AUC = {auc(fpr_dt, tpr_dt):.2f})', color='red')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('ex3_roc_comparison.png')
print("Saved: ex3_roc_comparison.png")

# --- 3. Side-by-Side Confusion Matrix Heatmaps ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# NB Heatmap
sns.heatmap(confusion_matrix(y_test, nb.predict(X_test)),
            annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Naïve Bayes Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# DT Heatmap
sns.heatmap(confusion_matrix(y_test, dt.predict(X_test)),
            annot=True, fmt='d', cmap='Reds', ax=ax2)
ax2.set_title('Decision Tree Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('ex3_cm_heatmaps.png')
print("Saved: ex3_cm_heatmaps.png")
