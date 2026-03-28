import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve

# 1. Load and split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 2. Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 3. Predict
y_pred = dt_model.predict(X_test)
y_probs = dt_model.predict_proba(X_test)[:, 1]

# 4. Evaluation Output
print("--- Experiment 2: Decision Tree Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Save Decision Tree Visualization
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=data.feature_names, 
          class_names=data.target_names, filled=True, rounded=True)
plt.savefig('ex2_decision_tree.png')
print("Tree visualization saved as ex2_decision_tree.png")

# 6. Save Precision-Recall Curve
plt.figure(figsize=(6, 4))
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision, color='green', lw=2)
plt.title('Decision Tree Precision-Recall Curve')
plt.savefig('ex2_pr_curve.png')
print("PR Curve saved as ex2_pr_curve.png")