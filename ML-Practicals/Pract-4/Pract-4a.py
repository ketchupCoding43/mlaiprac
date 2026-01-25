import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

os.makedirs("graphs", exist_ok=True)

# Generate synthetic binary classification dataset
np.random.seed(42)
n = 100
Age = np.random.randint(20, 60, n)
Experience = np.random.randint(1, 40, n)
Education_Years = np.random.randint(12, 22, n)

# Target: 1 if Age + Experience + random factor > threshold else 0
Y = ((Age + Experience + np.random.randint(0,10,n)) > 70).astype(int)

df = pd.DataFrame({
    "Age": Age,
    "Experience": Experience,
    "Education_Years": Education_Years,
    "Promoted": Y
})

X = df[["Age", "Experience", "Education_Years"]]
y = df["Promoted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.savefig("graphs/logistic_regression_roc.png", bbox_inches="tight")
plt.show()
plt.close()
