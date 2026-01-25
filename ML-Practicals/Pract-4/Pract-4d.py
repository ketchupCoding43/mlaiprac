import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("graphs", exist_ok=True)

# Generate example dataset
np.random.seed(42)
n = 100
Feature1 = np.random.randn(n) * 2 + 5
Feature2 = np.random.randn(n) * 2 + 3
Label = (Feature1 + Feature2 + np.random.randn(n)) > 8
Label = Label.astype(int)

df = pd.DataFrame({
    "Feature1": Feature1,
    "Feature2": Feature2,
    "Label": Label
})

X = df[["Feature1", "Feature2"]]
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit SVM with RBF kernel
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize decision boundary (only for 2D features)
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Feature1 (scaled)")
plt.ylabel("Feature2 (scaled)")
plt.title("SVM Decision Boundary")
plt.savefig("graphs/svm_decision_boundary.png", bbox_inches="tight")
plt.show()
plt.close()
