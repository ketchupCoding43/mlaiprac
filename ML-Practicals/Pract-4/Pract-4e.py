import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

os.makedirs("graphs", exist_ok=True)

np.random.seed(42)
n = 100
Feature1 = np.random.randn(n) * 2 + 5
Feature2 = np.random.randn(n) * 2 + 3
Label = ((Feature1 + Feature2 + np.random.randn(n)) > 8).astype(int)

df = pd.DataFrame({
    "Feature1": Feature1,
    "Feature2": Feature2,
    "Label": Label
})

X = df[["Feature1", "Feature2"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Single Decision Tree
# --------------------------
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree:.3f}")

plt.figure(figsize=(10,6))
plot_tree(dtree, feature_names=X.columns, class_names=["0","1"], filled=True)
plt.title("Single Decision Tree")
plt.savefig("graphs/single_decision_tree.png", bbox_inches="tight")
plt.close()

# --------------------------
# Random Forest
# --------------------------
rf = RandomForestClassifier(n_estimators=50, max_features=2, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.3f}")

# Feature importance plot
plt.figure(figsize=(6,4))
plt.bar(X.columns, rf.feature_importances_)
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.savefig("graphs/rf_feature_importance.png", bbox_inches="tight")
plt.show()
plt.close()

# --------------------------
# Decision Boundary Visualization
# --------------------------
h = 0.02
x_min, x_max = X["Feature1"].min() - 1, X["Feature1"].max() + 1
y_min, y_max = X["Feature2"].min() - 1, X["Feature2"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Use DataFrame with column names to avoid warning
X_grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
Z_rf = rf.predict(X_grid).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z_rf, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X["Feature1"], X["Feature2"], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Random Forest Decision Boundary")
plt.savefig("graphs/rf_decision_boundary.png", bbox_inches="tight")
plt.show()
plt.close()
