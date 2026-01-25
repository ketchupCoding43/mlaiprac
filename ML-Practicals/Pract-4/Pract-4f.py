import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

os.makedirs("graphs", exist_ok=True)

# ----------------------------
# 1. Generate Example Dataset
# ----------------------------
np.random.seed(42)
n = 200
Feature1 = np.random.randn(n) * 2 + 5
Feature2 = np.random.randn(n) * 2 + 3
Feature3 = np.random.randn(n) * 1.5 + 2
Label = ((Feature1 + Feature2 + Feature3 + np.random.randn(n)) > 9).astype(int)

df = pd.DataFrame({
    "Feature1": Feature1,
    "Feature2": Feature2,
    "Feature3": Feature3,
    "Label": Label
})

X = df[["Feature1", "Feature2", "Feature3"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 2. Train XGBoost Classifier
# ----------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------------
# 3. Feature Importance
# ----------------------------
plt.figure(figsize=(6, 4))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.savefig("graphs/xgb_feature_importance.png", bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------
# 4. Decision Boundary (Optional for 2 features only)
# ----------------------------
if X.shape[1] == 2:
    h = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    X_grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = xgb_model.predict(X_grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.title("XGBoost Decision Boundary")
    plt.savefig("graphs/xgb_decision_boundary.png", bbox_inches="tight")
    plt.show()
    plt.close()
