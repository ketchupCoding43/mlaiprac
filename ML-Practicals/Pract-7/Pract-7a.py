import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = load_iris()
X = data.data
y = data.target

model = LogisticRegression(max_iter=200)

# -----------------------------
# 2. K-Fold Cross Validation
# -----------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
kfold_scores = cross_val_score(model, X, y, cv=kfold)

print("K-Fold Scores:", kfold_scores)
print("Mean Accuracy (K-Fold):", np.mean(kfold_scores))

# -----------------------------
# 3. Stratified K-Fold
# -----------------------------
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
skfold_scores = cross_val_score(model, X, y, cv=skfold)

print("\nStratified K-Fold Scores:", skfold_scores)
print("Mean Accuracy (Stratified):", np.mean(skfold_scores))

# -----------------------------
# 4. Leave-One-Out Cross Validation
# -----------------------------
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo)

print("\nLeave-One-Out Mean Accuracy:", np.mean(loo_scores))
