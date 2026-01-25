import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import uniform

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Define Model
# -----------------------------
model = SVC()

# -----------------------------
# 3. Grid Search
# -----------------------------
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters (Grid Search):", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# -----------------------------
# 4. Randomized Search
# -----------------------------
param_dist = {
    'C': uniform(0.1, 100),
    'gamma': uniform(0.001, 1),
    'kernel': ['rbf', 'linear']
}

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nBest Parameters (Random Search):", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

# -----------------------------
# 5. Test Accuracy Comparison
# -----------------------------
best_grid_model = grid_search.best_estimator_
best_random_model = random_search.best_estimator_

print("\nTest Accuracy (Grid Search Model):", best_grid_model.score(X_test, y_test))
print("Test Accuracy (Random Search Model):", best_random_model.score(X_test, y_test))
