from skopt import BayesSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

search = BayesSearchCV(
    LogisticRegression(max_iter=3000),
    {
        "C": (1e-6, 1e+6, "log-uniform")
    },
    n_iter=10,
    cv=3
)

search.fit(X, y)

print("Best C:", search.best_params_)
