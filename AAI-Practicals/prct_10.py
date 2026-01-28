import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60],
    "salary": [25000, 32000, 52000, 80000, 61000, 90000, 88000, 100000],
    "purchased": [0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop("purchased", axis=1)
y = df["purchased"]

numeric_features = ["age", "salary"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features)
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression())
])

param_grid = [
    {"model": [LogisticRegression()]},
    {"model": [RandomForestClassifier()]}
]

grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(X, y)

print("Best Model:", type(grid.best_estimator_["model"]).__name__)
