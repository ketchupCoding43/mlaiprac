import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

os.makedirs("graphs", exist_ok=True)

np.random.seed(42)
n = 50
Age = np.random.randint(20, 50, n)
Experience = np.random.randint(1, 25, n)
Education_Years = np.random.randint(12, 22, n)
Salary = 20000 + (Age * 1000) + (Experience * 800) + (Education_Years * 500) + np.random.normal(0, 2000, n)

df = pd.DataFrame({
    "Age": Age,
    "Experience": Experience,
    "Education_Years": Education_Years,
    "Salary": Salary
})

X = df[["Age", "Experience", "Education_Years"]]
y = df["Salary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

models = {"Linear": lr, "Ridge": ridge, "Lasso": lasso, "ElasticNet": elastic}

results = {}
for name, model in models.items():
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    print(f"\n{name} Regression:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("MSE:", mse)
    print("R²:", r2)

plt.figure(figsize=(8,6))
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    plt.scatter(y, y_pred, label=name)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary (Regularized Models)")
plt.legend()
plt.savefig("graphs/regularized_regression_comparison.png", bbox_inches="tight")
plt.show()
plt.close()
