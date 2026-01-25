import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.makedirs("graphs", exist_ok=True)

data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50],
    "Experience": [1, 3, 5, 6, 10, 12, 15, 20],
    "Education_Years": [12, 14, 15, 16, 16, 18, 18, 20],
    "Salary": [25000, 30000, 32000, 35000, 40000, 45000, 48000, 52000]
}

df = pd.DataFrame(data)

X = df[["Age", "Experience", "Education_Years"]]
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nFeature Coefficients:\n", coeff_df)

predictions = model.predict(X)

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("\nMSE:", mse)
print("R-squared:", r2)

corr = df.corr(numeric_only=True)
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("graphs/mlr_correlation_heatmap.png", bbox_inches="tight")
plt.show()
plt.close()

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF Values:\n", vif_data)

plt.figure()
plt.scatter(y, predictions)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.savefig("graphs/mlr_actual_vs_predicted.png", bbox_inches="tight")
plt.show()
plt.close()
