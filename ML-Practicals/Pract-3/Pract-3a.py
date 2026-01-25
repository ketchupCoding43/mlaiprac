import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("graphs", exist_ok=True)

data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50],
    "Salary": [25000, 30000, 32000, 35000, 40000, 45000, 48000, 52000]
}

df = pd.DataFrame(data)

X = df[["Age"]]
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope (Coefficient):", slope)
print("Intercept:", intercept)
print(f"Regression Equation: Salary = {slope:.2f} * Age + {intercept:.2f}")

predictions = model.predict(X)
df["Predicted_Salary"] = predictions
print("\nPredictions:\n", df)

new_age = pd.DataFrame([[29]], columns=["Age"])
pred_salary = model.predict(new_age)
print(f"\nPredicted Salary for Age 29: {pred_salary[0]:.2f}")

mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

plt.figure()
plt.scatter(X, y, label="Actual Data")
plt.plot(X, predictions, label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.savefig("graphs/simple_linear_regression.png", bbox_inches="tight")
plt.show()
plt.close()
