import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Sample Data
# -----------------------------
np.random.seed(0)
X = np.linspace(0, 10, 20)
true_w = 2.5
true_b = 1.0
y = true_w * X + true_b + np.random.normal(0, 2, size=X.shape)

# Add bias column (for intercept)
X_design = np.vstack([np.ones(len(X)), X]).T

# -----------------------------
# 2. Define Prior
# -----------------------------
alpha = 2.0   # Prior precision (controls prior uncertainty)
beta = 1.0    # Noise precision

prior_mean = np.zeros(2)
prior_cov = (1/alpha) * np.eye(2)

# -----------------------------
# 3. Compute Posterior
# -----------------------------
posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + beta * X_design.T @ X_design)
posterior_mean = posterior_cov @ (beta * X_design.T @ y)

print("Prior Mean:", prior_mean)
print("Posterior Mean:", posterior_mean)
print("Posterior Covariance:\n", posterior_cov)

# -----------------------------
# 4. Predictions with Uncertainty
# -----------------------------
X_test = np.linspace(0, 10, 100)
X_test_design = np.vstack([np.ones(len(X_test)), X_test]).T

y_mean = X_test_design @ posterior_mean
y_var = 1/beta + np.sum(X_test_design @ posterior_cov * X_test_design, axis=1)

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Observed Data")
plt.plot(X_test, y_mean, label="Bayesian Regression Line")
plt.fill_between(
    X_test,
    y_mean - 2*np.sqrt(y_var),
    y_mean + 2*np.sqrt(y_var),
    alpha=0.3,
    label="Uncertainty (±2σ)"
)

plt.title("Bayesian Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

# ✅ SAVE GRAPH (important)
plt.savefig("bayesian_linear_regression.png", dpi=300, bbox_inches='tight')

# Show after saving
plt.show()
