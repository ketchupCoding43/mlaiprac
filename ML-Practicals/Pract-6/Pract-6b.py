import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# -----------------------------
# 1. Generate Sample Data
# -----------------------------
np.random.seed(0)

cluster1 = np.random.normal(loc=[2, 2], scale=0.8, size=(100, 2))
cluster2 = np.random.normal(loc=[7, 7], scale=1.0, size=(100, 2))
X = np.vstack((cluster1, cluster2))

# -----------------------------
# 2. Fit Gaussian Mixture Model
# -----------------------------
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(X)

# -----------------------------
# 3. Predict Clusters
# -----------------------------
labels = gmm.predict(X)

# -----------------------------
# 4. Density Estimation Grid
# -----------------------------
x = np.linspace(-2, 12, 200)
y = np.linspace(-2, 12, 200)
X_grid, Y_grid = np.meshgrid(x, y)
grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

Z = np.exp(gmm.score_samples(grid))
Z = Z.reshape(X_grid.shape)

# -----------------------------
# 5. Plot Results
# -----------------------------
plt.figure(figsize=(8,6))

# Density contour
plt.contour(X_grid, Y_grid, Z)

# Clustered points
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)

# Means of Gaussians
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
            marker='X', s=200, label='Centers')

plt.title("Gaussian Mixture Model - Clustering & Density Estimation")
plt.legend()

# ✅ Save graph
plt.savefig("gmm_clustering_density.png", dpi=300, bbox_inches='tight')

plt.show()

# -----------------------------
# 6. Print Model Parameters
# -----------------------------
print("Means:\n", gmm.means_)
print("\nCovariances:\n", gmm.covariances_)
print("\nWeights:\n", gmm.weights_)
