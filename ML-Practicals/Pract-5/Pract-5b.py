import numpy as np
from hmmlearn import hmm

# -----------------------------
# 1. Create Sample Observation Data
# -----------------------------
# Example: 1D continuous observations (like stock prices or sensor data)
observations = np.array([[1.0], [2.1], [1.9], [5.0], [6.1], [5.9]])

# -----------------------------
# 2. Build Gaussian HMM Model
# -----------------------------
model = hmm.GaussianHMM(
    n_components=2,      # number of hidden states
    covariance_type="diag",
    n_iter=100
)

# -----------------------------
# 3. Train the Model
# -----------------------------
model.fit(observations)

# -----------------------------
# 4. Predict Hidden States
# -----------------------------
hidden_states = model.predict(observations)

# -----------------------------
# 5. Output Model Parameters
# -----------------------------
print("Hidden States Sequence:")
print(hidden_states)

print("\nStart Probability:")
print(model.startprob_)

print("\nTransition Matrix:")
print(model.transmat_)

print("\nMeans of states:")
print(model.means_)

print("\nCovariances of states:")
print(model.covars_)
