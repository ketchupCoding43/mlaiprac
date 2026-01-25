import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# -----------------------------
# 1. Prior Parameters
# -----------------------------
alpha_prior = 2
beta_prior = 2

# -----------------------------
# 2. Observed Data
# -----------------------------
heads = 8
tails = 2

# -----------------------------
# 3. Posterior Parameters
# -----------------------------
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

print("Posterior Beta Parameters:", alpha_post, beta_post)

# -----------------------------
# 4. Plot Prior & Posterior
# -----------------------------
theta = np.linspace(0, 1, 100)

prior_dist = beta.pdf(theta, alpha_prior, beta_prior)
posterior_dist = beta.pdf(theta, alpha_post, beta_post)

plt.figure(figsize=(8,5))
plt.plot(theta, prior_dist, label="Prior Distribution")
plt.plot(theta, posterior_dist, label="Posterior Distribution")
plt.title("Bayesian Learning via Inference (Coin Toss)")
plt.xlabel("Probability of Heads")
plt.ylabel("Density")
plt.legend()

# Save graph
plt.savefig("bayesian_inference_coin.png", dpi=300, bbox_inches='tight')
plt.show()
