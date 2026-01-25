import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create Real Dataset (2D Gaussian)
# -----------------------------
real_data = torch.randn(500, 2) * 1.5 + torch.tensor([4.0, 4.0])

# -----------------------------
# 2. Generator Network
# -----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, z):
        return self.model(z)

# -----------------------------
# 3. Discriminator Network
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.001)
d_optimizer = optim.Adam(D.parameters(), lr=0.001)

# -----------------------------
# 4. Training Loop
# -----------------------------
epochs = 1000
batch_size = 64

for epoch in range(epochs):
    idx = np.random.randint(0, real_data.size(0), batch_size)
    real_batch = real_data[idx]

    # Real labels = 1, Fake labels = 0
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # ---- Train Discriminator ----
    z = torch.randn(batch_size, 2)
    fake_data = G(z)

    d_loss_real = criterion(D(real_batch), real_labels)
    d_loss_fake = criterion(D(fake_data.detach()), fake_labels)
    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # ---- Train Generator ----
    z = torch.randn(batch_size, 2)
    fake_data = G(z)
    g_loss = criterion(D(fake_data), real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# -----------------------------
# 5. Generate Samples
# -----------------------------
z = torch.randn(500, 2)
generated_data = G(z).detach().numpy()

# -----------------------------
# 6. Plot Results
# -----------------------------
plt.scatter(real_data[:, 0], real_data[:, 1], label="Real Data", alpha=0.5)
plt.scatter(generated_data[:, 0], generated_data[:, 1], label="Generated Data", alpha=0.5)
plt.legend()
plt.title("GAN Learning to Generate 2D Data")

plt.savefig("gan_output.png", dpi=300, bbox_inches='tight')
plt.show()
