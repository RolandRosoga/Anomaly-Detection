import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv

# Helper: compute leverage scores
def leverage_scores(X):
    X_ = np.hstack([np.ones((X.shape[0], 1)), X])  # add bias
    H = X_ @ inv(X_.T @ X_) @ X_.T
    return np.diag(H)

np.random.seed(0)

# Parameters
mus = [0, 0, 0, 0]
sigmas = [0.1, 1, 3, 5]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
a, b = 2.5, 1.0
x = np.linspace(0, 10, 100)
for i, sigma in enumerate(sigmas):
    noise = np.random.normal(0, sigma, size=x.shape)
    y = a * x + b + noise
    X = x.reshape(-1, 1)
    scores = leverage_scores(X)
    top_idx = np.argsort(scores)[-5:]

    ax = axs[i // 2, i % 2]
    ax.scatter(x, y, c='blue', s=20, label='points')
    ax.scatter(x[top_idx], y[top_idx], c='red', s=50, label='high leverage')
    ax.plot(x, a * x + b, 'k--', label='true model')
    ax.set_title(f'σ² = {sigma**2}')
    ax.legend()

plt.tight_layout()
plt.show()

# --- 2D case ---
np.random.seed(1)
a, b, c = 2, -1, 3
x1 = np.random.uniform(-5, 5, 200)
x2 = np.random.uniform(-5, 5, 200)
eps = np.random.normal(0, 2, 200)
y = a * x1 + b * x2 + c + eps

X = np.column_stack((x1, x2))
scores = leverage_scores(X)
top_idx = np.argsort(scores)[-10:]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y, c='blue', label='points')
ax.scatter(x1[top_idx], x2[top_idx], y[top_idx], c='red', label='high leverage', s=50)
ax.legend()
plt.show()
