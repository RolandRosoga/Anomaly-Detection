import numpy as np
from sklearn.metrics import balanced_accuracy_score

np.random.seed(51)
n, d = 1000, 3
contamination_rate = 0.1
multi = np.array([2.0, -1.0, 0.5])
sigma = np.array([[2.0, 0.5, 0.3],
                  [0.5, 1.5, 0.2],
                  [0.3, 0.2, 1.0]])

L = np.linalg.cholesky(sigma)
X = (L @ np.random.randn(d, n)).T + multi

#Injectarea Anomaliilor
n_out = int(contamination_rate * n)
out_idx = np.random.choice(n, n_out, replace=False)
y_true = np.zeros(n, dtype=int)
y_true[out_idx] = 1
X[out_idx] += np.random.normal(0, 8, size=(n_out, d))

#Scoruri Z
X_mean = np.mean(X, axis=0)
X_cov = np.cov(X, rowvar=False)
inv_cov = np.linalg.inv(X_cov)
difference = X - X_mean
mahal = np.sqrt(np.sum(difference @ inv_cov * difference, axis=1))

threshold = np.quantile(mahal, 1 - contamination_rate)
y_pred = (mahal > threshold).astype(int)
bal_accu_score = balanced_accuracy_score(y_true, y_pred)

print(f"The threshold is {threshold:.3f}, the balanced accuracy is {bal_accu_score:.3f}")
