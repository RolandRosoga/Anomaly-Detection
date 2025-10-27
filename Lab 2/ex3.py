from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import numpy as np

# Two clusters, different densities
X, y_true = make_blobs(
    n_samples=[200, 100],
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    n_features=2,
    random_state=42,
)

contamination = 0.07
neighbors_list = [5, 15]

fig, axs = plt.subplots(len(neighbors_list), 2, figsize=(10, 8))
for i, n in enumerate(neighbors_list):
    knn = KNN(contamination=contamination, n_neighbors=n)
    lof = LOF(contamination=contamination, n_neighbors=n)

    knn.fit(X)
    lof.fit(X)

    y_knn = knn.labels_
    y_lof = lof.labels_

    axs[i, 0].scatter(X[:, 0], X[:, 1], c=y_knn, cmap='coolwarm', s=20)
    axs[i, 0].set_title(f"KNN (n={n})")

    axs[i, 1].scatter(X[:, 0], X[:, 1], c=y_lof, cmap='coolwarm', s=20)
    axs[i, 1].set_title(f"LOF (n={n})")

plt.tight_layout()
plt.show()
