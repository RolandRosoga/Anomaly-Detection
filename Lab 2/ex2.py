from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1
)

neighbors_list = [3, 5, 10, 20]

fig, axs = plt.subplots(len(neighbors_list), 4, figsize=(16, 12))
for i, n in enumerate(neighbors_list):
    clf = KNN(n_neighbors=n)
    clf.fit(X_train)

    y_train_pred = clf.labels_
    y_test_pred = clf.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)

    print(f"n_neighbors={n} -> Train BA={ba_train:.3f}, Test BA={ba_test:.3f}")

    axs[i, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
    axs[i, 0].set_title(f"Train GT (n={n})")

    axs[i, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap='coolwarm')
    axs[i, 1].set_title("Train Pred")

    axs[i, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
    axs[i, 2].set_title("Test GT")

    axs[i, 3].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='coolwarm')
    axs[i, 3].set_title("Test Pred")

plt.tight_layout()
plt.show()
