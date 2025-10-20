import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score

x_data, _, y_data, _ = generate_data(
    n_train=1000, n_test=0, n_features=1, contamination=0.1, random_state=42
)
x_data = x_data.ravel()

z = (x_data - np.mean(x_data)) / np.std(x_data)

threshold = np.quantile(np.abs(z), 1 - 0.1)
y_pred = (np.abs(z) > threshold).astype(int)

bal_accu_score = balanced_accuracy_score(y_data, y_pred)
print(f"The threshold is {threshold:.3f}, The balanced accuracy is {bal_accu_score:.3f}")
