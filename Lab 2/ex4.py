import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.combination import average, maximization
from sklearn.metrics import balanced_accuracy_score

# --- Load cardio dataset from ODDS ---
data = loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

contamination = np.mean(y == 1)  # actual rate

n_neighbors_list = range(30, 121, 10)
train_scores, test_scores = [], []

print("Training ensemble models...\n")
for n in n_neighbors_list:
    clf = KNN(n_neighbors=n, contamination=contamination)
    clf.fit(X_train_norm)

    y_train_pred = clf.labels_
    y_test_pred = clf.predict(X_test_norm)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"n_neighbors={n} -> Train BA={ba_train:.3f}, Test BA={ba_test:.3f}")

    train_scores.append(clf.decision_scores_)
    test_scores.append(clf.decision_function(X_test_norm))

train_scores = np.array(train_scores).T
test_scores = np.array(test_scores).T

train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

# --- Average combination ---
avg_train, avg_test = average(train_scores_norm), average(test_scores_norm)
th_avg = np.quantile(avg_train, 1 - contamination)
y_test_pred_avg = (avg_test > th_avg).astype(int)
ba_avg = balanced_accuracy_score(y_test, y_test_pred_avg)

# --- Max combination ---
max_train, max_test = maximization(train_scores_norm), maximization(test_scores_norm)
th_max = np.quantile(max_train, 1 - contamination)
y_test_pred_max = (max_test > th_max).astype(int)
ba_max = balanced_accuracy_score(y_test, y_test_pred_max)

print("\n--- Ensemble Results ---")
print(f"Average strategy BA: {ba_avg:.3f}")
print(f"Maximization strategy BA: {ba_max:.3f}")
