import numpy as np
import matplotlib.pyplot as plot
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA

data = loadmat("shuttle.mat")
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.6,
                                                    random_state=42,
                                                    stratify=y)

scale = StandardScaler()
X_trainScaled = scale.fit_transform(X_train)
X_testScaled = scale.transform(X_test)

cont_rate = np.mean(y_train == 1)

pca = PCA(contamination=cont_rate, random_state=42)
pca.fit(X_trainScaled)

exp_var = pca.explained_variance_
cum_explained_var = np.cumsum(exp_var) / np.sum(exp_var)


plot.figure()
plot.bar(range(1, len(exp_var)+1), exp_var)
plot.step(range(1, len(exp_var)+1), cum_explained_var, where='mid', color='red')
plot.title("PCA variances explained")
plot.xlabel("Comp"); plot.ylabel("Var")
plot.show()

y_train_pred_pca = pca.labels_
train_score_pca = pca.decision_scores_
y_test_pred_pca = pca.predict(X_testScaled)
test_score_pca = pca.decision_function(X_testScaled)

ba_train_pca = balanced_accuracy_score(y_train, y_train_pred_pca)
ba_test_pca = balanced_accuracy_score(y_test, y_test_pred_pca)

print("PCA BA train:", ba_train_pca)
print("PCA BA test:", ba_test_pca)

kpca = KPCA(contamination=cont_rate, kernel="rbf", random_state=42)
kpca.fit(X_trainScaled)

y_train_pred_KPCA = kpca.labels_
y_test_pred_KPCA = kpca.predict(X_testScaled)

ba_test_KPCA = balanced_accuracy_score(y_test, y_test_pred_KPCA)
ba_train_KPCA = balanced_accuracy_score(y_train, y_train_pred_KPCA)


print("KPCA BA test:", ba_test_KPCA)
print("KPCA BA train:", ba_train_KPCA)

