import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.utility import standardizer

mat = loadmat(r"C:\Users\radur\Documents\UNIBUC FMI\AD\Labs AD\Lab 4\shuttle_1.mat")
X = mat["X"]
y = mat["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=True
)

X_train_s, X_test_s = standardizer(X_train, X_test)

m_oc = OCSVM(kernel="rbf", contamination=np.mean(y))
m_oc.fit(X_train_s)
p_oc = m_oc.predict(X_test_s)
s_oc = m_oc.decision_function(X_test_s)

ba_oc = balanced_accuracy_score(y_test, p_oc)
auc_oc = roc_auc_score(y_test, s_oc)

print(" OCSVM Balanced Accuracy: ", ba_oc)
print(" OCSVM Area Under Curve: ", auc_oc)

def run_svdd(hidden_layers):
    m = DeepSVDD(
        n_features=X.shape[1],
        hidden_neurons=hidden_layers,
        contamination=np.mean(y)
    )
    m.fit(X_train_s)
    p = m.predict(X_test_s)
    s = m.decision_function(X_test_s)
    bal_acu = balanced_accuracy_score(y_test, p)
    area_under_curve = roc_auc_score(y_test, s)
    return bal_acu, area_under_curve

architecture_matrix = [
    [32, 16],
    [64, 32],
    [128, 64],
    [128, 64, 32],
    [256, 128, 64]
]


for arch in architecture_matrix:
    bal_acu, area_under_curve = run_svdd(arch)
    print(" DeepSVDD ", arch, " Balanced Accuracy: ", bal_acu, " Area Under Curve: ", area_under_curve)