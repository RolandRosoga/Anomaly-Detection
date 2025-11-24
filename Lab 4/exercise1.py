from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data(
    n_train=300,
    n_test=200,
    n_features=3,
    contamination=0.15
)

def plot_3dimensions(title, Xtr, Xte, ytr, yte):
    fig = plt.figure(figsize=(10, 10))
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        if i < 2:
            c_tr = ['blue' if v == 0 else 'red' for v in ytr]
            c_te = ['blue' if v == 0 else 'red' for v in yte]
            ax.scatter(Xtr[:, 0], Xtr[:, 1], Xtr[:, 2], c=c_tr, s=20)
            ax.scatter(Xte[:, 0], Xte[:, 1], Xte[:, 2], c=c_te, s=20)
            ax.set_title(title + f" GT ({' Train ' if i==0 else ' Test '})")
        else:
            c_tr = ['green' if v == 0 else 'orange' for v in ytr]
            c_te = ['green' if v == 0 else 'orange' for v in yte]
            ax.scatter(Xtr[:, 0], Xtr[:, 1], Xtr[:, 2], c=c_tr, s=20)
            ax.scatter(Xte[:, 0], Xte[:, 1], Xte[:, 2], c=c_te, s=20)
            ax.set_title(title + f" Predict ( {' Train ' if i==2 else ' Test '} )")
    plt.tight_layout()
    plt.show()

#Linear
m_lin = OCSVM(kernel='linear', contamination=0.15)
m_lin.fit(X_train)
p_lin = m_lin.predict(X_test)
s_lin = m_lin.decision_function(X_test)

ba_lin = balanced_accuracy_score(y_test, p_lin)
roc_lin = roc_auc_score(y_test, s_lin)

print("OC-SVM Linear BA:", ba_lin)
print("OC-SVM Linear ROC AUC:", roc_lin)

plot_3dimensions("OC-SVM Linear",
        X_train, X_test,
        m_lin.labels_, p_lin)

#RBF
m_rbf = OCSVM(kernel='rbf', contamination=0.15)
m_rbf.fit(X_train)
p_rbf = m_rbf.predict(X_test)
s_rbf = m_rbf.decision_function(X_test)

ba_rbf = balanced_accuracy_score(y_test, p_rbf)
roc_rbf = roc_auc_score(y_test, s_rbf)

print("OC-SVM RBF Balanced Accuracy:", ba_rbf)
print("OC-SVM RBF Receiver Operating Characteristic, Area Under the ROC Curve:", roc_rbf)

plot_3dimensions("OC-SVM RBF",
        X_train, X_test,
        m_rbf.labels_, p_rbf)

#Deep SVDD
m_dsvdd = DeepSVDD(n_features=3, contamination=0.15)
m_dsvdd.fit(X_train)
p_dsvdd = m_dsvdd.predict(X_test)
s_dsvdd = m_dsvdd.decision_function(X_test)

ba_dsvdd = balanced_accuracy_score(y_test, p_dsvdd)
roc_dsvdd = roc_auc_score(y_test, s_dsvdd)

print("DeepSVDD Balanced Accuracy:", ba_dsvdd)
print("DeepSVDD Receiver Operating Characteristic, Area Under the ROC Curve:", roc_dsvdd)

plot_3dimensions("Deep SVDD",
        X_train, X_test,
        m_dsvdd.labels_, p_dsvdd)
