import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.loda import LODA

RNG = np.random.RandomState(42)

def ex1_simple_loda(n_points=500, n_proj=5, n_bins=20, test_range=3.0):
    X, _ = make_blobs(n_samples=n_points, n_features=2, centers=None, cluster_std=1.0, random_state=RNG)
    vecs = RNG.multivariate_normal([0, 0], np.eye(2), size=n_proj)
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    proj = X @ vecs.T
    hists = []
    for i in range(n_proj):
        hist, edges = np.histogram(proj[:, i], bins=n_bins, range=(proj[:, i].min()-1e-6, proj[:, i].max()+1e-6))
        p = hist.astype(float) / hist.sum()
        hists.append((p, edges))
    T = RNG.uniform(-test_range, test_range, size=(n_points, 2))
    Tp = T @ vecs.T
    scores = []
    for j in range(T.shape[0]):
        ps = []
        for i in range(n_proj):
            p, edges = hists[i]
            idx = np.searchsorted(edges, Tp[j, i]) - 1
            idx = np.clip(idx, 0, len(p) - 1)
            ps.append(p[idx])
        scores.append(np.mean(ps))
    scores = np.array(scores)
    plt.figure()
    plt.scatter(T[:, 0], T[:, 1], c=scores, s=12)
    plt.title("Simple LODA Scores")
    plt.colorbar(label='scores')
    plt.show()
    return

def ex2_if_loda(n_samples=1000):
    X2, _ = make_blobs(n_samples=n_samples, n_features=2, centers=[[10, 0], [0, 10]], cluster_std=1.0, random_state=RNG)
    T2 = RNG.uniform(-10, 20, size=(n_samples, 2))
    m_if = IForest(contamination=0.02, n_estimators=100, max_samples=256, random_state=RNG)
    m_if.fit(X2)
    score_if = m_if.decision_function(T2)
    m_loda = LODA(n_bins=20, n_random_cuts=10)
    m_loda.fit(X2)
    score_loda = m_loda.decision_function(T2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(T2[:, 0], T2[:, 1], c=score_if, s=12)
    ax[0].set_title("IsolationForest scores")
    ax[1].scatter(T2[:, 0], T2[:, 1], c=score_loda, s=12)
    ax[1].set_title("LODA scores")
    plt.tight_layout()
    plt.show()
    return

def safe_roc(y_true, scores):
    return roc_auc_score(y_true, scores)

def ex3_shuttle(path, n_splits=10, test_size=0.4):
    data = loadmat(path)
    X = data['X']
    y = data['y'].ravel()
    print("Loaded X:", X.shape, "y:", y.shape)
    ba_res = {"IF": [], "LODA": []}
    roc_res = {"IF": [], "LODA": []}
    for i in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RNG.randint(1e9))
        Xtr_s, Xte_s = standardizer(Xtr, Xte)
        print(f"Split {i+1}: Xtr {Xtr_s.shape}, Xte {Xte_s.shape}")
        m1 = IForest(contamination=0.02, n_estimators=100, max_samples=256, random_state=RNG)
        m1.fit(Xtr_s)
        s1 = m1.decision_function(Xte_s)
        pred1 = (s1 > np.quantile(s1, 0.98)).astype(int)
        ba_res['IF'].append(balanced_accuracy_score(yte, pred1))
        roc_res['IF'].append(safe_roc(yte, s1))
        m2 = LODA(n_bins=20, n_random_cuts=10)
        m2.fit(Xtr_s)
        s2 = m2.decision_function(Xte_s)
        pred2 = (s2 > np.quantile(s2, 0.98)).astype(int)
        ba_res['LODA'].append(balanced_accuracy_score(yte, pred2))
        roc_res['LODA'].append(safe_roc(yte, s2))
    print('BA:', {k: np.mean(v) for k, v in ba_res.items()})
    print('ROC:', {k: np.nanmean(v) for k, v in roc_res.items()})
    return

if __name__ == "__main__":
    SHUTTLE_PATH = r"C:\Users\radur\Documents\UNIBUC FMI\AD\Labs AD\Lab 3\shuttle.mat"
    ex1_simple_loda()
    ex2_if_loda()
    ex3_shuttle(SHUTTLE_PATH)
