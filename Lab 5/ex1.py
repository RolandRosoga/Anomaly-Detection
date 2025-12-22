import numpy as np
import matplotlib.pyplot as plot

np.random.seed(0)

mean = np.array([5, 10, 2])
cov = np.array([[3, 2, 2],
                [2, 10, 1],
                [2, 1, 2]])

X = np.random.multivariate_normal(mean, cov, size=500)

X_centered = X - X.mean(axis=0)

Sigma = np.cov(X_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

explained_var = eigenvalues / eigenvalues.sum()
cum_explained_var = np.cumsum(explained_var)

plot.figure()
plot.bar(range(1, 4), explained_var)
plot.step(range(1, 4), cum_explained_var, where='mid', color='red')
plot.xlabel("Principal Component")
plot.ylabel("EV")
plot.show()

X_proj = X_centered @ eigenvectors

cont = 0.1

pc3 = X_proj[:, 2]
pc3_dev = np.abs(pc3 - pc3.mean())
thr_pc3 = np.quantile(pc3_dev, 1 - cont)
labels_pc3 = pc3_dev > thr_pc3

pc2 = X_proj[:, 1]
pc2_dev = np.abs(pc2 - pc2.mean())
thr_pc2 = np.quantile(pc2_dev, 1 - cont)
labels_pc2 = pc2_dev > thr_pc2

X_norm = X_proj / np.sqrt(eigenvalues)
midd = X_norm.mean(axis=0)
dist = np.sum((X_norm - midd) ** 2, axis=1)

thr_dist = np.quantile(dist, 1 - cont)
labels_dist = dist > thr_dist

def plot_3d(X, lable, title):
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[~lable, 0], X[~lable, 1], X[~lable, 2], s=10)
    ax.scatter(X[lable, 0], X[lable, 1], X[lable, 2], s=10)
    ax.set_title(title)
    plot.show()

plot_3d(X, labels_pc3, "Anom based on PC3 deviation")
plot_3d(X, labels_pc2, "Anom based on PC2 deviation")
plot_3d(X, labels_dist, "Anom based on PCA normalized distance")
