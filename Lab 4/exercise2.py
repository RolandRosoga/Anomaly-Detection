import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

mat = loadmat(r"C:\Users\radur\Documents\UNIBUC FMI\AD\Labs AD\Lab 4\cardio_1.mat")
X = mat["X"]
y = mat["y"].ravel().astype(int)

print(mat.keys())

Y_sklearn = np.where(y == 0, 1, -1)

xtr, xte, ytr, yte = train_test_split(
    X, Y_sklearn, train_size=0.4, shuffle=True, stratify=Y_sklearn
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("oc", OneClassSVM())
])

grid = {
    "oc__kernel": ["linear", "rbf", "poly"],
    "oc__nu": [0.05, 0.1, 0.15, 0.2],
    "oc__gamma": ["scale", "auto"]
}

gs = GridSearchCV(
    pipe,
    grid,
    scoring="balanced_accuracy",
    cv=3,
    n_jobs=-1
)

gs.fit(xtr, ytr)

best_model = gs.best_estimator_
pred = best_model.predict(xte)

balanced_accuracy_score = balanced_accuracy_score(yte, pred)

print("Best parameters:", gs.best_params_)
print("Balanced Accuracy:", balanced_accuracy_score)
