import matplotlib.pyplot as plot_canvas
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train=400, n_test=100, n_features=2, contamination_rate=0.1, random_state=42
)

plot_canvas.figure(figure_size=(6, 5))
plot_canvas.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="blue", label="inliers")
plot_canvas.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="red", label="outliers")
plot_canvas.title("2D Values for a contamination_rate of 0.1)")
plot_canvas.legend()
plot_canvas.show()

#Exercitiul 2:

from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plot_canvas

contamination_rate = 0.1
current_model = KNN(contamination_rate=contamination_rate)
current_model.fit(X_train)

y_train_pred = current_model.labels_
y_test_pred = current_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
bal_accu_score = balanced_accuracy_score(y_test, y_test_pred)

print(f"Confusion matrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"The balanced accuracy is = {bal_accu_score:.3f}")

y_scores = current_model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plot_canvas.figure(figure_size=(6,5))
plot_canvas.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plot_canvas.plot([0,1],[0,1],"k--")
plot_canvas.xlabel("FPR")
plot_canvas.ylabel("TPR")
plot_canvas.title("ROC curve (KNN)")
plot_canvas.legend()
plot_canvas.show()

#Test pentru a vedea schimbarea metrica
for cont in [0.05, 0.15, 0.25]:
    m = KNN(contamination_rate=cont).fit(X_train)
    y_pred = m.predict(X_test)
    bal_accu_score = balanced_accuracy_score(y_test, y_pred)
    print(f"contamination_rate={cont:.2f} → Balanced Accuracy={bal_accu_score:.3f}")
