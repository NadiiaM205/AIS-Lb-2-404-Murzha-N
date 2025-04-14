import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Завантаження даних Iris
iris = load_iris()
X, y = iris.data, iris.target

# Поділ на тренувальну і тестову вибірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Створення і навчання Ridge-класифікатора
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

# Прогнозування
y_pred = clf.predict(X_test)

# Метрики якості
print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Precision:', np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))

# Звіт про класифікацію
print('\nClassification Report:\n', metrics.classification_report(y_test, y_pred))

# Матриця плутанини
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix - RidgeClassifier')
plt.savefig("Confusion.jpg")

# Збереження у форматі SVG
f = BytesIO()
plt.savefig(f, format="svg")
plt.show()
