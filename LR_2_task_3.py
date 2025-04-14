from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("Розмір датасету:", dataset.shape)
print("\nПерші 20 рядків:\n", dataset.head(20))
print("\nОписова статистика:\n", dataset.describe())
print("\nКількість об'єктів у класах:\n", dataset.groupby('class').size())

#КРОК 2:
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.suptitle("Boxplots")
pyplot.show()

dataset.hist()
pyplot.suptitle("Histograms")
pyplot.show()

scatter_matrix(dataset)
pyplot.suptitle("Scatter Matrix")
pyplot.show()

#КРОК 3
array = dataset.values
X = array[:, 0:4]  
Y = array[:, 4]    
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.20, random_state=1, stratify=Y
)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

#КРОК 4
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print("\n Точність моделей (10-кратна крос-валідація) \n")
for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %0.4f (+/- %0.4f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title('Порівняння алгоритмів')
pyplot.ylabel('Accuracy')
pyplot.grid(True)
pyplot.show()

#КРОК 6 - Навчання остаточної моделі
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)

#КРОК 7 - оцінка моделі на контрольній вибірці
predictions = model.predict(X_validation)

print("\n Оцінка якості на тестовій вибірці")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Confusion matrix:\n", confusion_matrix(Y_validation, predictions))
print("Classification report:\n", classification_report(Y_validation, predictions))

# КРОК 8 - Прогноз для нової квітки
X_new = np.array([[6.3, 3.7, 5.0, 2.5]])  # Нові вимірювання

print("\nФорма масиву X_new:", X_new.shape)

new_prediction = model.predict(X_new)
print("Прогноз:", new_prediction[0])
print("Спрогнозована мітка:", new_prediction[0])



