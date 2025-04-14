import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

input_file = 'income_data.txt'
X_raw = []
max_points = 25000
class1 = 0
class2 = 0

with open(input_file, 'r') as f:
    for line in f:
        if '?' in line:
            continue
        row = line.strip().split(', ')
        if row[-1] == '<=50K' and class1 < max_points:
            X_raw.append(row)
            class1 += 1
        elif row[-1] == '>50K' and class2 < max_points:
            X_raw.append(row)
            class2 += 1
        if class1 >= max_points and class2 >= max_points:
            break

X_raw = np.array(X_raw)
X_encoded = np.empty(X_raw.shape)
label_encoders = []

for i in range(X_raw.shape[1]):
    if X_raw[0, i].isdigit():
        X_encoded[:, i] = X_raw[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X_raw[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

clf = SVC(kernel='sigmoid')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n Сигмоїдальне ядро = ")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

#Тестова точка 
input_data = ['54', 'Private', '302146', 'HS-grad', '9', 'Separated',
              'Other-service', 'Unmarried', 'Black', 'Female',
              '0', '0', '20', 'United-States']

input_encoded = []
count = 0
for i, val in enumerate(input_data):
    if val.isdigit():
        input_encoded.append(int(val))
    else:
        input_encoded.append(int(label_encoders[count].transform([val])[0]))
        count += 1

prediction = clf.predict([input_encoded])
print("Клас тестової точки:", label_encoders[-1].inverse_transform(prediction)[0])
