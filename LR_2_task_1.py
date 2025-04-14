import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

input_file = 'income_data.txt'

X_raw = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X_raw.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X_raw.append(data)
            count_class2 += 1

X_raw = np.array(X_raw)
X_encoded = np.empty(X_raw.shape)
label_encoder = []

for i in range(X_raw.shape[1]):
    if X_raw[0, i].isdigit():
        X_encoded[:, i] = X_raw[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X_raw[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

# Прогнозування 
y_pred = classifier.predict(X_test)


print("Оцінка якості класифікації:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nЗвіт про класифікацію:\n")
print(classification_report(y_test, y_pred))

#Тестова точка
input_data = ['54', 'Private', '302146', 'HS-grad', '9', 'Separated', 'Other-service', 'Unmarried',
               'Black', 'Female', '0', '0', '20', 'United-States']

input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
        count += 1

input_data_encoded = np.array([input_data_encoded])  # має бути 2D масив
predicted_class = classifier.predict(input_data_encoded)

print("\nРезультат класифікації тестової точки:")
print("Належить до класу:", label_encoder[-1].inverse_transform(predicted_class)[0])
