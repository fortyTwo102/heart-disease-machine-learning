import csv
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.impute import SimpleImputer
from itertools import combinations

from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier, plot_importance
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


dataset = pd.read_csv('dataset.csv')

dataset = dataset.replace(" ",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(dataset)
dataset = pd.DataFrame(imp.transform(dataset))

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
X.columns = ['AGE','GENDER', 'TB','DB','ALKPHOS','SGPT','SGOT','TP','ALB','A/G']

# X = X.drop(['ALKPHOS','TP','A/G'], axis = 1)

print(X.head)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)


scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(C = 10) # XGBClassifier(n_estimators = 1000, random_state = 2) #RandomForestClassifier(n_estimators = 10000)#

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# print(X_train.shape, X_test.shape)

# accuracy = round(float((model.score(X_test, y_test)*100)),2)

accuracy = round(accuracy_score(y_test, y_pred)*100, 2)


print("Accuracy: ", accuracy)


# Logistic with C = 0.01 with [2,4,9] has 74.74% accuracy 
# MLPClassifier with [1,5,7] has 74.23% accuracy
# RandomForestClassifier with [4,6,8] has 77.32% accuracy
# RandomForestClassifier Accuracy:  78.35  with  [1, 3, 4, 6, 8]
# RandomForestClassifier Accuracy:  79.9  with  [0, 1, 2, 3, 4, 5, 7]
# DecisionTreeClassifier Accuracy:  75.26  with  [0, 1, 2, 5]
# MLPClassifier Accuracy:  75.77  with  [1, 2, 4, 5, 6, 7, 8, 9]
# Logistic with C = 0.1 Accuracy:  75.26  with  [2, 4, 8, 9]
# Logistic with C = 1 Accuracy:  77.32  with  [0, 1, 3, 4, 5, 6, 8]
# MLPClassifier + hidden_layer_sizes (10, 5) +  max_iter = 10000 Accuracy:  78.35  with  [0, 3, 5, 7, 8, 9]
# KNeightborsClassifier Accuracy:  76.29  with  [2, 5, 6]
# LogisticRegression Accuracy:  78.08  with  [0, 1, 2, 3, 5, 6, 8] test_size=1/4, StandardScaler, rand = 2, C = 10
# 
# new script with
# proper scaling
# data binning
# accuracy_score