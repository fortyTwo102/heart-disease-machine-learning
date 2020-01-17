import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.impute import SimpleImputer
from itertools import combinations

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv('dataset.csv')

dataset = dataset.replace(" ",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imp.fit(dataset)
dataset = pd.DataFrame(imp.transform(dataset))

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
X = preprocessing.scale(X)

max_acc = 0

X = pd.DataFrame(X)

for i in range(1,11): # no. of columns at a time

	for columns in combinations(range(10),i): # all combinations of i columns

		columns = list(columns)

		X = pd.DataFrame(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 2)


		X_train = X_train[columns]
		X_test = X_test[columns]

		model =  KNeighborsClassifier() # MLPClassifier(hidden_layer_sizes = (10, 5), max_iter = 10000) # LogisticRegression(solver = 'liblinear') # RandomForestClassifier() # SVC() #XGBClassifier() #LogisticRegression(C = 0.1)
		model.fit(X_train, y_train)
		#print(model.feature_importances_)
		y_pred = model.predict(X_test)

		accuracy = round(float((model.score(X_test, y_test)*100)),2)

		if accuracy > max_acc:

			max_acc = accuracy
			best_columns = columns
			print("max till now", max_acc)


print("Accuracy: ", max_acc, ' with ', best_columns)


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