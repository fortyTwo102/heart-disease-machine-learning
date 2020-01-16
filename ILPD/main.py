import csv
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('dataset.csv')

dataset = dataset.replace(" ",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imp.fit(dataset)
dataset = pd.DataFrame(imp.transform(dataset))

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)
model = LogisticRegression(C = 0.1, solver = 'liblinear') # RandomForestClassifier() # SVC() #XGBClassifier() #LogisticRegression(C = 0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Accuracy: ", float(model.score(X_test, y_test))*100)