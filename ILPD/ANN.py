# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from itertools import combinations

from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
# load the dataset
dataset = pd.read_csv('dataset.csv')

dataset = dataset.replace(" ",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(dataset)
dataset = pd.DataFrame(imp.transform(dataset))

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

# define the keras model

max_acc = 0

X = pd.DataFrame(X)

for i in range(1,11): # no. of columns at a time

	for columns in combinations(range(10),i): # all combinations of i columns

		columns = list(columns)

		X = pd.DataFrame(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)

		X_train = X_train[columns]
		X_test = X_test[columns]

		scaler = preprocessing.StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)


		model = Sequential()
		model.add(Dense(400, input_dim=i, activation='relu'))
		model.add(Dense(400, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))


		sgd = optimizers.SGD(lr=0.26, momentum=0.9, nesterov=False)
		# compile the keras model
		model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
		# fit the keras model on the dataset
		model.fit(X_train, y_train, epochs=100, batch_size=10)
		# evaluate the keras model
		_, accuracy = model.evaluate(X_test, y_test)
		
		if accuracy > max_acc:

			max_acc = accuracy
			best_columns = columns
			print("max till now", max_acc)


print("Accuracy: ", max_acc, ' with ', best_columns)
