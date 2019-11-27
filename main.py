import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# info about the features used

header_row = ['age','sex','pain','BP','chol','fbs','ecg', \
'maxhr','eiang','eist','slope','vessels','thal','diagnosis']

# Loading the dataset

dataset = pd.read_csv("processed.hungarian.data")

print(dataset.shape)

# Replacing the missing values with the Mean  value of its column

dataset = dataset.replace("?",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(dataset)
dataset = pd.DataFrame(imp.transform(dataset))
dataset.columns = header_row

df = dataset.iloc[:,:-1]
corr = df.corr()


f, ax = plt.subplots()
sns.heatmap(corr, vmax=.8,annot_kws={'size': 20}, annot=False);
plt.show()

# Extracting the features and the results

X, y = dataset.iloc[:,:-1], dataset.iloc[:, -1]

# Splitting the dataset into training and test sets to remove bias

X_train, X_test, y_train, y_test = train_test_split(X, y, \

test_size=1/3, random_state=2)


# Using Logistic Regression as the algorithm for classification

model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

y_pred = model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)


# The output suggests the % of times the model can correctly 
# predict the outcome  i.e. the characteristics would lead 
# to a heart disease or not

print(round(score*100,2), "% Accuracy") # Output: 83.67 % Accuracy

tn = confusion_matrix[0][0]
tp = confusion_matrix[1][1]
fp = confusion_matrix[0][1]
fn = confusion_matrix[1][0]

result = pd.DataFrame({
	"True Negative": [tn,tp],
	"True Positive" :[fp,fn]},
	index=["False Positive", "False Negative"]
	)

f, ax = plt.subplots()
sns.heatmap(result, vmax=62,annot_kws={'size': 10}, annot=False);

plt.show()

print ("True Negative =", tn, ", False Positive =", fp)
print ("False Negative =", fn, ", True Positive =", tp)

