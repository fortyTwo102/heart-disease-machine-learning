import pandas as pd 
import numpy as np

from sklearn.impute import SimpleImputer

import main

header_row = ['age','sex','pain','BP','chol','fbs','ecg', \
'maxhr','eiang','eist','slope','vessels','thl','diagnosis']

dataset = pd.read_csv("processed.hungarian.data")
print(pd.DataFrame(dataset))
dataset = dataset.replace("?",np.NaN)



imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(dataset)

dataset = pd.DataFrame(imp.transform(dataset))
print(dataset)

dataset.columns = header_row

dataset.to_csv("cleaned.csv")


