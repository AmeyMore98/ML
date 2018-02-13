from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import numpy as np

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df2 = pd.read_csv('pima_diabetes.csv')

df = pd.get_dummies(df2)
print(df.head())

X = df.values[:, :8]
Y =  df.values[:, 8]

X.reshape(-1,1)
Y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

gnb = GaussianNB()

model = gnb.fit(X_train, y_train)

y_pred = model.predict(y_test.reshape(-1,1))

#print(y_pred, y_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)