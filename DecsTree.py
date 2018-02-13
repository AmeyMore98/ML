import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz
import os

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df = pd.read_csv('playtennis.csv')

df2 = pd.get_dummies(df)

X = df2.drop(['playtennis_yes','playtennis_no'], axis = 1)
Y = df2['playtennis_yes']

#print(X)
#print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 190)

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)



os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML')
tree.export_graphviz(clf, out_file='tree.dot', feature_names = X.columns, filled = True)