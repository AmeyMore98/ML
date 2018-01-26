import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def accuracy(y_test, y_pred):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count += 1
    return count/len(y_test)


iris = load_iris()

df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns = iris['feature_names'] + ['target'])

X = df.values[:,0:4]
Y = df.values[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Expected:\n', y_test,'\nPrediction:\n',y_pred)

print("Accuracy (Implicit): ", accuracy_score(y_test, y_pred) * 100)
print("Accuracy (Explicit): ", accuracy(y_test, y_pred) * 100 )
