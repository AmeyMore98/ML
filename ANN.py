# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 05:11:13 2018

@author: Amay
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

X_train = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]])
X_train = np.insert(X_train, 0, 1, axis = 1)

y_train = np.array([0,1,1,1,1,1])
X_test = np.array([[1,1,0],[1,1,1]])

X_test = np.insert(X_test, 0, 1, axis = 1)
#print(X_test)
y_test = np.array([1,1])

weight = [0.0,0.0,0.0,0.0]

#training
for t_i in range(len(X_train)):
    o = 0
    for i in range(len(X_train[t_i])):
        o += (X_train[t_i][i] * weight[i])
    if o != y_train[t_i]:
        for i in range(len(weight)):
            weight[i] = weight[i] + (0.1 * (y_train[t_i] - o) * X_train[t_i][i])
    
#test 
pred = []
for example in X_test:
    for i in range(len(example)):
        o += example[i] * weight[i]
    pred.append(1) if o > 0.5 else pred.append(0)
    
print('ACC:', accuracy_score(y_test, pred)*100)