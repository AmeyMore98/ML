# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:19:50 2018

@author: Amay
"""
import numpy as np
from sklearn.metrics import accuracy_score

def predict(example, weights):
    o = 0
    for i in range(len(example)-1):
        o += weights[i] * example[i]
    return 1.0 if o >= 0.0 else 0.0

def train_weights(train, l_rate, epochs):
    #weights = [0.8 for i in range(len(train[0])-1)]
    #weights = np.random.uniform(0.5,1.0,(1,len(train[0])-1))
    weights = [-0.8, 0.5, 0.5, 0.5]
    print(weights)
    for epoch in range(epochs):
        for row in train:
            o = predict(row, weights)
            if o != row[-1]:
                for i in range(len(row)-1):
                    weights[i] = weights[i] + (l_rate * (row[-1] - o) * row[i])
    return weights

def perceptron(train, test, l_rate, epochs):
    pred = []
    weights = train_weights(train, l_rate, epochs)
    print(weights)
    for row in test:
        prediction = predict(row, weights)
        pred.append(prediction)
    return pred

train = [[0,0,0,0], [0,0,1,0],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,0,0],[1,0,0,0]]
train = np.insert(train, 0,1,axis=1)
test = [[1,0,1,0],[1,1,0,0],[1,0,0,0],[1,1,1,1]]
test = np.insert(test, 0,1,axis=1)

l_rate = 0.5
epochs = 2500
pred = perceptron(train, test, l_rate, epochs)
actual = [float(row[-1]) for row in test]

print("ACC:",accuracy_score(actual, pred)*100)
