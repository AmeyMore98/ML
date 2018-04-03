# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 07:09:03 2018

@author: Amay
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score

def predict(example, weights):
    o = 0
    #outputs = []
    for i in range(len(example)-1):
        o += weights[i] * example[i]
        #print(o)
    return 1.0 if o >= 0.5 else 0.0

def train_weights(train, l_rate, epochs):
    weights = [0.0 for i in range(len(train[0])-1)]
    for epoch in range(epochs):
        for row in train:
            o = predict(row, weights)
            for i in range(len(row)-1):
                weights[i] = weights[i] + (l_rate * (row[-1] - o) * row[i])
    return weights

def perceptron(train, test, l_rate, epochs):
    pred = []
    weights = train_weights(train, l_rate, epochs)
    for row in test:
        prediction = predict(row, weights)
        pred.append(prediction)
    return pred

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df = pd.read_csv('pima_diabetes.csv', header = None, names = ['Times_Pregnant', 'Glucose_conct', 'Blood_pressure', 'S_K_T', 'serum_insulin', 'BMI', 'Diabetes_pedigree', 'Age', 'Class'])

train = df.values[:536, :]
train = np.insert(train, 0, 1.0, axis=1)
test = df.values[536:, :]
test = np.insert(test, 0, 1.0, axis=1)

l_rate = 0.1
epochs = 1000
pred = perceptron(train, test, l_rate, epochs)
actual = test.transpose().tolist()[-1]

print('ACC:', accuracy_score(actual, pred)*100)
  

