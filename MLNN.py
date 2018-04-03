# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:54:36 2018

@author: Amay
"""

import pandas as pd
import os
import numpy as np
from random import seed
from random import random
from sklearn.metrics import accuracy_score
         
def init_net(n_input, n_hidden, n_output):
    network = []
    hidden_layers = [{'weights': [random() for i in range(n_input + 1)]} for i in range(n_hidden)]
    network.append(hidden_layers)
    output_layers = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_output)]
    network.append(output_layers)
    return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def frwd_propogate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = sigmoid(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def back_propogate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = []
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i+1]:
					error += neuron['weights'][j] * neuron['delta']
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i!=0:
			inputs = [neuron['output'] for neuron in network[i-1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_net(network, train, l_rate, n_epochs, n_outputs):
	for epoch in range(n_epochs):
		for row in train:
			outputs = frwd_propogate(network, row)
			expected = [row[-1] for row in train]
			#expected[int(row[-1] - 1)] = 1
			#print(expected)
			back_propogate_error(network, expected)
			update_weights(network, row, l_rate)

def predict(network, row):
	outputs = frwd_propogate(network, row)
	return outputs.index(max(outputs))

def back_propogation(train, test, l_rate, n_epochs, n_hidden):
    n_inputs = len(train[0]) - 1
    #print(n_inputs)
    n_outputs = len(set([row[-1] for row in train]))
    #print(n_outputs)
    network = init_net(n_inputs, n_hidden, n_outputs)
    train_net(network, train, l_rate, n_epochs, n_outputs)
    pred = []
    for row in test:
        prediction = predict(network, row[:-1])
        pred.append(prediction)
    return pred

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df = pd.read_csv('seed_dataset.csv')

train = df.values[:180, :]
test = df.values[180:, :]

"""train = [[0,0,0,0],[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0]]
test = [[1,1,0,0],[1,1,1,1]]"""
#print(df.dtypes)

l_rate = 0.5
epochs = 1
n_hidden = 8
pred = back_propogation(train, test, l_rate, epochs, n_hidden)
actual = [row[-1] for row in test]

print('ACC:', accuracy_score(actual, pred)*100)


