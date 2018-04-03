# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:58:03 2018

@author: Amay
"""
import numpy as np
data = open('go.txt', 'r').read()

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#print('data has %d, %d unique' % (data_size, vocab_size))

char_to_ix = {ch:ix for ix,ch in enumerate(chars)}
ix_to_char = {ix:ch for ix,ch in enumerate(chars)}
#print(char_to_ix)
#print(ix_to_char)

vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1
#print(vector_for_char_a.ravel())

#hyperpatameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

#model_parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))