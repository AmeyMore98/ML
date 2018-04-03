# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:25:06 2018

@author: Amay
"""
import numpy as np
from random import shuffle
import tensorflow as tf

train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
        temp_list = [0]*21
        temp_list[count]=1
        train_output.append(temp_list)
        
NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

data = tf.placeholder(tf.float32, [None, 20,1])
target = tf.placeholder(tf.float32, [None, 21])

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple = True)