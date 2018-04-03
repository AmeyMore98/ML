# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:14:36 2018

@author: Amay
"""

def fitness(password, test_word):
    if(len(password)!=len(test_word)):
        print('incompatible')
        return
    else:
        i = 0
        score = 0
        while(i < len(password)):
            if(password[i] == test_word[i]):
                score+=1
            i+=1
        return score * 100 / len(password)