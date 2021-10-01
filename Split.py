#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:39:47 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
allData = np.load('normalizedData.npy')

x_train, x_test, y_train, y_test = train_test_split(
                    allData[:,0], allData[:,1], random_state=91)

plt.plot(x_train, y_train, 'ko')
plt.plot(x_test, y_test, 'ro')
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('trainAndTestData.pdf')
plt.show()

plt.plot(x_train, y_train, 'ko')
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('trainingData.pdf')
plt.show()

plt.plot(x_test, y_test, 'ro')
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('testingData.pdf')
plt.show()