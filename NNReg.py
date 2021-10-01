#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small neural network for simple regression.

Created on Fri Oct  1 17:39:47 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the network architecture
nn_reg = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4, activation='relu', use_bias=False),
  tf.keras.layers.Dense(4, activation='relu', use_bias=False),
  tf.keras.layers.Dense(1, activation='linear', use_bias=False)])

# Select loss function and optimizer
nn_reg.compile(loss='mse', optimizer='adam')

# Load data
allData = np.load('normalizedData.npy')

# Split
x_train, x_test, y_train, y_test = train_test_split(
                    allData[:,0], allData[:,1], random_state=91)

# Train
results_reg = nn_reg.fit(x_train, y_train, epochs=10, 
                         batch_size=1, verbose=1)

# Test
y_pred = nn_reg.predict(x_test)

# Plot prediction obtained for test data
plt.plot(x_test, y_test, 'ro')
plt.plot(x_test, y_pred, 'bo')
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('testDataAndPred.pdf')
plt.show()

# Add linear regression (of training data!) as reference
fit = np.polyfit(x_train, y_train, deg=1)
plt.plot(x_test, y_test, 'ro')
plt.plot(x_test, y_pred, 'bo')
plt.plot(x_test, fit[0] * x_test + fit[1], color='black')
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('testDataAndPredAndLinReg.pdf')
plt.show()

# For the sake of completeness, plot prediction obtained for training data
y_train_pred = nn_reg.predict(x_train)
plt.plot(x_train, y_train, 'ro')
plt.plot(x_train, y_train_pred, 'bo')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('trainDataAndPred.pdf')
plt.show()