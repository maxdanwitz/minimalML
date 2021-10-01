#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small neural network successfully completing classification task.

Created on Fri Oct  1 17:39:47 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

allData = np.load('normalizedData.npy')
labels = np.load('clusterLabels.npy')

# We use the kMeans-clustering result as ground truth.
plt.scatter(allData[:,0], allData[:,1], c=labels)
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('classGroundTrut.pdf')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(allData, labels, random_state=91)

# Define NN
nn_class = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')])

# nn_class.compile(loss='binary_crossentropy', optimizer='adam')
nn_class.compile(loss='mse', optimizer='adam')

# Training
results_class = nn_class.fit(x_train, y_train, epochs=100, batch_size= 1, verbose=1)

# This is what we fed into the NN
plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('classTrainData.pdf')
plt.show()

# Testing
y_pred = nn_class.predict(x_test)

# This is what we test with
plt.scatter(x_test[:,0], x_test[:,1])
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('testDataAlone.pdf')
plt.show()

# Pretty good result
plt.scatter(x_test[:,0], x_test[:,1], c=y_pred)
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('classCheckPred.pdf')
plt.show()

plt.scatter(x_test[:,0], x_test[:,1], c=y_pred)
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('classCheckPredTransparent.pdf', transparent=True)
plt.show()

# True assignments of testing data
plt.scatter(x_test[:,0], x_test[:,1], c=y_test)
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('classCheckTestData.pdf')
plt.show()