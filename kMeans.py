#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:39:47 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load data
testData = np.load('normalizedData.npy')

# Let's see what we got
plt.scatter(testData[:,0], testData[:,1])
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('clusteredData0.pdf')
plt.show()

# Define two 'visually pleasing' -- otherwise bad -- initial guesses
centers = np.array([(-2,3),(-1,2)])

# The KMeans implementation performs computation of new means and reassignment of data
# points in one call. To visualize both steps, we generate two plots from each
# KMeans call.

oldlabels=None
j = 1

for i in range(10):
    
  k_means = KMeans(n_clusters=2, max_iter=1, init=centers, n_init=1, algorithm="full").fit(testData)
  labels = k_means.labels_
  centers = k_means.cluster_centers_
  
  plt.plot(centers[:,0],centers[:,1],'kx', markersize=15)
  plt.scatter(testData[:,0], testData[:,1], c=oldlabels)
  plt.xlabel('x')
  plt.ylabel('y')
  ax = plt.gca()
  ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
  ax.set_aspect('equal','box')
  name='clusturedData'+str(j)+'.pdf'
  plt.savefig(name)
  plt.show()
  
  j = j + 1
 
  plt.plot(centers[:,0],centers[:,1],'kx', markersize=15)
  plt.scatter(testData[:,0], testData[:,1], c=labels)
  plt.xlabel('x')
  plt.ylabel('y')
  ax = plt.gca()
  ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
  ax.set_aspect('equal','box')
  name='clusturedData'+str(j)+'.pdf'
  plt.savefig(name)
  plt.show()
  
  j = j + 1
  
  oldlabels = labels
  
# Save converged results
np.save('clusterLabels.npy', labels)