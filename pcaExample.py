#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:39:47 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

normalizedData = np.load('normalizedData.npy')

# PCA
pca = PCA(n_components=1).fit(normalizedData)
transformedData = PCA(n_components=1).fit_transform(normalizedData)
transformedCoords = transformedData * pca.components_

# Linear Regression as reference
fit = np.polyfit(normalizedData[:,0], normalizedData[:,1], deg=1)

# Plot data, PCA and lin reg.
plt.scatter(normalizedData[:,0],normalizedData[:,1])
plt.plot(normalizedData[:,0], fit[0] * normalizedData[:,0] + fit[1], color='black')
plt.scatter(transformedCoords[:,0],transformedCoords[:,1])
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('pcaTransformed.pdf')
plt.show()