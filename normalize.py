#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:39:47 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load, normalize, save, that's it.
exampleData = np.load('exampleData.npy')
normalizedData = StandardScaler().fit_transform(exampleData)
np.save('normalizedData.npy', normalizedData)

# Example data plot
plt.scatter(exampleData[:,0],exampleData[:,1])
plt.axis('equal')
plt.xlabel('CD4 Zellzahl/$\mu$l')
plt.ylabel('Pfad-Markierungs-Test-Dauer in Sek.')
plt.savefig('exampleData.pdf')
plt.show()

# Normalized data plot 
plt.scatter(normalizedData[:,0],normalizedData[:,1])
plt.xlabel('x')
plt.ylabel('y')
ax = plt.gca()
ax.set(xlim=(-2.5, 4), ylim=(-2.5, 4))
ax.set_aspect('equal','box')
plt.savefig('normalizedData.pdf')
plt.show()