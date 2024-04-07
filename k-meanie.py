# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:39:59 2024

@author: wende
"""

import numpy as np
#import matplotlib.pyplot as plt

X = np.genfromtxt('US_population_dataset.csv', delimiter=',')

def find_clusters(X, clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:clusters]
    centers = X[i]
    while True:
        labels = np.argmin(np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2)), axis=0)
        #centers2 = np.array([X[labels == j].mean(0)
                             #for j in range(clusters)])
        centers2 = np.array([X[labels == j].mean(axis=0) if X[labels == j].size > 0 else X[rng.randint(X.shape[0]), :] for j in range(clusters)])

        if np.all(centers == centers2):
            break
        centers = centers2

    return centers, labels

centers, labels = find_clusters(X, 4)
#plt.scatter(X[:, 0], X[:, 1], c= labels, s=50, cmap='viridis')
#plt.show()