# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:16:07 2015

@author: rlabbe
"""

import numpy as np
import math
from scipy.misc import comb


def coin_likelihood(N, d, bias):
    return comb(N, d, exact=True) * bias**d * (1-bias)**(N-d)

#### E-M Coin Toss Example as given in the EM tutorial paper by Do and Batzoglou* ####

# represent the experiments
experiments = np.array([5,9,8,4,7])
num_flips = 10
# initialise the pA(heads) and pB(heads)
theta = np.array([.6, .5])

# E-M begins!
delta = 0.0001
improvement = float('inf')
likelihoods = np.zeros(2)
expectation = np.zeros((5,2,2))
EA = np.zeros((5,2))
EB = np.zeros((5,2))

while improvement > delta:
    for i, num_heads in enumerate(experiments):
        for coin in range(2):
            # likelihood of e given coin bias
            likelihoods[coin] = coin_likelihood(10, num_heads, theta[coin])

        likelihoods /= sum(likelihoods) # normalize

        e = [num_heads, num_flips-num_heads]
        EA = np.dot(likelihoods[0], e)
        EB = np.dot(likelihoods[1], e)

        for coin in range(2):
            f = np.dot(likelihoods[coin], e)
            expectation[i,:,coin] = np.dot(likelihoods[coin], e)

        expectation[i] = np.outer(likelihoods, [num_heads, num_flips-num_heads])

    old_theta = theta.copy()
    for coin in range(2):
        theta[coin] = sum(expectation[:, coin, 0]) / np.sum(expectation[:, coin,:])



    improvement = max(abs(theta - old_theta))

print(theta)
