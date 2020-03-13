import numpy as np


def linear_cost_derivate(X, y, theta, lamda):
    h = np.matmul(X, theta)
    m, _ = X.shape
    result = np.matmul((h - y).T, X).T / m
    regularization = (lamda/m) * theta.sum()
    resultFinish = regularization + result
    return resultFinish