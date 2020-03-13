import numpy as np


def gradient_descent(
        X,
        y,
        theta_0,
        cost,
        cost_derivate,
        lamda = 3,
        alpha=0.01,
        treshold=0.0001,
        max_iter=10000):
    theta, i = theta_0, 0
    costs = []
    gradient_norms = []
    while np.linalg.norm(cost_derivate(X, y, theta, lamda)) > treshold and i < max_iter:
        theta -= alpha * cost_derivate(X, y, theta, lamda)
        i += 1
        costs.append(cost(X, y, theta, lamda))
        gradient_norms.append(cost_derivate(X, y, theta, lamda))
    return theta, costs, gradient_norms
