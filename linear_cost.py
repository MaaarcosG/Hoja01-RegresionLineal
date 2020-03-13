import numpy as np


def linear_cost(X, y, theta, lamda):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    result = sq.sum() / (2 * m)
    theta02 = theta ** 2
    linear = theta02.sum() / (2 * m)
    resultFinish = result + (lamda*linear)
    return resultFinish
