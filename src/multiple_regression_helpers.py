import math
import copy
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

def compute_cost(X, y, w, b):
    m = X.shape[0] # no. of training examples
    cost_sum = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b # our model prediction
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = cost_sum / (2 * m)
    return total_cost

def compute_gradient(X, y, w, b):
    m,n = X.shape # no. of examples and no. of features

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db 

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):
    
    j_history = [] # cost history while our parameters are changing
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        j_history.append(compute_cost(X, y, w, b))


        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")

    return w, b, j_history