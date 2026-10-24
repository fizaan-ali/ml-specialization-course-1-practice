import copy
from pathlib import Path
import numpy as np

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = cost_sum / (2 * m)
    return total_cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i][j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):

    hist = {}
    hist["Cost"] = []
    hist["Gradients"] = []
    hist["Parameters"] = []
    hist["Iteration"] = []

    w = w_in
    b = b_in

    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        hist["Cost"].append(compute_cost(X, y, w, b))
        hist["Gradients"].append((dj_dw, dj_db))
        hist["Parameters"].append((w, b))
        hist["Iteration"].append(i)

    return w, b, hist


def run_gradient_descent(X, y, alpha=1.0e-6, iterations=1000):
    w = np.zeros(X.shape[1])
    b = 0
    w_out, b_out, _ = gradient_descent(X, y, w, b, alpha, iterations, compute_cost, compute_gradient)

    print(f"w, b found by gradient descent: {w_out}, {b_out}")

    return w_out, b_out

def z_score_normalization(X):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def load_house_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "house_prices.csv"
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:,:4]
    y = data[:,4]
    return X, y