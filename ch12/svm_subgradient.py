"""Listing 12.2 — Linear SVM by subgradient descent

From Numbers to Thought — Chapter 12: Decision Boundaries: Support Vector Machines and Kernel Methods
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np

def svm_subgradient(X, y, C=1.0, eta=0.5, epochs=20000):
    w, b = np.zeros(X.shape[1]), 0.0
    for t in range(epochs):
        m = y * (X @ w + b)             # margins y_n f(x_n)
        mask = m < 1                    # nonzero hinge loss
        gw = w - C * (y[mask, None] * X[mask]).sum(axis=0)
        gb = -C * y[mask].sum()
        step = eta / (1 + 0.01 * t)     # decaying step size
        w, b = w - step * gw, b - step * gb
    return w, b

X = np.array([[1, 2], [2, 0], [0, 1], [4, 2],
              [4, 4], [6, 1], [3, 2], [4, 3]])
y = np.array([-1, -1, -1, 1, 1, 1, 1, -1])
w, b = svm_subgradient(X, y, C=0.1)
obj = 0.5 * w @ w + 0.1 * np.maximum(0, 1 - y * (X @ w + b)).sum()
print(w.round(2), round(b, 2), round(obj, 3))
