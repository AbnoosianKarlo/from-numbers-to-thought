"""Solution to Exercise 12.15 — The hard-margin SVM dual with a generic QP solver

From Numbers to Thought — Chapter 12: Decision Boundaries: Support Vector Machines and Kernel Methods
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from scipy.optimize import minimize

X = np.array([[1, 2], [2, 0], [0, 1], [4, 2], [4, 4], [6, 1]], float)
y = np.array([-1, -1, -1, 1, 1, 1], float)
Q = (y[:, None] * y[None, :]) * (X @ X.T)

def objective(a):                 # (1/2) a^T Q a - 1^T a
    return 0.5 * a @ Q @ a - a.sum()

res = minimize(objective, np.zeros(6), jac=lambda a: Q @ a - 1,
               bounds=[(0, None)] * 6,
               constraints={"type": "eq", "fun": lambda a: a @ y,
                            "jac": lambda a: y},
               method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
alpha = res.x
w = (alpha * y) @ X
sv = alpha > 1e-6
b = np.mean(y[sv] - X[sv] @ w)
print(alpha.round(3))             # [0.111 0.167 0.    0.278 0.    0.   ]
print(w.round(3), round(b, 3))    # [0.667 0.333] -2.333
