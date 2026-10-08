"""Listing 9.1 — Normal equations vs. a least-squares solver

From Numbers to Thought — Chapter 9: Continuous Prediction: Regression from Least Squares to the Bayesian View
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
x = np.array([-1, 0, 1, 2.]); y = np.array([3, 1, 2, 7.])
Phi = np.vander(x, 3, increasing=True)            # columns: 1, x, x^2
 
theta_ne = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)   # normal equations (9.19)
theta_ls, *_ = np.linalg.lstsq(Phi, y, rcond=None)  # SVD-based, preferred
print(theta_ne, theta_ls)                          # -> [0.85 -0.45 1.75]
print(Phi.T @ (y - Phi @ theta_ls))               # residual is orthogonal: ~0
