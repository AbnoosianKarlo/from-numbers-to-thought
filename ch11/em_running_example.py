"""Listing 11.1 — The EM algorithm on the running example

From Numbers to Thought — Chapter 11: Hidden Structure: Gaussian Mixture Models and the EM Algorithm
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from scipy.stats import norm
 
x = np.array([-4, -3, -1, 0, 1, 3, 5.])
mu, var, pi = np.array([-5, 0, 6.]), np.array([1, 0.5, 2.]), np.ones(3) / 3
 
for it in range(5):
    R = pi * norm.pdf(x[:, None], mu, np.sqrt(var))     # E-step
    loglik = np.log(R.sum(axis=1)).sum()
    R /= R.sum(axis=1, keepdims=True)                  # responsibilities
    Nk = R.sum(axis=0)
    mu = (R * x[:, None]).sum(axis=0) / Nk             # M-step
    var = (R * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    pi = Nk / len(x)
    print(it, round(loglik, 1), mu.round(2), var.round(2), pi.round(2))
