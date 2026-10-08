"""Listing 1.2 — Reproducing Figure 1.3: polynomial fits of increasing degree

From Numbers to Thought — Chapter 1: From Data to Knowledge: A Roadmap of Machine Learning
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)                  # the unknown truth
x_tr = rng.uniform(0, 1, 12); y_tr = f(x_tr) + rng.normal(0, 0.2, 12)
x_te = rng.uniform(0, 1, 10); y_te = f(x_te) + rng.normal(0, 0.2, 10)
 
for degree in (1, 3, 11):
    coef = np.polyfit(x_tr, y_tr, degree)           # least-squares fit
    err = lambda x, y: np.mean((np.polyval(coef, x) - y) ** 2)
    print(degree, err(x_tr, y_tr), err(x_te, y_te))
