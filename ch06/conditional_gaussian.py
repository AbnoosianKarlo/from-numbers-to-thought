"""Listing 6.2 — Verifying Gaussian conditioning by sampling

From Numbers to Thought — Chapter 6: The Logic of Uncertainty: Probability and Statistics for Learning Machines
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
rng = np.random.default_rng(1)
mu = np.array([1.0, -1.0]); Sigma = np.array([[0.5, -0.6], [-0.6, 2.0]])
X = rng.multivariate_normal(mu, Sigma, size=2_000_000)
 
near = np.abs(X[:, 1] - 1.0) < 0.01               # condition on x2 ~ 1
x1 = X[near, 0]
print(x1.mean(), x1.var())                        # -> ~0.40, ~0.32
