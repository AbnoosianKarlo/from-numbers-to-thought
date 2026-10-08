"""Listing 6.1 — Simulating the two-coin experiment

From Numbers to Thought — Chapter 6: The Logic of Uncertainty: Probability and Statistics for Learning Machines
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
rng = np.random.default_rng(0)
draws = rng.random((100_000, 2)) < 0.4        # True = gold coin
X = draws.sum(axis=1)                           # number of gold coins
 
for k, p in zip((0, 1, 2), (0.36, 0.48, 0.16)):
    print(k, round(np.mean(X == k), 4), p)    # empirical vs exact
