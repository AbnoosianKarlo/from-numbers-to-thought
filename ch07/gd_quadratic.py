"""Listing 7.1 — Gradient descent on the quadratic of Example 7.1

From Numbers to Thought — Chapter 7: The Art of Optimization: From Gradient Descent to Adam and Duality
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
Q = np.array([[3, 1], [1, 12]]); c = np.array([4, 2])
grad = lambda x: Q @ x - c                     # gradient of (7.7)
 
x, gamma = np.array([-3.0, -1.5]), 0.13
for k in range(50):
    x = x - gamma * grad(x)                    # update (7.6)
print(x, np.linalg.solve(Q, c))               # -> both ~ [1.314, 0.057]
