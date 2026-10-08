"""Listing 7.2 — Solving linear program (7.44) with SciPy

From Numbers to Thought — Chapter 7: The Art of Optimization: From Gradient Descent to Adam and Duality
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from scipy.optimize import linprog
 
c = np.array([-4, -3])                          # minimize -[4,3]^T x
A = np.array([[1, 1], [2, 1], [-1, 0], [0, -1], [0, 1]])
b = np.array([8, 12, 0, 0, 6])
res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 2)
print(res.x, res.fun)                          # -> [4. 4.] -28.0
