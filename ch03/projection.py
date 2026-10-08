"""Listing 3.2 — Orthogonal projection and least squares

From Numbers to Thought — Chapter 3: The Geometry of Similarity: Distance, Angle and Projection
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
B = np.array([[1., 1.], [0., 1.], [1., 0.]])   # basis of U in the columns
x = np.array([2., 2., 3.])
 
lam = np.linalg.solve(B.T @ B, B.T @ x)          # normal equation (3.61)
proj = B @ lam                                   # -> [3. 1. 2.]
P = B @ np.linalg.inv(B.T @ B) @ B.T             # projection matrix (3.64)
 
lam_ls, *_ = np.linalg.lstsq(B, x, rcond=None)   # least squares: same answer
print(lam, lam_ls, np.allclose(P @ P, P), B.T @ (x - proj))
