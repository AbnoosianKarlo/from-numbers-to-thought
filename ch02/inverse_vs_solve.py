"""Listing 2.2 — Matrix inverse or direct solve?

From Numbers to Thought — Chapter 2: The Language of Vectors: Linear Algebra for Data and Models
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from scipy.linalg import hilbert
 
A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
print(np.round(np.linalg.inv(A)))                 # matches (2.58)
 
H = hilbert(12)                                  # notoriously ill-conditioned
x_true = np.ones(12); b = H @ x_true
x_inv = np.linalg.inv(H) @ b                     # avoid in practice
x_sol = np.linalg.solve(H, b)                    # LU factorization
print(np.linalg.norm(H @ x_inv - b), np.linalg.norm(H @ x_sol - b))
