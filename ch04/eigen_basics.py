"""Listing 4.1 — Determinants, eigenvalues and eigenvectors with NumPy

From Numbers to Thought — Chapter 4: Anatomy of Matrices: Eigenvalues, SVD and Hidden Structure
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
A3 = np.array([[2, 1, 3], [0, 4, 1], [1, 0, 2]])
print(np.linalg.det(A3))                    # -> 5.0  (example 4.3)
 
A = np.array([[5, 2], [2, 2]])
w, V = np.linalg.eigh(A)                    # symmetric -> eigh
print(w)                                    # -> [1. 6.]
print(V)                                    # columns ~ [1,-2]/sqrt5, [2,1]/sqrt5
print(np.allclose(A @ V, V * w))            # A v_i = lambda_i v_i
