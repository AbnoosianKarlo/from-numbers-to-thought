"""Listing 4.2 — Computing the SVD and its link to the eigendecomposition

From Numbers to Thought — Chapter 4: Anatomy of Matrices: Eigenvalues, SVD and Hidden Structure
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
A = np.array([[1, 1, 0], [1, -1, 2]], dtype=float)
U, s, Vt = np.linalg.svd(A)
print(s)                                     # -> [2.449 1.414] = sqrt(6), sqrt(2)
print(np.sqrt(np.linalg.eigvalsh(A.T @ A))[::-1][:2])
 
Sigma = np.zeros_like(A); Sigma[:2, :2] = np.diag(s)
print(np.allclose(U @ Sigma @ Vt, A))        # reconstruct A
