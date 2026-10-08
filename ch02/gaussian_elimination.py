"""Listing 2.1 — Solving a linear system and computing the reduced row-echelon form

From Numbers to Thought — Chapter 2: The Language of Vectors: Linear Algebra for Data and Models
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
import sympy as sp
 
A = np.array([[1, 1, 1], [1, -1, 1], [1, 2, -1]], dtype=float)
b = np.array([6, 2, 2], dtype=float)
print(np.linalg.solve(A, b))                  # -> [1. 2. 3.]  system (2.5)
 
aug = sp.Matrix([[1, 2, -1, 1, 3],            # augmented matrix of (2.44), a = -1
                  [2, 4, -1, 3, 8],
                  [-1, -2, 2, 0, -1]])
R, pivots = aug.rref()                          # exact reduced row-echelon form
print(R, pivots)                               # pivot columns: (0, 2)
