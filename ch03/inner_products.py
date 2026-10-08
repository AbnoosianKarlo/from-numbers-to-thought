"""Listing 3.1 — Length, distance and angle under an arbitrary inner product

From Numbers to Thought — Chapter 3: The Geometry of Similarity: Distance, Angle and Projection
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
def inner(x, y, A=None):
    A = np.eye(len(x)) if A is None else A
    return x @ A @ y                              # <x, y> = x^T A y
 
def angle(x, y, A=None):
    c = inner(x, y, A) / np.sqrt(inner(x, x, A) * inner(y, y, A))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
 
x, y = np.array([1., 2.]), np.array([2., -1.])
print(angle(x, y))                                # 90.0   (dot product)
print(angle(x, y, np.diag([1., 3.])))             # 114.79 (Example 3.7)
