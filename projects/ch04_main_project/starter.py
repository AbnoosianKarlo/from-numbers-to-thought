"""Main project — Restoring a damaged image with low-rank approximation — starter code (as printed in the book)

From Numbers to Thought — Chapter 4
"""
import numpy as np
 
def soft_impute(Y, mask, lam, iters=150):
    X = np.where(mask, Y, Y[mask].mean())         # initial fill
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U * np.maximum(s - lam, 0)) @ Vt     # shrink singular values
        X = np.where(mask, Y, L)                  # keep observed pixels
    return L
