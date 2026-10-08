"""Listing 10.1 — PCA by hand and with a library

From Numbers to Thought — Chapter 10: The Essence of Data: Dimensionality Reduction and PCA
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from sklearn.decomposition import PCA
 
X = np.array([[0, 0], [2, 2], [3, 1], [1, 3]], dtype=float)
Xc = X - X.mean(axis=0)
S = Xc.T @ Xc / len(X)                      # covariance (1/N convention)
lam, B = np.linalg.eigh(S)                  # ascending eigenvalues
print(lam[::-1], B[:, ::-1])                # -> [1.5 1.0]
 
pca = PCA(n_components=1).fit(X)
print(pca.explained_variance_ratio_)         # -> [0.6]
print(pca.transform(X).ravel())             # codes, up to sign
