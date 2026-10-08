"""Listing 10.2 — Compressing and reconstructing digits with PCA

From Numbers to Thought — Chapter 10: The Essence of Data: Dimensionality Reduction and PCA
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
 
X, y = load_digits(return_X_y=True)           # 1797 images, 64 pixels
for M in (1, 10, 21, 40):
    pca = PCA(n_components=M).fit(X)
    X_rec = pca.inverse_transform(pca.transform(X))
    err = np.mean(np.sum((X - X_rec) ** 2, axis=1))
    print(M, round(pca.explained_variance_ratio_.sum(), 3), round(err, 1))
