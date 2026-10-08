"""Main project — Face retrieval in feature space — starter code (as printed in the book)

From Numbers to Thought — Chapter 3
"""
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
 
faces = fetch_olivetti_faces()            # downloads the data on first use
X, y = faces.data, faces.target           # X: 400 x 4096
gallery = np.hstack([np.where(y == p)[0][:7] for p in range(40)])
query   = np.hstack([np.where(y == p)[0][7:] for p in range(40)])
d = np.linalg.norm(X[query][:, None] - X[gallery][None], axis=2)
pred = y[gallery][d.argmin(axis=1)]
print("Euclidean accuracy:", np.mean(pred == y[query]))
