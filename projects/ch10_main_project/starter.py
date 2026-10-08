"""Main project — Dimensionality reduction of MNIST images — starter code (as printed in the book)

From Numbers to Thought — Chapter 10
"""
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
 
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0; Xtr, Xte, ytr, yte = X[:60000], X[60000:], y[:60000], y[60000:]
for M in (10, 50, 100):
    pca = PCA(n_components=M).fit(Xtr)          # fit on training data only
    knn = KNeighborsClassifier(3).fit(pca.transform(Xtr), ytr)
    print(M, knn.score(pca.transform(Xte), yte))
