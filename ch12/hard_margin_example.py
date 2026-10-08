"""Listing 12.1 — Reproducing Example 12.2 with scikit-learn

From Numbers to Thought — Chapter 12: Decision Boundaries: Support Vector Machines and Kernel Methods
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from sklearn.svm import SVC

X = np.array([[1, 2], [2, 0], [0, 1], [4, 2], [4, 4], [6, 1]])
y = np.array([-1, -1, -1, 1, 1, 1])
svm = SVC(kernel="linear", C=1e6).fit(X, y)   # very large C: hard margin
w, b = svm.coef_[0], svm.intercept_[0]
print(w.round(3), b.round(3))   # [0.667 0.333] -2.333
print(svm.support_)   # [0 1 3]: x1, x2, x4
print(np.abs(svm.dual_coef_).round(3))   # [[0.111 0.167 0.278]]
print(1 / np.linalg.norm(w))   # 1.342 = 3 / sqrt(5)
