"""Listing 12.3 — Tuning C and gamma by grid search

From Numbers to Thought — Chapter 12: Decision Boundaries: Support Vector Machines and Kernel Methods
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=400, noise=0.25, random_state=7)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]}
search = GridSearchCV(pipe, grid, cv=5).fit(Xtr, ytr)
print(search.best_params_, round(search.best_score_, 3))
print(round(search.score(Xte, yte), 3))   # accuracy on held-out data
print(search.best_estimator_[-1].n_support_)   # support vectors per class
