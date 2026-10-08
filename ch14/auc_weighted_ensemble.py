"""Listing 14.1 — AUC-weighted ensemble of classifiers on a public dataset

From Numbers to Thought — Chapter 14: From Classroom to Clinic: Machine Learning in Medicine
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                      stratify=y, random_state=0)
models = [make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
          DecisionTreeClassifier(max_depth=4, random_state=0),
          RandomForestClassifier(200, random_state=0),
          GaussianNB()]
weights, probs = [], []
for m in models:
    inner = cross_val_predict(m, Xtr, ytr, cv=5,
                              method="predict_proba")[:, 1]
    weights.append(roc_auc_score(ytr, inner))   # weight = inner AUC
    probs.append(m.fit(Xtr, ytr).predict_proba(Xte)[:, 1])
w, P = np.array(weights), np.array(probs)
ensemble = w @ P / w.sum()                       # AUC-weighted average
print(w.round(3))
print([round(roc_auc_score(yte, p), 3) for p in P])
print(round(roc_auc_score(yte, ensemble), 3))
