"""Main project — Naive Bayes for text classification — starter code (as printed in the book)

From Numbers to Thought — Chapter 6
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
 
cats = ["sci.space", "rec.autos"]
train = fetch_20newsgroups(subset="train", categories=cats)
vec = CountVectorizer(min_df=2)
X = vec.fit_transform(train.data).toarray(); y = train.target
log_prior = np.log(np.bincount(y) / len(y))
counts = np.array([X[y == c].sum(0) for c in (0, 1)]) + 1      # Laplace
log_lik = np.log(counts / counts.sum(1, keepdims=True))
