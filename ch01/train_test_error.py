"""Listing 1.1 — Train/test split and training vs. test error

From Numbers to Thought — Chapter 1: From Data to Knowledge: A Roadmap of Machine Learning
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(200, 1))
y = 3 * X[:, 0] + 2 + rng.normal(0, 2, size=200)      # noisy linear data
 
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
model = LinearRegression().fit(X_tr, y_tr)              # learn from training set only
print("train MSE:", mean_squared_error(y_tr, model.predict(X_tr)))
print("test  MSE:", mean_squared_error(y_te, model.predict(X_te)))
