"""Listing 8.1 — Maximum likelihood estimation for the data of Table 8.2

From Numbers to Thought — Chapter 8: The Learning Framework: Data, Models and Inference
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
age = np.array([36, 47, 26, 68, 33.])
salary = np.array([89.563, 123.543, 23.989, 138.769, 113.888])   # thousands
X = np.c_[np.ones_like(age), age]                # augmented features [1, x]
 
theta, *_ = np.linalg.lstsq(X, salary, rcond=None)
sigma2 = np.mean((salary - X @ theta) ** 2)      # ML estimate of noise variance
print(theta, sigma2)                             # -> [8.81 2.12]  655.5
print(theta @ [1, 60])                           # prediction for age 60: ~136.2
