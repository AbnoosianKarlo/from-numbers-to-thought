"""Listing 13.1 — Checking the gradients of Example 13.5 numerically

From Numbers to Thought — Chapter 13: Neural Networks from Scratch: From a Single Neuron to Backpropagation
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np

def loss(theta, x, y):                  # network of Example 13.3
    W1, b1 = theta[:4].reshape(2, 2), theta[4:6]
    W2, b2 = theta[6:8], theta[8]
    a1 = np.maximum(0, W1 @ x + b1)
    p = 1 / (1 + np.exp(-(W2 @ a1 + b2)))
    return -np.log(p) if y == 1 else -np.log(1 - p)

theta = np.array([1, -1, 0.5, 0.5, 0, -1, 2, -2, -1.0])
x, y, eps = np.array([2.0, 1.0]), 1, 1e-5
num = np.zeros_like(theta)
for i in range(theta.size):             # central differences
    e = np.zeros_like(theta); e[i] = eps
    d = loss(theta + e, x, y) - loss(theta - e, x, y)
    num[i] = d / (2 * eps)
print(num.round(4))  # [-2. -1. 2. 1. -1. 1. -0.5 -0.25 -0.5]
