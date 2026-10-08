"""Main project — Training a small neural network with hand-written backpropagation — starter code (as printed in the book)

From Numbers to Thought — Chapter 5
"""
import numpy as np
from sklearn.datasets import make_moons
 
X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.5, (16, 2)); b1 = np.zeros(16)
w2 = rng.normal(0, 0.5, 16);      b2 = 0.0
sigmoid = lambda z: 1 / (1 + np.exp(-z))
 
def forward(X):
    H = np.tanh(X @ W1.T + b1)            # N x 16
    p = sigmoid(H @ w2 + b2)              # N
    return H, p
