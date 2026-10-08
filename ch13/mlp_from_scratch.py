"""Listing 13.2 — A multilayer perceptron in pure NumPy

From Numbers to Thought — Chapter 13: Neural Networks from Scratch: From a Single Neuron to Backpropagation
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
from sklearn.datasets import make_moons

def init(sizes, rng):                   # He initialization
    return [(rng.normal(0, np.sqrt(2 / m), (n, m)), np.zeros((n, 1)))
            for m, n in zip(sizes[:-1], sizes[1:])]

def forward(params, X):                 # one example per column
    A, cache = X, [X]
    for l, (W, b) in enumerate(params):
        Z = W @ A + b
        last = l == len(params) - 1
        A = 1 / (1 + np.exp(-Z)) if last else np.maximum(0, Z)
        cache.append(A)
    return A, cache

def backward(params, cache, Y):
    N = Y.shape[1]
    delta = cache[-1] - Y               # sigmoid + cross-entropy
    grads = []
    for l in reversed(range(len(params))):
        grads.append((delta @ cache[l].T / N,
                      delta.mean(1, keepdims=True)))
        if l > 0:
            delta = (params[l][0].T @ delta) * (cache[l] > 0)
    return grads[::-1]

rng = np.random.default_rng(0)
X, y = make_moons(400, noise=0.2, random_state=0)
X, Y = X.T, y[None, :]
params = init([2, 16, 16, 1], rng)
for epoch in range(2001):
    P, cache = forward(params, X)
    grads = backward(params, cache, Y)
    params = [(W - 0.5 * dW, b - 0.5 * db)
              for (W, b), (dW, db) in zip(params, grads)]
    if epoch % 500 == 0:
        loss = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        print(epoch, round(loss, 3), round(((P > 0.5) == Y).mean(), 3))
