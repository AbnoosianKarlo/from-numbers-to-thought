"""Solution to Exercise 13.14 — Mini-batch SGD for the network of Listing 13.2

From Numbers to Thought — Chapter 13: Neural Networks from Scratch: From a Single Neuron to Backpropagation
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import time
import numpy as np
from sklearn.datasets import make_moons


def init(sizes, rng):
    return [(rng.normal(0, np.sqrt(2 / m), (n, m)), np.zeros((n, 1)))
            for m, n in zip(sizes[:-1], sizes[1:])]


def forward(params, X):
    A, cache = X, [X]
    for l, (W, b) in enumerate(params):
        Z = W @ A + b
        A = 1 / (1 + np.exp(-Z)) if l == len(params) - 1 else np.maximum(0, Z)
        cache.append(A)
    return A, cache


def backward(params, cache, Y):
    N = Y.shape[1]
    delta = cache[-1] - Y
    grads = []
    for l in reversed(range(len(params))):
        grads.append((delta @ cache[l].T / N, delta.mean(1, keepdims=True)))
        if l > 0:
            delta = (params[l][0].T @ delta) * (cache[l] > 0)
    return grads[::-1]


def loss(params, X, Y):
    P, _ = forward(params, X)
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def train(X, Y, batch_size, epochs=200, eta=0.1, seed=0):
    rng = np.random.default_rng(seed)
    params = init([2, 16, 16, 1], rng)
    N = X.shape[1]
    history, start = [], time.perf_counter()
    for epoch in range(epochs):
        perm = rng.permutation(N)                  # shuffle every epoch
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            _, cache = forward(params, X[:, idx])
            grads = backward(params, cache, Y[:, idx])
            params = [(W - eta * dW, b - eta * db)
                      for (W, b), (dW, db) in zip(params, grads)]
        history.append(loss(params, X, Y))
    return history, time.perf_counter() - start


if __name__ == "__main__":
    X, y = make_moons(400, noise=0.2, random_state=0)
    X, Y = X.T, y[None, :]
    for bs in (8, 32, 400):
        hist, seconds = train(X, Y, bs)
        print(f"batch={bs:3d}  loss@10={hist[9]:.3f}  loss@200={hist[-1]:.3f}"
              f"  time={seconds:.2f}s")
