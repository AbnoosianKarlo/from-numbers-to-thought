"""Listing 5.1 — Gradient checking with finite differences

From Numbers to Thought — Chapter 5: The Calculus of Change: Gradients, Backpropagation and the Engine of Learning
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
f = lambda x: x[0]**3 * x[1] + 2 * x[0] * x[1]**2
grad = lambda x: np.array([3*x[0]**2*x[1] + 2*x[1]**2, x[0]**3 + 4*x[0]*x[1]])
 
def numerical_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)   # central difference
    return g
 
x = np.array([1.5, -0.7])
print(grad(x), numerical_grad(f, x))         # should agree to ~1e-9
