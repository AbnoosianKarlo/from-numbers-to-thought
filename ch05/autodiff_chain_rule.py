"""Listing 5.2 — Automatic differentiation with PyTorch

From Numbers to Thought — Chapter 5: The Calculus of Change: Gradients, Backpropagation and the Engine of Learning
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import torch
 
t = torch.tensor(0.8, requires_grad=True)
x1, x2 = t**2, 3*t                            # x = g(t)
h = x1**2 * x2 + torch.sin(x2)                 # h = f(g(t))
h.backward()                                   # reverse-mode autodiff
 
exact = 15 * 0.8**4 + 3 * torch.cos(torch.tensor(3 * 0.8))
print(t.grad.item(), exact.item())             # identical up to float error
