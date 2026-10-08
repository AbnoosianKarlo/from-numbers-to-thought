"""Main project — Comparing optimizers for training a neural network — starter code (as printed in the book)

From Numbers to Thought — Chapter 7
"""
def cosine_schedule(t, T, eta_max=1e-3, eta_min=1e-5, warmup=50):
    if t < warmup:
        return eta_max * t / warmup                 # linear warm-up
    s = (t - warmup) / (T - warmup)
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * s))
