"""Listing 9.2 — Bayesian linear regression: posterior and predictive distribution

From Numbers to Thought — Chapter 9: Continuous Prediction: Regression from Least Squares to the Bayesian View
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
import numpy as np
 
def posterior(Phi, y, sigma2, S0):
    SN = np.linalg.inv(np.linalg.inv(S0) + Phi.T @ Phi / sigma2)   # posterior covariance
    mN = SN @ Phi.T @ y / sigma2                                    # posterior mean, m0 = 0
    return mN, SN
 
def predict(phi_star, mN, SN, sigma2):
    mean = phi_star @ mN
    var = np.sum(phi_star @ SN * phi_star, axis=1) + sigma2         # parameter + noise variance
    return mean, var
