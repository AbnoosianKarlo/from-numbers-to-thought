"""Main project — Detecting suspicious transactions with GMMs — starter code (as printed in the book)

From Numbers to Thought — Chapter 11
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
 
rng = np.random.default_rng(0)
normal = np.vstack([rng.normal([3, 12], [0.5, 2], (6000, 2)),     # daily purchases
                    rng.normal([6, 20], [0.4, 1], (2000, 2))])    # monthly payments
fraud = rng.normal([7.5, 3], [0.6, 1], (40, 2))                   # large, at night
X = np.vstack([normal, fraud]); y = np.r_[np.zeros(len(normal)), np.ones(40)]
sc = StandardScaler().fit(normal[:6000])
gmm = GaussianMixture(2, random_state=0).fit(sc.transform(normal[:6000]))
score = -gmm.score_samples(sc.transform(X))                        # high = anomalous
