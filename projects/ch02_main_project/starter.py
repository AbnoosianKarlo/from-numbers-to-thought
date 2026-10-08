"""Main project — Image processing with linear algebra only — starter code (as printed in the book)

From Numbers to Thought — Chapter 2
"""
import numpy as np
from sklearn.datasets import load_sample_image
 
img = load_sample_image("china.jpg") / 255.0      # H x W x 3
w = np.array([0.299, 0.587, 0.114])
gray = img @ w                                     # linear map R^3 -> R
H, W = gray.shape
J = np.eye(W)[::-1]                                # anti-diagonal permutation
flipped = gray @ J                                 # horizontal flip
print(gray.shape, np.linalg.matrix_rank(gray))
