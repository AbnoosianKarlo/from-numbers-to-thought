"""Listing 13.3 — A neural network for handwritten digits with scikit-learn

From Numbers to Thought — Chapter 13: Neural Networks from Scratch: From a Single Neuron to Backpropagation
https://github.com/AbnoosianKarlo/from-numbers-to-thought
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)     # 1797 images, 8 x 8
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)
net = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    alpha=1e-3,         # weight decay
                    early_stopping=True, random_state=0, max_iter=500)
model = make_pipeline(StandardScaler(), net).fit(Xtr, ytr)
print(net.n_iter_, round(model.score(Xtr, ytr), 3),
      round(model.score(Xte, yte), 3))
n_par = sum(W.size for W in net.coefs_)
print(n_par + sum(b.size for b in net.intercepts_))
