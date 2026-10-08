"""Project 1 — A small disease-diagnosis system — starter code (as printed in the book)

From Numbers to Thought — Chapter 1
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
 
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_tr)             # fit on training data only!
model = LogisticRegression().fit(scaler.transform(X_tr), y_tr)
y_hat = model.predict(scaler.transform(X_te))
print(accuracy_score(y_te, y_hat), recall_score(y_te, y_hat))
