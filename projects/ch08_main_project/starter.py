"""Main project — A complete, trustworthy ML pipeline — starter code (as printed in the book)

From Numbers to Thought — Chapter 8
"""
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
 
X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
inner = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=5)
outer = cross_val_score(inner, X, y, cv=5, scoring="recall")   # nested CV
print(outer.mean(), outer.std())
