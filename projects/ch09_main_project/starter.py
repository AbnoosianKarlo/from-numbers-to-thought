"""Main project — Building a house-price prediction model — starter code (as printed in the book)

From Numbers to Thought — Chapter 9
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
 
X, y = fetch_california_housing(return_X_y=True)   # downloads on first use
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LassoCV(cv=5))
model.fit(Xtr, ytr); print("test R^2:", model.score(Xte, yte))
