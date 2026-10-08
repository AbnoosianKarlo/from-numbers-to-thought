"""Main project — Disease diagnosis with SVMs — starter code (as printed in the book)

From Numbers to Thought — Chapter 12
"""
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
 
X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 1e-1, 1]}
inner = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc")
print(cross_val_score(inner, X, y, cv=5, scoring="roc_auc").mean())
