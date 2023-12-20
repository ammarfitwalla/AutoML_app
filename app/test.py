import pandas as pd
from autofeat import FeatureSelector
from sklearn.datasets import load_wine

X,y = load_wine(return_X_y=True)
fsel = FeatureSelector(verbose=1)
new_X = fsel.fit_transform(pd.DataFrame(X), pd.DataFrame(y))
print(new_X.columns.tolist())
print(pd.DataFrame(X).columns.tolist())
