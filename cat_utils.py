# CatBoost input: strings stay strings (NaN -> "NA"), numbers stay numbers
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CatPrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cat_:
            X[c] = X[c].astype(str).fillna("NA")
        return X
