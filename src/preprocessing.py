import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def clean_text_column(X):
    text = X.iloc[:, 0].fillna("").astype(str).str.lower()
    text = text.str.replace(r"[^a-z\s]", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text


class Winsorize(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bound_ = X.quantile(self.lower)
        self.upper_bound_ = X.quantile(self.upper)
        self.columns_ = X.columns
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bound_", "upper_bound_"])
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_bound_, upper=self.upper_bound_, axis=1)

    def get_feature_names_out(self, input_features=None):
        names = self.columns_ if input_features is None else input_features
        return np.asarray([str(n) for n in names], dtype=object)   