from sklearn.base import BaseEstimator, TransformerMixin


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        ...

    def transform(self, X):
        ...

class BasisEncoder(BaseEncoder):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        ...

    def transform(self, X):
        binary_strings = []
        for row in X:
            binary_strings.append(self._encode(row))

    def _encode(self, row):
        ...