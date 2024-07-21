import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

class BasisEncoder(BaseEncoder):
    ...

class AmplitudeEncoder(BaseEncoder):
    def transform(self, X):
        return X / np.linalg.norm(X)
    
class RepeatedAmplitudeEncoder(BaseEncoder):
    ...

class AngleEncoder(BaseEncoder):
    ...

class CoherentStateEncoder(BaseEncoder):
    ...

class TimeEvolutionEncoder(BaseEncoder):
    ...
