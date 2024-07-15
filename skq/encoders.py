import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.gates import Gate

class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self


class AmplitudeEncoder(BaseEncoder):
    def transform(self, X):
        return X / np.linalg.norm(X)
    
class GateTransformer(BaseEncoder):
    def __init__(self, gate: Gate):
        self.gate = gate

    def fit(self, X, y=None):
        assert self.gate.is_unitary(), "Gate must be unitary"
        return self
    
    def transform(self, X) -> np.ndarray:
        # Ensure every element is a 2 element array
        if len(X.shape) == 1:
            raise ValueError("Input must be a 2D array")
        
        for x in X:
            if len(x) != 2:
                raise ValueError("Input must be a 2D array with 2 elements in each row")

        # Ensure that inputs are normalized state vectors
        if not np.allclose(np.linalg.norm(X, axis=-1), 1):
            # Get vectors that are not normalized
            not_normalized = X[np.linalg.norm(X, axis=-1) != 1]
            raise ValueError(f"Input state vectors must be normalized. Got non-normalized vectors: '{not_normalized}'")
        
        # Apply dot product to every row
        return np.array([self.gate @ x for x in X])
    