import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.gates import Gate


class BaseQubitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gate: Gate):
        assert isinstance(gate, Gate), f"Gate must be a valid instance of Gate. Got '{type(gate)}'. For custom gates, inherit from the 'skq.gates.Gate' class or use 'skq.gates.CustomGate'."
        self.gate = gate

    def fit(self, X, y=None):
        self._check_array(X)
        return self
    
    def transform(self, X) -> np.ndarray:
        self._check_array(X)
        # Dot product of gate and input state vectors
        return np.array([self.gate @ x for x in X])
    
    def _check_array(self, X):
        """ Ensure that input are normalized state vectors of the right size."""
        # Check if input is a 2D array.
        if len(X.shape) != 2:
            raise ValueError("Input must be a 2D array")
        
        # Check if all inputs are complex
        if not np.iscomplexobj(X):
            raise ValueError("Input must be a complex array.")

        # Check if input is normalized.
        normalized = np.allclose(np.linalg.norm(X, axis=-1), 1)
        if not normalized:
            not_normalized = X[np.linalg.norm(X, axis=-1) != 1]
            raise ValueError(f"Input state vectors must be normalized. Got non-normalized vectors: '{not_normalized}'")
        
        # Check if array is has correct number of elements using the gate.
        elements_needed = 2 ** self.gate.num_qubits()
        if X.shape[1] != elements_needed:
            raise ValueError(f"Input must be a 2D array with {elements_needed} elements in each row")
        return True

class SingleQubitTransformer(BaseQubitTransformer):
    """
    Transformer using a single qubit gate to transform input state vectors
    """
    def __init__(self, gate: Gate):
        assert gate.shape == (2, 2), "Single Qubit Gate must be a 2x2 matrix"
        super().__init__(gate)
    
class MultiQubitTransformer(BaseQubitTransformer):
    """ Transformer which involves multiple qubits, like CNOT, SWAP, etc. """
    def __init__(self, gate: Gate):
        assert gate.shape[0] >= 4 and gate.shape[1] >= 4, "Multi Qubit Gate must be a matrix of at least 4x4"
        super().__init__(gate)
    