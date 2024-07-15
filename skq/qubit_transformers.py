import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.gates import Gate
from skq.utils import _check_quantum_state_array


class BaseQubitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gate: Gate):
        assert isinstance(gate, Gate), f"Gate must be a valid instance of Gate. Got '{type(gate)}'. For custom gates, inherit from the 'skq.gates.Gate' class or use 'skq.gates.CustomGate'."
        self.gate = gate

    def fit(self, X, y=None):
        _check_quantum_state_array(X)
        return self
    
    def transform(self, X) -> np.ndarray:
        _check_quantum_state_array(X)
        # Dot product of gate and input state vectors
        return np.array([self.gate @ x for x in X])

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
    