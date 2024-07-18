import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.gates.single_qubit import Gate
from skq.utils import _check_quantum_state_array


class BaseQubitTransformer(BaseEstimator, TransformerMixin):
    """ 
    Scikit-learn transformer for quantum state vectors.
    :param gate: A valid skq quantum Gate object
    :param qubits: List of qubit indices to apply the gate to.
    """
    def __init__(self, gate: Gate, *, qubits: int | list[int]):
        assert isinstance(gate, Gate), f"Gate must be a valid instance of Gate. Got '{type(gate)}'. For custom gates, inherit from the 'skq.gates.Gate' class or use 'skq.gates.CustomGate'."
        self.gate = gate
        self.qubits = qubits

    def fit(self, X, y=None):
        _check_quantum_state_array(X)
        return self
    
    def transform(self, X) -> np.ndarray:
        _check_quantum_state_array(X)
        # Dot product of gate and input state vectors
        return np.array([self.gate @ x for x in X])
    