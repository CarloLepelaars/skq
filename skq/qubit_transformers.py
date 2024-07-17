import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.gates import Gate
from skq.utils import _check_quantum_state_array


class BaseQubitTransformer(BaseEstimator, TransformerMixin):
    """ 
    Scikit-learn transformer for quantum state vectors.
    :param gate: A valid skq quantum Gate object
    :param qubits: List of qubit indices to apply the gate to.
    """
    def __init__(self, gate: Gate, qubits: int | list):
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

class SingleQubitTransformer(BaseQubitTransformer):
    """
    Transformer using a single qubit gate to transform input state vectors.
    :param gate: A valid skq quantum Gate object
    :param qubits: The qubit index to apply the gate to.
    """
    def __init__(self, gate: Gate, qubits: int):
        assert gate.shape == (2, 2), "Single Qubit Gate must be a 2x2 matrix"
        assert isinstance(qubits, int), "Single Qubit Transformer must have a single qubit integer index."
        super().__init__(gate=gate, qubits=qubits)
    
class MultiQubitTransformer(BaseQubitTransformer):
    """ 
    Transformer which involves multiple qubits, like CNOT, SWAP, etc. 
    :param gate: A valid skq quantum Gate object
    :param qubits: List of qubit indices to apply the gate to.
    """
    def __init__(self, gate: Gate, qubits: list):
        assert gate.shape[0] >= 4 and gate.shape[1] >= 4, "Multi Qubit Gate must be a matrix of at least 4x4"
        super().__init__(gate=gate, qubits=qubits)
        assert isinstance(qubits, list), "Multi Qubit Transformer must have a list of qubit indices."
        assert len(qubits) == gate.num_qubits(), f"Number of qubits in gate ({gate.num_qubits()}) must match the number of defined qubits in MultiQubitTransformer ({len(qubits)})."
    