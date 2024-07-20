import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.gates.qubit.single import QubitGate
from skq.utils import _check_quantum_state_array


class BaseQubitTransformer(BaseEstimator, TransformerMixin):
    """ 
    Scikit-learn transformer for quantum state vectors.
    :param gate: A valid skq quantum Gate object
    :param qubits: List of qubit indices to apply the gate to.
    """
    def __init__(self, gate: QubitGate, *, qubits: list[int]):
        assert isinstance(gate, QubitGate), f"Gate must be a valid instance of Gate. Got '{type(gate)}'. For custom gates, inherit from the 'skq.gates.QubitGate' class or use 'skq.gates.QubitCustomGate'."
        self.gate = gate
        assert isinstance(qubits, list), f"Qubits must be a list of integers. Got '{type(qubits)}'."
        assert all(isinstance(q, int) and q >= 0 for q in qubits), f"Qubits must be a list of non-negative integers. Got '{qubits}'."
        self.qubits = qubits

    def fit(self, X, y=None):
        _check_quantum_state_array(X)
        return self
    
    def transform(self, X) -> np.ndarray:
        _check_quantum_state_array(X)
        # Dot product of gate and input state vectors
        return np.array([self.gate @ x for x in X])
    