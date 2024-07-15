import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skq.utils import _check_quantum_state_array

class MeasurementTransformer(BaseEstimator, TransformerMixin):
    """
    Sample measurement outcomes from the state vector amplitudes.
    :param repeat: Number of times to repeat the measurement
    """
    def __init__(self, repeat: int = 1):
        self.repeat = repeat

    def fit(self, X, y=None):
        _check_quantum_state_array(X)
        return self

    def transform(self, X):
        _check_quantum_state_array(X)
        _, n_states = X.shape
        num_qubits = int(np.log2(n_states))
        assert n_states == 2**num_qubits, f"Expected state vector length {2**num_qubits}, got {n_states}"

        probabilities = np.abs(X)**2
        sampled_indices = [np.random.choice(2**num_qubits, p=prob) for prob in probabilities]
        
        # Convert sampled indices to binary representation
        for _ in range(self.repeat - 1):
            sampled_indices = np.append(sampled_indices, [np.random.choice(2**num_qubits, p=prob) for prob in probabilities])

        measurements = np.array([list(np.binary_repr(index, width=num_qubits)) for index in sampled_indices], dtype=int)
        return measurements