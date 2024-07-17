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
        """
        Checks that inputs are valid quantum state vectors.
        :param X: Input state vectors
        :param y: Only there for compatibility with scikit-learn.
        """
        _check_quantum_state_array(X)
        return self

    def transform(self, X) -> np.array:
        """
        Perform measurement on the input state vectors.
        :param X: Input state vectors
        :return: Measurement outcomes in binary representation.
        Output is a 2D numpy array where each row represents a measurement outcome. 
        Length of each row corresponds to the number of qubits.
        """
        _check_quantum_state_array(X)

        # Sample from probabilities of the state vector amplitudes squared to simulate measurement.
        probabilities = np.abs(X)**2
        num_qubits = int(np.log2(probabilities.shape[1]))
        sampled_indices = np.hstack([
            np.random.choice(2**num_qubits, size=self.repeat, p=prob) for prob in probabilities
        ])
        measurements = np.array(
            [list(np.binary_repr(index, width=num_qubits)) for index in sampled_indices], dtype=int
        )        
        return measurements