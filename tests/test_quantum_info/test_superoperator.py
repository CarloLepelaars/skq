import numpy as np

from src.quantum_info.superoperator import SuperOperator


def test_superoperator():
    # Example Choi matrix
    arr = np.array([[1, 0, 0, 0], 
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=complex) / 2
    matrix = SuperOperator(arr)

    assert matrix.dtype == complex, "Superoperator should be complex"
    assert matrix.is_power_of_n_shape(2), "Superoperator shape should be a power of 2"
    assert matrix.is_at_least_nxn(4), "Superoperator should be at least 4x4"
