import pytest
import qiskit
import numpy as np

from skq.state import Statevector
from skq.density import DensityMatrix

def test_statevector_initialization():
    state_vector = Statevector([1, 0, 0])
    assert isinstance(state_vector, Statevector)
    assert state_vector.is_normalized()

    # Non-normalized state vector
    with pytest.raises(AssertionError):
        Statevector([1, 1, 1])

def test_density_matrix():
    state_vector = Statevector([1, 0, 0])
    density_matrix = state_vector.density_matrix()
    assert isinstance(density_matrix, DensityMatrix)
    expected_density_matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_almost_equal(density_matrix, expected_density_matrix)

def test_to_qiskit():
    state_vector = Statevector([1, 0, 0])
    qiskit_sv = state_vector.to_qiskit()
    assert isinstance(qiskit_sv, qiskit.quantum_info.Statevector)
    np.testing.assert_array_almost_equal(qiskit_sv.data, state_vector)

def test_from_qiskit():
    qiskit_sv = qiskit.quantum_info.Statevector([1, 0, 0])
    state_vector = Statevector.from_qiskit(qiskit_sv)
    assert isinstance(state_vector, Statevector)
    np.testing.assert_array_almost_equal(state_vector, qiskit_sv.data)
