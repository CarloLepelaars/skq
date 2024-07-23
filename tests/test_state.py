import pytest
import qiskit
import numpy as np
import pennylane as qml

from skq.state import Statevector
from skq.density import DensityMatrix

def test_statevector_initialization():
    # |00>
    state_vector = Statevector([1, 0, 0, 0])
    assert isinstance(state_vector, Statevector)
    assert state_vector.is_normalized()

    # Bell state (|00> + |11>) / sqrt(2)
    state_vector = Statevector([1, 0, 0, 1] / np.sqrt(2))
    assert isinstance(state_vector, Statevector)
    assert state_vector.is_normalized()
    
    # Non-normalized state vector
    with pytest.raises(AssertionError):
        Statevector([1, 1])

    # Non-power of two state vector
    with pytest.raises(AssertionError):
        Statevector([1, 0, 0])

    # State vector not 1D (like a matrix)
    with pytest.raises(AssertionError):
        Statevector([[1, 0], [0, 1]])

def test_density_matrix():
    # |0><0|
    state_vector = Statevector([1, 0])
    density_matrix = state_vector.density_matrix()
    assert isinstance(density_matrix, DensityMatrix)
    expected_density_matrix = np.array([[1, 0], [0, 0]])
    np.testing.assert_array_almost_equal(density_matrix, expected_density_matrix)

def test_measure():
    # |00>
    state_vector = Statevector([1, 0, 0, 0])
    measured_state = state_vector.measure_index()
    assert measured_state == 0
    bitstring = state_vector.measure_bitstring()
    assert bitstring == "00"

    # Bell state (|00> + |11>) / sqrt(2)
    state_vector = Statevector([1, 0, 0, 1] / np.sqrt(2))
    measured_state = state_vector.measure_index()
    assert measured_state in [0, 3]
    bitstring = state_vector.measure_bitstring()
    assert bitstring in ["00", "11"]

def test_to_qiskit():
    # |00> in big-endian
    state_vector = Statevector([1, 0, 0, 0])
    qiskit_sv = state_vector.to_qiskit()
    assert isinstance(qiskit_sv, qiskit.quantum_info.Statevector)
    np.testing.assert_array_almost_equal(qiskit_sv.data, state_vector[::-1])

def test_from_qiskit():
    # |00> in little-endian
    qiskit_sv = qiskit.quantum_info.Statevector([0, 0, 0, 1])
    state_vector = Statevector.from_qiskit(qiskit_sv)
    assert isinstance(state_vector, Statevector)
    np.testing.assert_array_almost_equal(state_vector, qiskit_sv.data[::-1])

def test_to_pennylane():
    # |00> in big-endian
    state_vector = Statevector([1, 0, 0, 0])
    pennylane_sv = state_vector.to_pennylane()
    assert isinstance(pennylane_sv, qml.QubitStateVector)
    np.testing.assert_array_almost_equal(pennylane_sv.data[0], state_vector)

def test_from_pennylane():
    # |00> in big-endian
    pennylane_sv = qml.QubitStateVector([1, 0, 0, 0], wires=range(2))
    state_vector = Statevector.from_pennylane(pennylane_sv)
    assert isinstance(state_vector, Statevector)
    np.testing.assert_array_almost_equal(state_vector, pennylane_sv.data[0])
