import pytest
import qiskit
import numpy as np
import pennylane as qml

from skq.quantum_info.state import *
from skq.quantum_info import DensityMatrix


def test_statevector_initialization():
    # |00>
    state_vector = Statevector([1, 0, 0, 0])
    assert isinstance(state_vector, Statevector)
    assert state_vector.dtype == complex
    assert state_vector.is_normalized()

    # Bell state (|00> + |11>) / sqrt(2)
    state_vector = Statevector([1, 0, 0, 1] / np.sqrt(2))
    assert isinstance(state_vector, Statevector)
    assert state_vector.dtype == complex
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

def test_conjugate_transpose():
    # Bell state (|00> + |11>) / sqrt(2)
    state_vector = Statevector([1, 0, 0, 1] / np.sqrt(2))
    complex_conjugate = state_vector.conjugate_transpose()
    expected_complex_conjugate = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_array_almost_equal(complex_conjugate, expected_complex_conjugate)

    # (A^H)^H = A
    double_conjugate = complex_conjugate.conjugate_transpose()
    np.testing.assert_array_almost_equal(double_conjugate, state_vector)

    # (A + B)^H = A^H + B^H
    A_plus_B_conjugate = (state_vector + state_vector).conjugate_transpose()
    A_conjugate_plus_B_conjugate = state_vector.conjugate_transpose() + state_vector.conjugate_transpose()
    np.testing.assert_array_almost_equal(A_plus_B_conjugate, A_conjugate_plus_B_conjugate)

    # (AB)^H = B^H A^H
    AB_conjugate = (state_vector @ state_vector).conjugate_transpose()
    BA_conjugate = state_vector.conjugate_transpose() @ state_vector.conjugate_transpose()
    np.testing.assert_array_almost_equal(AB_conjugate, BA_conjugate)

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

def test_bell_states_initialization():
    # Bell state |Φ+>
    # (|00> + |11>) / sqrt(2)
    phi_plus = PhiPlusState()
    assert isinstance(phi_plus, Statevector)
    np.testing.assert_array_almost_equal(phi_plus, [1, 0, 0, 1] / np.sqrt(2))
    
    # Bell state |Φ->
    # (|00> - |11>) / sqrt(2)
    phi_minus = PhiMinusState()
    assert isinstance(phi_minus, Statevector)
    np.testing.assert_array_almost_equal(phi_minus, [1, 0, 0, -1] / np.sqrt(2))
    
    # Bell state |Ψ+>
    # (|01> + |10>) / sqrt(2)
    psi_plus = PsiPlusState()
    assert isinstance(psi_plus, Statevector)
    np.testing.assert_array_almost_equal(psi_plus, [0, 1, 1, 0] / np.sqrt(2))
    
    # Bell state |Ψ->
    # (|01> - |10>) / sqrt(2)
    psi_minus = PsiMinusState()
    assert isinstance(psi_minus, Statevector)
    np.testing.assert_array_almost_equal(psi_minus, [0, 1, -1, 0] / np.sqrt(2))

def test_ghz_state_initialization():
    # GHZ state |000> + |111> for 3 qubits
    ghz_state = GHZState(3)
    assert isinstance(ghz_state, Statevector)
    assert ghz_state.is_normalized()
    expected_state = np.zeros(2**3)
    expected_state[0] = 1 / np.sqrt(2)
    expected_state[-1] = 1 / np.sqrt(2)
    np.testing.assert_array_almost_equal(ghz_state, expected_state)
    
def test_w_state_initialization():
    # W state |001> + |010> + |100> for 3 qubits
    w_state = WState(3)
    assert isinstance(w_state, Statevector)
    assert w_state.is_normalized()
    expected_state = np.zeros(2**3)
    for i in range(3):
        expected_state[2**i] = 1 / np.sqrt(3)
    np.testing.assert_array_almost_equal(w_state, expected_state)

def test_specific_state_density_matrix():
    # PhiPlusState
    phi_plus = PhiPlusState()
    density_matrix = phi_plus.density_matrix()
    expected_density_matrix = np.outer([1, 0, 0, 1] / np.sqrt(2), np.array([1, 0, 0, 1]).conj() / np.sqrt(2))
    np.testing.assert_array_almost_equal(density_matrix, expected_density_matrix)

    # GHZState for 3 qubits
    ghz_state = GHZState(3)
    density_matrix = ghz_state.density_matrix()
    expected_state = np.zeros(2**3)
    expected_state[0] = 1 / np.sqrt(2)
    expected_state[-1] = 1 / np.sqrt(2)
    expected_density_matrix = np.outer(expected_state, expected_state.conj())
    np.testing.assert_array_almost_equal(density_matrix, expected_density_matrix)

def test_specific_state_measurement():
    # GHZState for 3 qubits
    # Outputs should be |000> or |111>
    ghz_state = GHZState(3)
    measured_state = ghz_state.measure_index()
    assert measured_state in [0, 7]
    bitstring = ghz_state.measure_bitstring()
    assert bitstring in ["000", "111"]

    # WState for 3 qubits
    # Outputs should be |001>, |010>, or |100>
    w_state = WState(3)
    measured_state = w_state.measure_index()
    assert measured_state in [1, 2, 4]
    bitstring = w_state.measure_bitstring()
    assert bitstring in ["001", "010", "100"]
