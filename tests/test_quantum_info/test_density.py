import qiskit
import numpy as np
import pennylane as qml

from skq.quantum_info import Statevector, DensityMatrix, GibbsState


def test_zero_and_one_density_matrix():
    # Zero state |0⟩
    zero_state = Statevector([1, 0])
    zero_density_matrix = zero_state.density_matrix()
    assert isinstance(zero_density_matrix, DensityMatrix)
    assert zero_density_matrix.is_pure()
    assert zero_density_matrix.num_qubits() == 1
    assert zero_density_matrix.is_multi_qubit() == False
    assert zero_density_matrix.is_hermitian()
    assert zero_density_matrix.is_positive_semidefinite()
    assert zero_density_matrix.trace_equal_to_one()
    assert zero_density_matrix.dtype == complex
    assert np.allclose(zero_density_matrix, np.array([[1, 0], 
                                                      [0, 0]]))
    np.testing.assert_array_almost_equal(zero_density_matrix**2, zero_density_matrix)
    np.testing.assert_array_almost_equal((zero_density_matrix**2).trace(), 1)

    # One state |1⟩
    one_state = Statevector([0, 1])
    one_density_matrix = one_state.density_matrix()
    assert isinstance(one_density_matrix, DensityMatrix)
    assert one_density_matrix.is_pure()
    assert np.allclose(one_density_matrix, np.array([[0, 0], 
                                                     [0, 1]]))
    
    assert np.isclose(one_density_matrix.entropy(), 0)
    assert np.isclose(one_density_matrix.internal_energy(np.array([[0, 0], [0, 1]])), 1)
    assert (one_density_matrix**2).trace() == 1
    np.testing.assert_array_almost_equal(one_density_matrix**2, one_density_matrix)

    
def test_density_mixed_state():
    # Mixed state |ψ⟩ = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1|
    mixed_state = np.array([[0.5, 0], 
                            [0, 0.5]])
    mixed_density_matrix = DensityMatrix(mixed_state)
    assert mixed_density_matrix.dtype == complex
    assert mixed_density_matrix.is_mixed()
    assert not mixed_density_matrix.is_pure()
    assert np.allclose(mixed_density_matrix, np.array([[0.5, 0], 
                                                       [0, 0.5]]))
    assert np.allclose(mixed_density_matrix.probabilities(), [0.5, 0.5])
    assert mixed_density_matrix.num_qubits() == 1
    assert np.allclose(mixed_density_matrix.bloch_vector(), np.array([0, 0, 0]))
    
    assert np.isclose(mixed_density_matrix.entropy(), np.log(2))
    assert np.isclose(mixed_density_matrix.internal_energy(np.array([[0, 0], [0, 1]])), 0.5)

    assert not np.allclose(mixed_density_matrix**2, mixed_density_matrix)
    assert (mixed_density_matrix**2).trace() < 1

def test_density_conjugate_transpose():
    mixed_state = DensityMatrix(np.array([[0.25+0.j, 0.02+0.05j, 0.0125-0.025j, 0.0125+0.0125j],
                                          [0.02-0.05j, 0.25+0.j, 0.0125+0.025j, 0.0125-0.0125j],
                                          [0.0125+0.025j, 0.0125-0.025j, 0.25+0.j, 0.025+0.025j],
                                          [0.0125-0.0125j, 0.0125+0.0125j, 0.025-0.025j, 0.25+0.j]]))
    complex_conjugate = mixed_state.conjugate_transpose()
    # Same as initial state because a DensityMatrix is Hermitian
    expected_complex_conjugate = DensityMatrix(np.array([[0.25+0.j, 0.02+0.05j, 0.0125-0.025j, 0.0125+0.0125j],
                                                         [0.02-0.05j, 0.25+0.j, 0.0125+0.025j, 0.0125-0.0125j],
                                                         [0.0125+0.025j, 0.0125-0.025j, 0.25+0.j, 0.025+0.025j],
                                                         [0.0125-0.0125j, 0.0125+0.0125j, 0.025-0.025j, 0.25+0.j]]))
    np.testing.assert_array_almost_equal(complex_conjugate, expected_complex_conjugate)

    # (A^H)^H = A
    double_conjugate = complex_conjugate.conjugate_transpose()
    np.testing.assert_array_almost_equal(double_conjugate, mixed_state)

    # (A + B)^H = A^H + B^H
    A_plus_B_conjugate = (mixed_state + mixed_state).conjugate_transpose()
    A_conjugate_plus_B_conjugate = mixed_state.conjugate_transpose() + mixed_state.conjugate_transpose()
    np.testing.assert_array_almost_equal(A_plus_B_conjugate, A_conjugate_plus_B_conjugate)

    # (AB)^H = B^H A^H
    AB_conjugate = (mixed_state @ mixed_state).conjugate_transpose()
    BA_conjugate = mixed_state.conjugate_transpose() @ mixed_state.conjugate_transpose()
    np.testing.assert_array_almost_equal(AB_conjugate, BA_conjugate)

def test_density_from_to_qiskit():
    mixed_state = np.array([[0.25+0.j, 0.02+0.05j, 0.0125-0.025j, 0.0125+0.0125j],
                           [0.02-0.05j, 0.25+0.j, 0.0125+0.025j, 0.0125-0.0125j],
                           [0.0125+0.025j, 0.0125-0.025j, 0.25+0.j, 0.025+0.025j],
                           [0.0125-0.0125j, 0.0125+0.0125j, 0.025-0.025j, 0.25+0.j]])
    mixed_density_matrix = DensityMatrix(mixed_state)
    qiskit_density_matrix = mixed_density_matrix.to_qiskit()
    assert isinstance(qiskit_density_matrix, qiskit.quantum_info.DensityMatrix)
    assert np.allclose(qiskit_density_matrix.data, mixed_state)

    test_qiskit_density_matrix = qiskit.quantum_info.DensityMatrix(mixed_state)
    skq_density_matrix = DensityMatrix.from_qiskit(test_qiskit_density_matrix)
    assert isinstance(skq_density_matrix, DensityMatrix)
    assert np.allclose(skq_density_matrix, test_qiskit_density_matrix.data)
    assert np.allclose(skq_density_matrix, mixed_state)

def test_density_from_to_pennylane():
    mixed_state = np.array([[0.25+0.j, 0.02+0.05j, 0.0125-0.025j, 0.0125+0.0125j],
                           [0.02-0.05j, 0.25+0.j, 0.0125+0.025j, 0.0125-0.0125j],
                           [0.0125+0.025j, 0.0125-0.025j, 0.25+0.j, 0.025+0.025j],
                           [0.0125-0.0125j, 0.0125+0.0125j, 0.025-0.025j, 0.25+0.j]])
    mixed_density_matrix = DensityMatrix(mixed_state)
    pennylane_density_matrix = mixed_density_matrix.to_pennylane()
    assert isinstance(pennylane_density_matrix, qml.QubitDensityMatrix)
    assert np.allclose(pennylane_density_matrix.parameters, mixed_state)

    test_pennylane_density_matrix = qml.QubitDensityMatrix(mixed_state, wires=[0, 1])
    skq_density_matrix = DensityMatrix.from_pennylane(test_pennylane_density_matrix)
    assert isinstance(skq_density_matrix, DensityMatrix)
    assert np.allclose(skq_density_matrix, test_pennylane_density_matrix.parameters)
    assert np.allclose(skq_density_matrix, mixed_state)

def test_gibbs_state():
    # Define Hamiltonian and Temperature
    simple_hamiltonian = np.array([[0, 0], [0, 1]])
    temperature = 1.0

    # Create Gibbs state
    gibbs_state = GibbsState(simple_hamiltonian, temperature)

    assert isinstance(gibbs_state, DensityMatrix)
    eigenvalues = gibbs_state.eigenvalues()
    assert np.all(eigenvalues >= 0)
    assert np.isclose(np.sum(eigenvalues), 1)

    # 0K temperature
    zero_temperature = 1e-10 
    gibbs_state_zero = GibbsState(simple_hamiltonian, zero_temperature)
    # Ground state projection
    expected_density_matrix_zero = np.array([[1, 0], [0, 0]]) 
    np.testing.assert_allclose(gibbs_state_zero, expected_density_matrix_zero)

    # Very high temperature
    high_temperature = 1e30
    gibbs_state_high = GibbsState(simple_hamiltonian, high_temperature)
     # Maximally mixed state
    expected_density_matrix_high = np.array([[0.5, 0], [0, 0.5]])
    np.testing.assert_allclose(gibbs_state_high, expected_density_matrix_high)

    assert gibbs_state.free_energy() <= 0
    assert gibbs_state.entropy() >= 0  
    assert gibbs_state.internal_energy(simple_hamiltonian) >= 0 
    assert gibbs_state.heat_capacity() >= 0 
