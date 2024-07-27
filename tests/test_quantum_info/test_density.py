import qiskit
import numpy as np
import pennylane as qml
from skq.quantum_info import Statevector
from skq.quantum_info import DensityMatrix

def test_zero_and_one_density_matrix():
    # Zero state |0⟩
    zero_state = Statevector([1, 0])
    zero_density_matrix = zero_state.density_matrix()
    assert isinstance(zero_density_matrix, DensityMatrix)
    assert zero_density_matrix.is_pure()
    assert zero_density_matrix.num_qubits() == 1
    assert zero_density_matrix.is_multi_qubit() == False
    assert zero_density_matrix.is_hermitian()
    assert zero_density_matrix.dtype == complex
    assert np.allclose(zero_density_matrix, np.array([[1, 0], 
                                                      [0, 0]]))

    # One state |1⟩
    one_state = Statevector([0, 1])
    one_density_matrix = one_state.density_matrix()
    assert np.allclose(one_density_matrix, np.array([[0, 0], 
                                                     [0, 1]]))
    
def test_density_mixed_state():
    # Mixed state |ψ⟩ = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1|
    mixed_state = np.array([[0.5, 0], 
                            [0, 0.5]])
    mixed_density_matrix = DensityMatrix(mixed_state)
    assert mixed_density_matrix.dtype == complex
    assert mixed_density_matrix.is_mixed()
    assert np.allclose(mixed_density_matrix, np.array([[0.5, 0], 
                                                       [0, 0.5]]))
    assert np.allclose(mixed_density_matrix.probabilities(), [0.5, 0.5])
    assert mixed_density_matrix.num_qubits() == 1
    assert np.allclose(mixed_density_matrix.bloch_vector(), np.array([0, 0, 0]))

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
