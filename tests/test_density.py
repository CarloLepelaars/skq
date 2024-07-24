import pytest
import numpy as np
from skq.state import Statevector
from skq.density import DensityMatrix, schmidt_decomposition

def test_zero_and_one_density_matrix():
    # Zero state |0⟩
    zero_state = Statevector([1, 0])
    zero_density_matrix = zero_state.density_matrix()
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
    assert mixed_density_matrix.is_mixed()
    assert np.allclose(mixed_density_matrix, np.array([[0.5, 0], 
                                                       [0, 0.5]]))
    assert np.allclose(mixed_density_matrix.probabilities(), [0.5, 0.5])
    assert mixed_density_matrix.num_qubits() == 1
    assert np.allclose(mixed_density_matrix.bloch_vector(), np.array([0, 0, 0]))

def test_schmidt_decomposition():
    # Schmidt decomposition is not applicable for single qubit states
    with pytest.raises(AssertionError):
        schmidt_decomposition(np.array([1, 0]))
    # Non-bipartite states
    with pytest.raises(AssertionError):
        schmidt_decomposition(np.array([1, 0, 0]))

    # Basis state |00>
    basis_state_00 = np.array([1, 0, 0, 0])
    coeffs, basis_A, basis_B = schmidt_decomposition(basis_state_00)
    assert np.allclose(coeffs, [1, 0])
    assert np.allclose(basis_A[:, 0], [1, 0])
    assert np.allclose(basis_B[:, 0], [1, 0])

    # Bell state |ψ⟩ = (|00⟩ + |11⟩) / sqrt(2)
    bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    coeffs, basis_A, basis_B = schmidt_decomposition(bell_state)
    assert np.allclose(coeffs, [1/np.sqrt(2), 1/np.sqrt(2)])
    assert np.allclose(basis_A[:, 0], [1, 0]) or np.allclose(basis_A[:, 0], [0, 1])
    assert np.allclose(basis_B[:, 0], [1, 0]) or np.allclose(basis_B[:, 0], [0, 1])

    # State |ψ⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2
    intricate_state = np.array([1/2, 1/2, 1/2, 1/2])
    coeffs, basis_A, basis_B = schmidt_decomposition(intricate_state)
    assert np.allclose(coeffs, [1, 0])
    assert (np.allclose(basis_A[:, 0], [1/np.sqrt(2), 1/np.sqrt(2)]) or
            np.allclose(basis_A[:, 0], [1/np.sqrt(2), -1/np.sqrt(2)]) or
            np.allclose(basis_A[:, 0], [-1/np.sqrt(2), 1/np.sqrt(2)]) or
            np.allclose(basis_A[:, 0], [-1/np.sqrt(2), -1/np.sqrt(2)]))
    assert (np.allclose(basis_B[:, 0], [1/np.sqrt(2), 1/np.sqrt(2)]) or
            np.allclose(basis_B[:, 0], [1/np.sqrt(2), -1/np.sqrt(2)]) or
            np.allclose(basis_B[:, 0], [-1/np.sqrt(2), 1/np.sqrt(2)]) or
            np.allclose(basis_B[:, 0], [-1/np.sqrt(2), -1/np.sqrt(2)]))

    # State |ψ⟩ = (sqrt(3)/2 |00⟩ + 1/2 |11⟩)
    intricate_state_2 = np.array([np.sqrt(3)/2, 0, 0, 1/2])
    coeffs, basis_A, basis_B = schmidt_decomposition(intricate_state_2)
    assert np.allclose(coeffs, [np.sqrt(3)/2, 1/2])
    assert np.allclose(basis_A[:, 0], [1, 0]) or np.allclose(basis_A[:, 0], [0, 1])
    assert np.allclose(basis_B[:, 0], [1, 0]) or np.allclose(basis_B[:, 0], [0, 1])

    # Three-qubit GHZ state (i.e. maximum entanglement across 3 qubits) |ψ⟩ = (|000⟩ + |111⟩) / sqrt(2)
    ghz_state = np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
    coeffs, basis_A, basis_B = schmidt_decomposition(ghz_state)
    assert np.allclose(coeffs, [1/np.sqrt(2), 1/np.sqrt(2)])
    assert np.allclose(basis_A[:, 0], [1, 0]) or np.allclose(basis_A[:, 0], [0, 1])
    assert (np.allclose(basis_B[:, 0], [1, 0, 0, 0]) or np.allclose(basis_B[:, 0], [0, 0, 0, 1]))
