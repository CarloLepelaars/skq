import pytest
import numpy as np
from skq.decomposition import schmidt_decomposition

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
