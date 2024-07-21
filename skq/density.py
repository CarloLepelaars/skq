import numpy as np
from scipy.linalg import svd


class DensityMatrix(np.ndarray):
    """
    Density matrix representation of a quantum state.
    """
    def __new__(cls, input_array):
        arr = np.asarray(input_array)
        obj = arr.view(cls)
        assert obj.is_hermitian(), "Density matrix must be Hermitian."
        assert obj.is_positive_semidefinite(), "Density matrix must be positive semidefinite (All eigenvalues >= 0)."
        assert obj.trace_equal_to_one(), "Density matrix must have trace equal to one."
        return obj
    
    def is_hermitian(self) -> bool:
        """ Check if the density matrix is Hermitian: U = U^dagger. """
        return np.allclose(self, self.conjugate_transpose())
    
    def is_positive_semidefinite(self) -> bool:
        """ Check if the density matrix is positive semidefinite. """
        return np.all(self.eigenvalues() >= 0)
    
    def trace_equal_to_one(self) -> bool:
        """ Check if the trace of the density matrix is equal to one. """
        return np.isclose(np.trace(self), 1)
    
    def probabilities(self) -> float:
        """ Return the probabilities of all possible states. """
        return np.real(np.diag(self))

    def eigenvalues(self) -> np.ndarray:
        """ Return the eigenvalues of the density matrix. """
        return np.linalg.eigvals(self)
    
    def eigenvectors(self) -> np.ndarray:
        """ Return the eigenvectors of the density matrix. """
        _, vectors = np.linalg.eig(self)
        return vectors
    
    def conjugate_transpose(self) -> np.ndarray:
        """
        Return the conjugate transpose (Hermitian adjoint) of the density matrix.
        1. Transpose the matrix
        2. Take the complex conjugate of each element (Flip the sign of the imaginary part)
        """
        return self.conj().T


def schmidt_decomposition(state_vector: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Perform Schmidt decomposition on a bipartite quantum state.
    :param state_vector: Bipartite quantum state vector
    :return: Tuple of Schmidt coefficients, Basis A and Basis B
    """
    assert len(state_vector) > 2, "Invalid state vector: Schmidt decomposition is not applicable for single qubit states."
    assert len(state_vector) % 2 == 0, "Invalid state vector: Not a bipartite state"
    
    # Infer dimensions
    N = len(state_vector)
    dim_A = int(np.sqrt(N))
    dim_B = N // dim_A

    # SVD on the state matrix
    state_matrix = state_vector.reshape(dim_A, dim_B)
    U, S, Vh = svd(state_matrix)

    # Coefficients (S), Basis A (U) and Basis B (Vh^dagger)
    return S, U, Vh.conj().T
