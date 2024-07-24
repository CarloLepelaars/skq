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
    
    def is_pure(self) -> bool:
        """ Check if the density matrix is a pure state. """
        return np.isclose(np.trace(self @ self), 1)
    
    def is_mixed(self) -> bool:
        """ Check if the density matrix is a mixed state. """
        return not self.is_pure()
    
    def trace_equal_to_one(self) -> bool:
        """ Check if the trace of the density matrix is equal to one. """
        return np.isclose(np.trace(self), 1)
    
    def probabilities(self) -> float:
        """ Return the probabilities of all possible state measurements. """
        return np.diag(self).real

    def eigenvalues(self) -> np.ndarray:
        """ Return the eigenvalues of the density matrix. """
        return np.linalg.eigvals(self)
    
    def eigenvectors(self) -> np.ndarray:
        """ Return the eigenvectors of the density matrix. """
        _, vectors = np.linalg.eig(self)
        return vectors
    
    def num_qubits(self) -> int:
        """ Return the number of qubits in the density matrix. """
        return int(np.log2(len(self)))
    
    def is_multi_qubit(self) -> bool:
        """ Check if the density matrix represents a multi-qubit state. """
        return self.num_qubits() > 1
    
    def trace_norm(self) -> float:
        """ Return the trace norm of the density matrix. """
        return np.trace(np.sqrt(self.conjugate_transpose() @ self))
    
    def distance(self, other: 'DensityMatrix') -> float:
        """ Return the trace norm distance between two density matrices. """
        assert isinstance(other, DensityMatrix), "'other' argument must be a valid DensityMatrix object."
        return self.trace_norm(self - other)
    
    def bloch_vector(self) -> np.ndarray:
        """ Calculate the Bloch vector of the density matrix. """
        if self.num_qubits() > 1:
            raise NotImplementedError("Bloch vector is not yet implemented for multi-qubit states.")
        # Pauli matrices
        sigma_x = np.array([[0, 1], 
                            [1, 0]])
        sigma_y = np.array([[0, -1j], 
                            [1j, 0]])
        sigma_z = np.array([[1, 0], 
                            [0, -1]])
        
        # Bloch vector components
        bx = np.trace(np.dot(self, sigma_x)).real
        by = np.trace(np.dot(self, sigma_y)).real
        bz = np.trace(np.dot(self, sigma_z)).real
        return np.array([bx, by, bz])
    
    def conjugate_transpose(self) -> np.ndarray:
        """
        Return the conjugate transpose (Hermitian adjoint) of the density matrix.
        1. Take the complex conjugate of each element (Flip the sign of the imaginary part)
        2. Transpose the matrix
        """
        return self.conj().T
    
    @staticmethod
    def from_probabilities(probabilities: np.array) -> 'DensityMatrix':
        """
        Create a density matrix from a list of probabilities.
        :param probabilities: A 1D array of probabilities
        :return: Density matrix
        """
        assert np.isclose(np.sum(probabilities), 1), f"Probabilities must sum to one. Got sum: {np.sum(probabilities)}"
        assert len(probabilities.shape) == 1, f"Probabilities must be a 1D array. Got shape: {probabilities.shape}"
        return DensityMatrix(np.diag(probabilities))


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
