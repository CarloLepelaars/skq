import qiskit
import numpy as np
import pennylane as qml

from skq.gates.qubit import XGate, YGate, ZGate


class DensityMatrix(np.ndarray):
    """
    Density matrix representation of a quantum state.
    """
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=complex)
        obj = arr.view(cls)
        assert obj.is_hermitian(), "Density matrix must be Hermitian."
        assert obj.is_positive_semidefinite(), "Density matrix must be positive semidefinite (All eigenvalues >= 0)."
        assert obj.trace_equal_to_one(), "Density matrix must have trace equal to one. Normalize to unit trace if you want to use this matrix as a DensityMatrix."
        return obj
    
    def is_hermitian(self) -> bool:
        """ Check if the density matrix is Hermitian: U = U^dagger. """
        return np.allclose(self, self.conjugate_transpose())
    
    def is_positive_semidefinite(self):
        """ Check if the matrix is positive semidefinite. """
        eigenvalues = np.linalg.eigvalsh(self)
        return np.all(eigenvalues >= 0)
    
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
        
        # Bloch vector components
        bx = np.trace(np.dot(self, XGate())).real
        by = np.trace(np.dot(self, YGate())).real
        bz = np.trace(np.dot(self, ZGate())).real
        return np.array([bx, by, bz])
    
    def conjugate_transpose(self) -> np.ndarray:
        """
        Return the conjugate transpose (Hermitian adjoint) of the density matrix.
        1. Take the complex conjugate of each element (Flip the sign of the imaginary part)
        2. Transpose the matrix
        """
        return self.conj().T
    
    def kron(self, other: 'DensityMatrix') -> 'DensityMatrix':
        """
        Compute the Kronecker (tensor) product of two density matrices.
        This can be used to create so-called "product states" that represent 
        the independence between two quantum systems.
        :param other: Density matrix
        :return: Kronecker product of the two density matrices
        """
        return DensityMatrix(np.kron(self, other))
    
    def to_qiskit(self) -> qiskit.quantum_info.DensityMatrix:
        """
        Convert the density matrix to a Qiskit DensityMatrix object.
        :return: Qiskit DensityMatrix object
        """
        return qiskit.quantum_info.DensityMatrix(self)
    
    @staticmethod
    def from_qiskit(density_matrix: qiskit.quantum_info.DensityMatrix) -> 'DensityMatrix':
        """
        Create a DensityMatrix object from a Qiskit DensityMatrix object.
        :param density_matrix: Qiskit DensityMatrix object
        :return: DensityMatrix object
        """
        return DensityMatrix(density_matrix.data)
    
    def to_pennylane(self, wires: list[int] | int = None) -> qml.QubitDensityMatrix:
        """
        Convert the density matrix to a PennyLane QubitDensityMatrix.
        :param wires: List of wires to apply the density matrix to
        :return: PennyLane QubitDensityMatrix object
        """
        wires = wires if wires is not None else range(self.num_qubits())
        return qml.QubitDensityMatrix(self, wires=wires)
    
    @staticmethod
    def from_pennylane(density_matrix: qml.QubitDensityMatrix) -> "DensityMatrix":
        """
        Convert a PennyLane QubitDensityMarix object to a scikit-q StateVector.
        :param density_matrix: PennyLane QubitDensityMatrix object
        :return: scikit-q StateVector object
        """
        return DensityMatrix(density_matrix.data[0])
    
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
    