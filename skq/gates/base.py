import qiskit
import numpy as np
import pennylane as qml


class BaseGate(np.ndarray):
    """ 
    Base class for quantum gates of any computational basis. 
    The gate must be a 2D unitary matrix.
    :param input_array: Input array to create the gate. Will be converted to a NumPy array.
    """
    def __new__(cls, input_array) -> 'BaseGate':
        arr = np.asarray(input_array, dtype=complex)
        obj = arr.view(cls)
        assert obj.is_unitary(), "Gate must be unitary."
        assert obj.is_2d(), "Gate must be a 2D matrix."
        assert obj.is_at_least_nxn(n=1), "Gate must be at least a 1x1 matrix."
        return obj
    
    def is_unitary(self) -> bool:
        """ Check if the gate is unitary (U*U^dagger = I) """
        identity = np.eye(self.shape[0])
        return np.allclose(self @ self.conjugate_transpose(), identity)
    
    def is_2d(self) -> bool:
        """ Check if the gate is a 2D matrix. """
        return len(self.shape) == 2
    
    def is_at_least_nxn(self, n: int) -> bool:
        """ Check if the gate is at least an n x n matrix. """
        rows, cols = self.shape
        return rows >= n and cols >= n

    def is_power_of_n_shape(self, n: int) -> bool:
        """ 
        Check if the gate shape is a power of n. 
        Qubits: n=2, Qutrits: n=3, Ququarts: n=4, etc.
        """
        def _is_power_of_n(x, n):
            if x < 1:
                return False
            while x % n == 0:
                x //= n
            return x == 1
        rows, cols = self.shape
        return _is_power_of_n(rows, n) and _is_power_of_n(cols, n)
    
    def num_levels(self) -> int:
        """ Number of rows. Used for checking valid shapes. """
        return self.shape[0]
    
    def is_hermitian(self) -> bool:
        """ Check if the gate is Hermitian: U = U^dagger """
        return np.allclose(self, self.conjugate_transpose())
    
    def is_identity(self) -> bool:
        """ Check if the gate is the identity matrix. """
        return np.allclose(self, np.eye(self.shape[0]))
    
    def commute(self, other: 'BaseGate') -> bool:
        """
        Check if the gate commutes with another gate.
        Two gates U and V commute if UV = VU.
        :param other: Gate object to check commutation with.
        """
        assert isinstance(other, BaseGate), "Other object must be a valid Gate object."
        assert self.num_levels() == other.num_levels(), "Gates must have the same number of rows for the commutation check."
        return np.allclose(self @ other, other @ self)
    
    def matrix_trace(self) -> complex:
        """ Compute the trace of the gate. The trace is the sum of the diagonal elements. """
        return np.trace(self)

    def conjugate_transpose(self) -> np.ndarray:
        """
        Return the conjugate transpose (i.e. Hermitian adjoint or 'dagger operation') of the gate.
        1. Take the complex conjugate of each element (Flip the sign of the imaginary part)
        2. Transpose the matrix
        """
        return self.conj().T
    
    def eigenvalues(self) -> np.ndarray:
        """ Return the eigenvalues of the gate """
        return np.linalg.eigvals(self)

    def eigenvectors(self) -> np.ndarray:
        """ Return the eigenvectors of the gate. """
        _, vectors = np.linalg.eig(self)
        return vectors

    def frobenius_norm(self) -> float:
        """ Compute the Frobenius norm """
        return np.linalg.norm(self)
    
    def is_equal(self, other) -> bool:
        """ Check if the gate is effectively equal to another gate. 
        NOTE: Do not overwrite __eq__ method to avoid issues with native NumPy array comparison. 
        """
        return np.allclose(self, other, atol=1e-8)
    
    def kernel_density(self, other: 'BaseGate') -> complex:
        """ 
        Calculate the quantum kernel using density matrices.
        The kernel is defined as Tr(U * V).
        :param other: Gate object to compute the kernel with.
        :return kernel: Complex number representing the kernel density.
        """
        assert isinstance(other, BaseGate), "Other object must be an instance of BaseUnitary (i.e. a Unitary matrix)."
        assert self.num_levels() == other.num_levels(), "Gates must have the same number of rows for the kernel density."
        return np.trace(self @ other)

    def hilbert_schmidt_inner_product(self, other: 'BaseGate') -> complex:
        """ 
        Calculate the Hilbert-Schmidt inner product with another gate. 
        The inner product is Tr(U^dagger * V).
        :param other: Gate object to compute the inner product with.
        :return inner_product: Complex number representing the inner product.
        """
        assert isinstance(other, BaseGate), "Other object must be an instance of BaseUnitary (i.e. a Unitary matrix)."
        assert self.num_levels() == other.num_levels(), "Gates must have the same number of rows for the Hilbert-Schmidt inner product."
        return np.trace(self.conjugate_transpose() @ other)
    
    def to_qiskit(self):
        """ Convert gate to a Qiskit Gate object. """
        raise NotImplementedError(f"Conversion to Qiskit Gate is not implemented for {self.__class__.__name__}.")
    
    def from_qiskit(self, qiskit_gate: qiskit.circuit.gate.Gate) -> 'BaseGate':
        """ 
        Convert a Qiskit Gate to scikit-q Gate. 
        :param qiskit_gate: Qiskit Gate object
        :return: scikit-q Gate object
        """
        return NotImplementedError(f"Conversion from Qiskit Gate is not implemented for {self.__class__.__name__}.")
    
    def to_pennylane(self):
        """ Convert gate to a PennyLane gate object. """
        raise NotImplementedError(f"Conversion to PennyLane is not implemented for {self.__class__.__name__}.")
    
    def from_pennylane(self, pennylane_gate: qml.operation.Operation) -> 'BaseGate':
        """
        Convert a PennyLane Operation to scikit-q Gate.
        :param pennylane_gate: PennyLane Operation object.
        :return: scikit-q Gate object
        """
        return NotImplementedError(f"Conversion from PennyLane is not implemented for {self.__class__.__name__}.")
