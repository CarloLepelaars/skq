import qiskit
import numpy as np
import scipy as sp


# I, X, Y, Z Pauli matrices
SINGLE_QUBIT_PAULI_MATRICES = [
    np.eye(2, dtype=complex),  # I
    np.array([[0, 1], [1, 0]], dtype=complex),  # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
    np.array([[1, 0], [0, -1]], dtype=complex)  # Z
]

# X, Y, Z, H and S gates are Clifford gates
SINGLE_QUBIT_CLIFFORD_MATRICES = SINGLE_QUBIT_PAULI_MATRICES + [
    np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # H
    np.array([[1, 0], [0, 1j]], dtype=complex)  # S
]


class Gate(np.ndarray):
    """ Base class for quantum gates with NumPy. """
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=complex)
        obj = arr.view(cls)
        assert obj.is_unitary(), "Gate must be unitary"
        assert obj.is_2d(), "Gate must be a 2D matrix"
        assert obj.is_at_least_2x2(), "Gate must be at least a 2x2 matrix"
        assert obj.is_power_of_two_shape(), "Gate shape must be a power of 2"
        return obj

    def is_unitary(self) -> bool:
        """ Check if the gate is unitary: U*U^dagger = I """
        identity = np.eye(self.shape[0])
        return np.allclose(self @ self.conjugate_transpose(), identity)
    
    def is_2d(self) -> bool:
        return len(self.shape) == 2, "Gate must be a 2D matrix"
    
    def is_at_least_2x2(self) -> bool:
        rows, cols = self.shape
        return rows >= 2 and cols >= 2, "Gate must be at least a 2x2 matrix"
    
    def is_power_of_two_shape(self) -> bool:
        rows, cols = self.shape
        rows_valid = (rows > 0) and (rows & (rows - 1)) == 0, f"Number of rows for gate must be a power of 2. Got '{rows}'"
        cols_valid = (cols > 0) and (cols & (cols - 1)) == 0, f"Number of columns for gate must be a power of 2. Got '{cols}'"
        return rows_valid and cols_valid

    def is_hermitian(self) -> bool:
        """ Check if the gate is Hermitian: U = U^dagger """
        return np.allclose(self, self.conjugate_transpose())
    
    def is_identity(self) -> bool:
        """ Check if the gate is the identity matrix. """
        return np.allclose(self, np.eye(self.shape[0]))

    def eigenvalues(self) -> np.ndarray:
        """ Return the eigenvalues of the gate """
        return np.linalg.eigvals(self)

    def eigenvectors(self) -> np.ndarray:
        """ Return the eigenvectors of the gate. """
        _, vectors = np.linalg.eig(self)
        return vectors
    
    def matrix_trace(self) -> complex:
        """ Compute the trace of the gate. The trace is the sum of the diagonal elements. """
        return np.trace(self)

    def conjugate_transpose(self) -> np.ndarray:
        """
        Return the conjugate transpose (Hermitian adjoint) of the gate.
        1. Transpose the matrix
        2. Take the complex conjugate of each element (Flip the sign of the imaginary part)
        """
        return self.conj().T

    def frobenius_norm(self) -> float:
        """ Compute the Frobenius norm """
        return np.linalg.norm(self)
    
    def num_qubits(self) -> int:
        """ Return the number of qubits involved in the gate. """
        return int(np.log2(self.shape[0]))
    
    def is_multi_qubit(self) -> bool:
        """ Check if the gate involves multiple qubits. """
        return self.num_qubits() > 1
    
    def is_single_qubit_pauli(self) -> bool:
        """ Check if the gate is a single-qubit Pauli gate. """
        # Multi qubit gates are not single qubit Pauli gates
        if self.is_multi_qubit():
            return False
        # Check if the gate is in the list of single qubit Pauli gates
        return any(np.allclose(self, pauli) for pauli in SINGLE_QUBIT_PAULI_MATRICES)
    
    def is_single_qubit_clifford(self) -> bool:
        """ Check if the gate is a single-qubit Clifford gate. """
        # Multi qubit gates are not single qubit Clifford gates
        if self.is_multi_qubit():
            return False
        # Check if the gate is in the list of single qubit Clifford gates
        return any(np.allclose(self, clifford) for clifford in SINGLE_QUBIT_CLIFFORD_MATRICES)
    
    def is_equal(self, other) -> bool:
        """ Check if the gate is effectively equal to another gate. 
        NOTE: Do not overwrite __eq__ method to avoid issues with NumPy array comparison. 
        """
        return np.allclose(self, other, atol=1e-8)

    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        """ Convert gate to a Qiskit Gate object. """
        gate_name = self.__class__.__name__
        print(f"No Qiskit alias (to_qiskit) defined for '{gate_name}'. Initializing as UnitaryGate.")
        return qiskit.circuit.library.UnitaryGate(self, label=gate_name)
        
    @staticmethod 
    def from_qiskit(qiskit_gate: qiskit.circuit.gate.Gate) -> 'CustomGate':
        """ 
        Convert a Qiskit Gate to scikit-q CustomGate. 
        :param qiskit_gate: Qiskit Gate object
        """
        assert isinstance(qiskit_gate, qiskit.circuit.gate.Gate), "Input must be a Qiskit Gate object"
        return CustomGate(qiskit_gate.to_matrix())
    
    def sqrt(self) -> 'CustomGate':
        """ Compute the square root of the gate. """
        sqrt_matrix = sp.linalg.sqrtm(self)
        return CustomGate(sqrt_matrix)
    
    def kron(self, other) -> 'CustomGate':
        """ Compute the Kronecker product of two gates. """
        kron_matrix = np.kron(self, other)
        return CustomGate(kron_matrix)

class CustomGate(Gate):
    """ Bespoke gate. Must be unitary to function as a quantum gate. """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_unitary(), "Custom gate must be unitary"
        return obj
    
    def to_qiskit(self) -> str:
        return qiskit.circuit.library.UnitaryGate(self, label="CustomGate")
    