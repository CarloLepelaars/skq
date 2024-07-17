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
        if self.num_qubits() != 1:
            return False
        # Check if the gate is in the list of single qubit Pauli gates
        return any(np.allclose(self, pauli) for pauli in SINGLE_QUBIT_PAULI_MATRICES)
    
    def is_single_qubit_clifford(self) -> bool:
        """ Check if the gate is a single-qubit Clifford gate. """
        # Multi qubit gates are not single qubit Clifford gates
        if self.num_qubits() != 1:
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
    
class IGate(Gate):
    """ 
    Identity gate: 
    [[1, 0]
    [0, 1]]
    """
    def __new__(cls):
        return super().__new__(cls, np.eye(2))
    
    def to_qiskit(self) -> qiskit.circuit.library.IGate:
        return qiskit.circuit.library.IGate()
    
class XGate(Gate):
    """ Pauli X (NOT) Gate. """
    def __new__(cls):
        return super().__new__(cls, [[0, 1], 
                                     [1, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.XGate:
        return qiskit.circuit.library.XGate()

class YGate(Gate):
    """ Pauli Y gate. """
    def __new__(cls):
        return super().__new__(cls, [[0, -1j], 
                                     [1j, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.YGate:
        return qiskit.circuit.library.YGate()
    
class ZGate(Gate):
    """ Pauli Z gate. 
    Special case of a phase shift gate with phi = pi.
    """
    def __new__(cls):
        return super().__new__(cls, [[1, 0], 
                                     [0, -1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.ZGate:
        return qiskit.circuit.library.ZGate()

class HGate(Gate):
    """ 
    Hadamard gate. Used to create superposition. 
    |0> -> (|0> + |1>) / sqrt(2)
    |1> -> (|0> - |1>) / sqrt(2)
    """
    def __new__(cls):
        return super().__new__(cls, [[1, 1], 
                                     [1, -1]] / np.sqrt(2))
    
    def to_qiskit(self) -> qiskit.circuit.library.HGate:
        return qiskit.circuit.library.HGate()
    
class PhaseGate(Gate):
    """ General phase shift gate. 
    Special cases of phase gates:
    - S gate: theta = pi / 2
    - T gate: theta = pi / 4
    - Z gate: theta = pi
    """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[1, 0], 
                                    [0, np.exp(1j * theta)]])
        obj.theta = theta
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.PhaseGate:
        return qiskit.circuit.library.PhaseGate(self.theta)
    
class TGate(PhaseGate):
    """ T gate: phase shift gate with theta = pi / 4. """
    def __new__(cls):
        theta = np.pi / 4
        return super().__new__(cls, theta=theta)
    
    def to_qiskit(self) -> qiskit.circuit.library.TGate:
        return qiskit.circuit.library.TGate()
    
class SGate(PhaseGate):
    """ S gate: phase shift gate with theta = pi / 2. """
    def __new__(cls):
        theta = np.pi / 2
        return super().__new__(cls, theta=theta)
    
    def to_qiskit(self) -> qiskit.circuit.library.SGate:
        return qiskit.circuit.library.SGate()

class CXGate(Gate):
    """ 
    Controlled-X (CNOT) gate. 
    Used to entangle two qubits.
    If the control qubit is |1>, the target qubit is flipped.
    """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 1], 
                                     [0, 0, 1, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CXGate:
        return qiskit.circuit.library.CXGate()

class CYGate(Gate):
    """ Controlled-Y gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, -1j], 
                                     [0, 0, 1j, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CYGate:
        return qiskit.circuit.library.CYGate()
    
class CZGate(Gate):
    """ Controlled-Z gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 0, 0, -1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CZGate:
        return qiskit.circuit.library.CZGate()
    
class CHGate(Gate):
    """ Controlled-Hadamard gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], 
                                     [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CHGate:
        return qiskit.circuit.library.CHGate()
    
class CPhaseGate(Gate):
    """ General controlled phase shift gate. 
    Special cases of CPhase gates:
    """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, np.exp(1j * theta)]])
        obj.theta = theta
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.CPhaseGate:
        return qiskit.circuit.library.CPhaseGate(self.theta)
    
class CSGate(CPhaseGate):
    """ Controlled-S gate. """
    def __new__(cls):
        theta = np.pi / 2
        return super().__new__(cls, theta=theta)
    
    def to_qiskit(self) -> qiskit.circuit.library.CSGate:
        return qiskit.circuit.library.CSGate()
    
class CTGate(CPhaseGate):
    """ Controlled-T gate. """
    def __new__(cls):
        theta = np.pi / 4
        return super().__new__(cls, theta=theta)
    
class SWAPGate(Gate):
    """ Swap gate. Swaps the states of two qubits. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.SwapGate:
        return qiskit.circuit.library.SwapGate()
    
class CCXGate(Gate):
    """ A 3-qubit controlled-controlled-X (CCX) gate. Also known as the Toffoli gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, 1], 
                                     [0, 0, 0, 0, 0, 0, 1, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CCXGate:
        return qiskit.circuit.library.CCXGate()
    
class CSwapGate(Gate):
    """ A controlled-SWAP gate. Also known as the Fredkin gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 1, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, 1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CSwapGate:
        return qiskit.circuit.library.CSwapGate()
    
class RXGate(Gate):
    """ Generalized X rotation gate. """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.cos(theta / 2), -1j * np.sin(theta / 2)], 
                                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
        obj.theta = theta
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.RXGate:
        return qiskit.circuit.library.RXGate(self.theta)
    
class RYGate(Gate):
    """ Generalized Y rotation gate. """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.cos(theta / 2), -np.sin(theta / 2)], 
                                     [np.sin(theta / 2), np.cos(theta / 2)]])
        obj.theta = theta
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.RYGate:
        return qiskit.circuit.library.RYGate(self.theta)
    
class RZGate(Gate):
    """ Generalized Z rotation gate. """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.exp(-1j * theta / 2), 0], 
                                     [0, np.exp(1j * theta / 2)]])
        obj.theta = theta
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.RZGate:
        return qiskit.circuit.library.RZGate(self.theta)

class U3Gate(Gate):
    """ Rotation around 3-axes using one single qubit gate."""
    def __new__(cls, theta, phi, lam):
        # Rotation matrices
        Rx = RXGate(theta)
        Ry = RYGate(phi)
        Rz = RZGate(lam)
        combined_matrix = Rz @ Ry @ Rx
        
        obj = super().__new__(cls, combined_matrix)
        obj.theta = theta
        obj.phi = phi
        obj.lam = lam
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.U3Gate:
        return qiskit.circuit.library.U3Gate(self.theta, self.phi, self.lam)
    

