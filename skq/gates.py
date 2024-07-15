import numpy as np

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


class BaseGate(np.ndarray):
    """ Base class for quantum gates with NumPy. """
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=complex)
        obj = arr.view(cls)
        return obj

    def is_unitary(self) -> bool:
        """ Check if the gate is unitary: U*U^dagger = I """
        identity = np.eye(self.shape[0])
        return np.allclose(self @ self.conjugate_transpose(), identity)

    def is_hermitian(self) -> bool:
        """ Check if the gate is Hermitian: U = U^dagger """
        return np.allclose(self, self.conjugate_transpose())

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

    def determinant(self) -> complex:
        """ Compute the determinant of the gate. """
        return np.linalg.det(self)

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
    
class CustomGate(BaseGate):
    """ Bespoke gate. Must be unitary to function as a quantum gate. """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_unitary(), "Custom gate must be unitary"
        return obj
    
class IdentityGate(BaseGate):
    """ 
    Identity gate: 
    [[1, 0]
    [0, 1]]
    """
    def __new__(cls):
        return super().__new__(cls, np.eye(2))
    
class XGate(BaseGate):
    """ Pauli X (NOT) Gate. """
    def __new__(cls):
        return super().__new__(cls, [[0, 1], 
                                     [1, 0]])
    
class YGate(BaseGate):
    """ Pauli Y gate. """
    def __new__(cls):
        return super().__new__(cls, [[0, -1j], 
                                     [1j, 0]])
    
class ZGate(BaseGate):
    """ Pauli Z gate. 
    Special case of a phase shift gate with phi = pi.
    """
    def __new__(cls):
        return super().__new__(cls, [[1, 0], 
                                     [0, -1]])
    
class HadamardGate(BaseGate):
    """ 
    Hadamard gate. Used to create superposition. 
    |0> -> (|0> + |1>) / sqrt(2)
    |1> -> (|0> - |1>) / sqrt(2)
    """
    def __new__(cls):
        return super().__new__(cls, [[1, 1], 
                                     [1, -1]]) / np.sqrt(2)
    
class PhaseGate(BaseGate):
    """ General phase shift gate. 
    Special cases of phase gates:
    - S gate: phi = pi / 2
    - T gate: phi = pi / 4
    - Z gate: phi = pi
    """
    def __new__(cls, phi):
        obj = super().__new__(cls, [[1, 0], 
                                    [0, np.exp(1j * phi)]])
        obj.phi = phi
        return obj
    
class TGate(PhaseGate):
    """ T gate: phase shift gate with phi = pi / 4. """
    def __new__(cls):
        phi = np.pi / 4
        return super().__new__(cls, phi=phi)
    
class SGate(PhaseGate):
    """ S gate: phase shift gate with phi = pi / 2. """
    def __new__(cls):
        phi = np.pi / 2
        return super().__new__(cls, phi=phi)

class CXGate(BaseGate):
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

class CYGate(BaseGate):
    """ Controlled-Y gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, -1j], 
                                     [0, 0, 1j, 0]])
    
class CZGate(BaseGate):
    """ Controlled-Z gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 0, 0, -1]])
    
class CHGate(BaseGate):
    """ Controlled-Hadamard gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)], 
                                     [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]])
    
class CPhaseGate(BaseGate):
    """ General controlled phase shift gate. 
    Special cases of CPhase gates:
    """
    def __new__(cls, phi):
        obj = super().__new__(cls, [[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, np.exp(1j * phi)]])
        obj.phi = phi
        return obj
    
class CSGate(CPhaseGate):
    """ Controlled-S gate. """
    def __new__(cls):
        phi = np.pi / 2
        return super().__new__(cls, phi=phi)
    
class CTGate(CPhaseGate):
    """ Controlled-T gate. """
    def __new__(cls):
        phi = np.pi / 4
        return super().__new__(cls, phi=phi)
    
class SWAPGate(BaseGate):
    """ Swap gate. Swaps the states of two qubits. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 1]])
    
class ToffoliGate(BaseGate):
    """ Toffoli gate. A 3-qubit controlled-controlled-X (CCX) gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, 1], 
                                     [0, 0, 0, 0, 0, 0, 1, 0]])
    
class FredkinGate(BaseGate):
    """ Fredkin gate. A controlled-SWAP gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 1, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, 1]])
    
class RotXGate(BaseGate):
    """ Generalized X rotation gate. """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.cos(theta / 2), -1j * np.sin(theta / 2)], 
                                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
        obj.theta = theta
        return obj
    
class RotYGate(BaseGate):
    """ Generalized Y rotation gate. """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.cos(theta / 2), -np.sin(theta / 2)], 
                                     [np.sin(theta / 2), np.cos(theta / 2)]])
        obj.theta = theta
        return obj
    
class RotZGate(BaseGate):
    """ Generalized Z rotation gate. """
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.exp(-1j * theta / 2), 0], 
                                     [0, np.exp(1j * theta / 2)]])
        obj.theta = theta
        return obj

class GeneralizedRotationGate(BaseGate):
    """ Rotation around 3-axes using one single qubit gate. Also known as a U3 Gate. """
    def __new__(cls, theta_x, theta_y, theta_z):
        # Rotation matrices
        Rx = RotXGate(theta_x)
        Ry = RotYGate(theta_y)
        Rz = RotZGate(theta_z)
        combined_matrix = Rz @ Ry @ Rx
        
        obj = super().__new__(cls, combined_matrix)
        obj.theta_x = theta_x
        obj.theta_y = theta_y
        obj.theta_z = theta_z
        return obj
