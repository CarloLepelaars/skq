import qiskit
import numpy as np
import scipy as sp
from skq.gates.base import BaseGate

# I, X, Y, Z Pauli matrices
SINGLE_QUBIT_PAULI_MATRICES = [
    np.eye(2, dtype=complex),  # I
    np.array([[0, 1], [1, 0]], dtype=complex),  # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
    np.array([[1, 0], [0, -1]], dtype=complex)  # Z
]
# Any two-qubit Pauli matrix can be expressed as a Kronecker product of two single-qubit Pauli matrices
TWO_QUBIT_PAULI_MATRICES = [
    np.kron(pauli1, pauli2) for pauli1 in SINGLE_QUBIT_PAULI_MATRICES for pauli2 in SINGLE_QUBIT_PAULI_MATRICES
]

# X, Y, Z, H and S gates are Clifford gates
SINGLE_QUBIT_CLIFFORD_MATRICES = SINGLE_QUBIT_PAULI_MATRICES + [
    np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # H
    np.array([[1, 0], [0, 1j]], dtype=complex)  # S
]
# Two qubit Clifford gates can be expressed as a Kronecker product of two single-qubit Clifford gates + CNOT and SWAP
TWO_QUBIT_CLIFFORD_MATRICES = [
    np.kron(clifford1, clifford2) for clifford1 in SINGLE_QUBIT_CLIFFORD_MATRICES for clifford2 in SINGLE_QUBIT_CLIFFORD_MATRICES
] + [
    np.array([[1, 0, 0, 0], 
              [0, 1, 0, 0], 
              [0, 0, 0, 1], 
              [0, 0, 1, 0]], dtype=complex),  # CX (CNOT)
    np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, -1j],
              [0, 0, 1j, 0]], dtype=complex),  # CY
    np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, -1]], dtype=complex),  # CZ
    np.array([[1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]], dtype=complex),  # SWAP
    np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
              [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex), # CH
    np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1j]], dtype=complex), # CS
]

class QubitGate(BaseGate):
    """ Base class for Qubit gates. """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_at_least_2x2(), "Gate must be at least a 2x2 matrix"
        assert obj.is_power_of_two_shape(), "Gate shape must be a power of 2"
        return obj
    
    def is_at_least_2x2(self) -> bool:
        rows, cols = self.shape
        return rows >= 2 and cols >= 2, "Gate must be at least a 2x2 matrix"
    
    def is_power_of_two_shape(self) -> bool:
        rows, cols = self.shape
        rows_valid = (rows > 0) and (rows & (rows - 1)) == 0, f"Number of rows for gate must be a power of 2. Got '{rows}'"
        cols_valid = (cols > 0) and (cols & (cols - 1)) == 0, f"Number of columns for gate must be a power of 2. Got '{cols}'"
        return rows_valid and cols_valid
    
    def num_qubits(self) -> int:
        """ Return the number of qubits involved in the gate. """
        return int(np.log2(self.shape[0]))
    
    def is_multi_qubit(self) -> bool:
        """ Check if the gate involves multiple qubits. """
        return self.num_qubits() > 1
    
    def is_pauli(self) -> bool:
        """ Check if the gate is a Pauli gate. """
        # I, X, Y, Z Pauli matrices
        if self.num_qubits() == 1:
            return any(np.allclose(self, pauli) for pauli in SINGLE_QUBIT_PAULI_MATRICES)
        # Combinations of single-qubit Pauli matrices
        elif self.num_qubits() == 2:
            return any(np.allclose(self, pauli) for pauli in TWO_QUBIT_PAULI_MATRICES)
        else:
            return NotImplementedError("Pauli check not supported for gates with more than 2 qubits")
        
    def is_clifford(self) -> bool:
        """ Check if the gate is a Clifford gate. """
        # X, Y, Z, H and S
        if self.num_qubits() == 1:
            return any(np.allclose(self, clifford) for clifford in SINGLE_QUBIT_CLIFFORD_MATRICES)
        # Combinations of single-qubit Clifford gates + CNOT and SWAP
        elif self.num_qubits() == 2:
            return any(np.allclose(self, clifford) for clifford in TWO_QUBIT_CLIFFORD_MATRICES)
        else:
            return NotImplementedError("Clifford check not supported for gates with more than 2 qubits")

    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        """ Convert gate to a Qiskit Gate object. """
        gate_name = self.__class__.__name__
        print(f"No Qiskit alias (to_qiskit) defined for '{gate_name}'. Initializing as UnitaryGate.")
        return qiskit.circuit.library.UnitaryGate(self, label=gate_name)
        
    @staticmethod 
    def from_qiskit(qiskit_gate: qiskit.circuit.gate.Gate) -> 'CustomQubitGate':
        """ 
        Convert a Qiskit Gate to scikit-q CustomGate. 
        :param qiskit_gate: Qiskit Gate object
        """
        assert isinstance(qiskit_gate, qiskit.circuit.gate.Gate), "Input must be a Qiskit Gate object"
        return CustomQubitGate(qiskit_gate.to_matrix())
    
    def sqrt(self) -> 'CustomQubitGate':
        """ Compute the square root of the gate. """
        sqrt_matrix = sp.linalg.sqrtm(self)
        return CustomQubitGate(sqrt_matrix)
    
    def kron(self, other: 'QubitGate') -> 'CustomQubitGate':
        """ Compute the Kronecker product of two gates. """
        kron_matrix = np.kron(self, other)
        return CustomQubitGate(kron_matrix)

class CustomQubitGate(QubitGate):
    """ Bespoke gate. Must be unitary to function as a quantum gate. """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        return obj
    
    def to_qiskit(self) -> str:
        return qiskit.circuit.library.UnitaryGate(self, label="CustomGate")
    