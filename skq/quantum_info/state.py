import qiskit
import numpy as np
import pennylane as qml

from skq.quantum_info.density import DensityMatrix


class Statevector(np.ndarray):
    """
    Statevector representation for a quantum (qubit) state.
    NOTE: Statevector is assumed to be constructed in big-endian format.
    Little-endian -> Least significant qubit (LSB) is on the right. Like |q1 q0> where q0 is the LSB.
    Big-endian -> Least significant qubit (LSB) is on the left. Like |q0 q1> where q0 is the LSB.
    """
    def __new__(cls, input_array):
        arr = np.asarray(input_array)
        obj = arr.view(cls)
        assert obj.is_1d(), "State vector must be 1D."
        assert obj.is_normalized(), "State vector must be normalized."
        assert obj.is_power_of_two_len(), "State vector length must be a power of 2."
        return obj
    
    def is_1d(self) -> bool:
        """ Check if the state vector is 1D. """
        return self.ndim == 1
    
    def is_normalized(self) -> bool:
        """ Check if the state vector is normalized: ||ψ|| = 1. """
        return np.isclose(np.linalg.norm(self), 1)
    
    def is_power_of_two_len(self) -> bool:
        """ Check if a number is a power of two """
        n = len(self)
        return (n > 0) and (n & (n - 1)) == 0
    
    def num_qubits(self) -> int:
        """ Return the number of qubits in the state vector. """
        return int(np.log2(len(self)))
    
    def is_multi_qubit(self) -> bool:
        """ Check if the state vector represents a multi-qubit state. """
        return self.num_qubits() > 1
    
    def is_indistinguishable(self, other: 'Statevector') -> bool:
        """ Check if two state vectors are indistinguishable by checking if their density matrices are the same. """
        return np.allclose(self.density_matrix(), other.density_matrix())
    
    def magnitude(self) -> float:
        """ 
        Magnitude (or norm) of the state vector. 
        sqrt(<ψ|ψ>)
        """
        np.linalg.norm(self)

    def density_matrix(self) -> DensityMatrix:
        """ Return the density matrix representation of the state vector. """
        return DensityMatrix(np.outer(self, self.conj()))
    
    def probabilities(self) -> np.ndarray:
        """ Return the probabilities of all possible states. """
        return np.abs(self) ** 2
    
    def measure_index(self) -> int:
        """ 
        Simulate a measurement of the state and get a sampled index. 
        :return: Index of the measured state.
        For example: 
        |0> -> 0
        |00> -> 0
        |11> -> 3
        """
        return np.random.choice(len(self), p=self.probabilities())
    
    def measure_bitstring(self) -> str:
        """ 
        Simulate a measurement of the state vector and get a bitstring representing the sample. 
        :return: Bitstring representation of the measured state.
        For example:
        |0> -> "0"
        |00> -> "00"
        |11> -> "11"
        """
        return bin(self.measure_index())[2:].zfill(self.num_qubits())
    
    def reverse(self) -> 'Statevector':
        """ Reverse the order of the state vector to account for endianness. 
        For example Qiskit uses little endian convention. 
        Little-endian -> Least significant qubit (LSB) is on the right. Like |q1 q0> where q0 is the LSB.
        Big-endian -> Least significant qubit (LSB) is on the left. Like |q0 q1> where q0 is the LSB.
        """
        return Statevector(self[::-1])
    
    def bloch_vector(self) -> np.ndarray:
        """
        Return the Bloch vector representation of the state vector using the (pure) density matrix.
        :return: Bloch vector representation of the quantum state
        """
        return self.density_matrix().bloch_vector()
    
    def to_qiskit(self) -> qiskit.quantum_info.Statevector:
        """
        Convert the state vector to a Qiskit QuantumCircuit object.
        Qiskit uses little-endian convention for state vectors.
        :return: Qiskit StateVector object
        """
        return qiskit.quantum_info.Statevector(self.reverse())
    
    @staticmethod
    def from_qiskit(statevector: qiskit.quantum_info.Statevector) -> "Statevector":
        """
        Convert a Qiskit StateVector object to a scikit-q StateVector.
        Qiskit uses little-endian convention for state vectors.
        :param statevector: Qiiskit StateVector object
        :return: scikit-q StateVector object
        """
        return Statevector(statevector.data[::-1])
    
    def to_pennylane(self) -> qml.QubitStateVector:
        """
        Convert the state vector to a PennyLane QubitStateVector object.
        PennyLane uses big-endian convention for state vectors.
        :return: PennyLane QubitStateVector object
        """
        return qml.QubitStateVector(self, wires=range(self.num_qubits()))
    
    @staticmethod
    def from_pennylane(statevector: qml.QubitStateVector) -> "Statevector":
        """
        Convert a PennyLane QubitStateVector object to a scikit-q StateVector.
        PennyLane uses big-endian convention for state vectors.
        :param statevector: PennyLane QubitStateVector object
        :return: scikit-q StateVector object
        """
        return Statevector(statevector.data[0])
    
class ZeroState(Statevector):
    """ Zero state |0...0> """
    def __new__(cls, num_qubits: int):
        return super().__new__(cls, [1] + [0] * (2 ** num_qubits - 1))
    
class OneState(Statevector):
    """ One state |1...1> """
    def __new__(cls, num_qubits: int):
        return super().__new__(cls, [0] * (2 ** num_qubits - 1) + [1])
    
class PlusState(Statevector):
    """ Single Qubit |+> superpoisition state """
    def __new__(cls):
        return super().__new__(cls, [1, 1] / np.sqrt(2))
    
class MinusState(Statevector):
    """ Single Qubit |-> superposition state """
    def __new__(cls):
        return super().__new__(cls, [1, -1] / np.sqrt(2))
    