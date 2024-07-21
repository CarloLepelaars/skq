import qiskit
import numpy as np

from skq.density import DensityMatrix


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
    
    def to_qiskit(self) -> qiskit.quantum_info.Statevector:
        """
        Convert the state vector to a Qiskit QuantumCircuit object.
        :return: QuantumCircuit object representing the state vector
        """
        return qiskit.quantum_info.Statevector(self.reverse())
    
    @staticmethod
    def from_qiskit(statevector: qiskit.quantum_info.Statevector) -> "Statevector":
        """
        Convert a Qiskit QuantumCircuit object to a state vector.
        :param statevector: QuantumCircuit object representing the state vector
        :return: State vector representation of the quantum state
        """
        return Statevector(statevector.data[::-1])
    