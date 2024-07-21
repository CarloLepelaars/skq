import qiskit
import numpy as np

from skq.density import DensityMatrix


class Statevector(np.ndarray):
    """
    State vector representation for a quantum state.
    """
    def __new__(cls, input_array):
        arr = np.asarray(input_array)
        obj = arr.view(cls)
        assert obj.is_1d(), "State vector must be 1D."
        assert obj.is_normalized(), "State vector must be normalized."
        return obj
    
    def is_1d(self) -> bool:
        """ Check if the state vector is 1D. """
        return self.ndim == 1
    
    def is_normalized(self) -> bool:
        """ Check if the state vector is normalized: ||ψ|| = 1. """
        return np.isclose(np.linalg.norm(self), 1)
    
    def density_matrix(self) -> DensityMatrix:
        """ Return the density matrix representation of the state vector. """
        return DensityMatrix(np.outer(self, self.conj()))
    
    def measure(self) -> int:
        """ Simulate a measurement of the state vector using the amplitudes. """
        probabilities = np.abs(self) ** 2
        return np.random.choice(len(self), p=probabilities)
    
    def to_qiskit(self) -> qiskit.quantum_info.Statevector:
        """
        Convert the state vector to a Qiskit QuantumCircuit object.
        :return: QuantumCircuit object representing the state vector
        """
        return qiskit.quantum_info.Statevector(self)
    
    @staticmethod
    def from_qiskit(statevector: qiskit.quantum_info.Statevector) -> "Statevector":
        """
        Convert a Qiskit QuantumCircuit object to a state vector.
        :param statevector: QuantumCircuit object representing the state vector
        :return: State vector representation of the quantum state
        """
        return Statevector(statevector.data)
    