import qiskit
import numpy as np
import pennylane as qml

from skq.gates.base import BaseGate
from skq.gates.qubit.base import CustomQubitGate


class QutritGate(BaseGate):
    """ 
    Base class for Qutrit gates. 
    These are quantum systems with a basis of 3 states. |0>, |1>, |2>. 
    Analogous to spin 1 particles.
    """
    def __new__(cls, input_array: np.array):
        obj = super().__new__(cls, input_array)
        assert obj.is_at_least_nxn(n=3), "Gate must be at least a 3x3 matrix"
        assert obj.is_power_of_n_shape(n=3), "Gate shape must be a power of 3"
        return obj
    
    def num_qutrits(self) -> int:
        """ Return the number of qutrits involved in the gate. """
        return int(np.log(self.shape[0]) / np.log(3))
    
    def is_multi_qutrit(self) -> bool:
        """ Check if the gate involves multiple qutrits. """
        return self.num_qutrits() > 1
    
    def qutrit_to_qubit(self) -> CustomQubitGate:
        """ 
        Convert the qutrit gate to an equivalent qubit gate. 
        :return: CustomQubitGate object
        """
        num_qutrits = self.num_qutrits()
        dim_qutrit = 3 ** num_qutrits
        num_qubits = int(np.ceil(num_qutrits * np.log2(3)))
        dim_qubit = 2 ** num_qubits
        qubit_gate = np.eye(dim_qubit, dtype=complex)
        qubit_gate[:dim_qutrit, :dim_qutrit] = self
        return CustomQubitGate(qubit_gate)
    
    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        """ 
        Convert Qutrit gate to a Qiskit Gate object. 
        Qiskit only supports qubit gates, so we convert the qutrit gate to a qubit gate first.
        :return: Qiskit UnitaryGate object
        """
        qubit_gate = self.qutrit_to_qubit()
        return qubit_gate.to_qiskit()
    
    def to_pennylane(self, wires: list[int] | int= None) -> qml.QubitUnitary:
        """ 
        Convert gate to a PennyLane QubitUnitary. 
        PennyLane only supports qubit gates, so we convert the qutrit gate to a qubit gate first.
        :param wires: List of wires the gate acts on
        :return: PennyLane QubitUnitary object
        """
        qubit_gate = self.qutrit_to_qubit()
        return qubit_gate.to_pennylane(wires=wires)
