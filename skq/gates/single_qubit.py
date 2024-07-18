import qiskit
import numpy as np

from skq.gates.base import Gate


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
    """ Rotation around 3-axes. Single qubit gate."""
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
    

