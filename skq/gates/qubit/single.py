import qiskit
import numpy as np
import pennylane as qml

from skq.gates.qubit.base import QubitGate


class IGate(QubitGate):
    """ 
    Identity gate: 
    [[1, 0]
    [0, 1]]
    """
    def __new__(cls):
        return super().__new__(cls, np.eye(2))
    
    def to_qiskit(self) -> qiskit.circuit.library.IGate:
        return qiskit.circuit.library.IGate()
    
    def to_pennylane(self, wires: list[int] | int) -> qml.I:
        return qml.I(wires=wires)
    
class XGate(QubitGate):
    """ Pauli X (NOT) Gate. """
    def __new__(cls):
        return super().__new__(cls, [[0, 1], 
                                     [1, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.XGate:
        return qiskit.circuit.library.XGate()
    
    def to_pennylane(self, wires: list[int] | int) -> qml.X:
        return qml.X(wires=wires)

class YGate(QubitGate):
    """ Pauli Y gate. """
    def __new__(cls):
        return super().__new__(cls, [[0, -1j], 
                                     [1j, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.YGate:
        return qiskit.circuit.library.YGate()
    
    def to_pennylane(self, wires: list[int] | int) -> qml.Y:
        return qml.Y(wires=wires)
    
class ZGate(QubitGate):
    """ Pauli Z gate. 
    Special case of a phase shift gate with phi = pi.
    """
    def __new__(cls):
        return super().__new__(cls, [[1, 0], 
                                     [0, -1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.ZGate:
        return qiskit.circuit.library.ZGate()
    
    def to_pennylane(self, wires: list[int] | int) -> qml.Z:
        return qml.Z(wires=wires)

class HGate(QubitGate):
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
    
    def to_pennylane(self, wires: list[int] | int) -> qml.Hadamard:
        return qml.Hadamard(wires=wires)
    
class PhaseGate(QubitGate):
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
    
    def to_qiskit(self) -> qiskit.circuit.library.PhaseGate:
        return qiskit.circuit.library.PhaseGate(self.phi)
    
    def to_pennylane(self, wires: list[int] | int) -> qml.PhaseShift:
        return qml.PhaseShift(phi=self.phi, wires=wires)
    
class TGate(PhaseGate):
    """ T gate: phase shift gate with phi = pi / 4. """
    def __new__(cls):
        phi = np.pi / 4
        return super().__new__(cls, phi=phi)
    
    def to_qiskit(self) -> qiskit.circuit.library.TGate:
        return qiskit.circuit.library.TGate()
    
    def to_pennylane(self, wires: list[int] | int) -> qml.PhaseShift:
        return qml.T(wires=wires)
    
class SGate(PhaseGate):
    """ S gate: phase shift gate with phi = pi / 2. """
    def __new__(cls):
        phi = np.pi / 2
        return super().__new__(cls, phi=phi)
    
    def to_qiskit(self) -> qiskit.circuit.library.SGate:
        return qiskit.circuit.library.SGate()
    
    def to_pennylane(self, wires: list[int] | int) -> qml.S:
        return qml.S(wires=wires)
    
class RXGate(QubitGate):
    """ Generalized X rotation gate. """
    def __new__(cls, phi):
        obj = super().__new__(cls, [[np.cos(phi / 2), -1j * np.sin(phi / 2)], 
                                     [-1j * np.sin(phi / 2), np.cos(phi / 2)]])
        obj.phi = phi
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.RXGate:
        return qiskit.circuit.library.RXGate(self.phi)
    
    def to_pennylane(self, wires: list[int] | int = None) -> qml.RX:
        return qml.RX(phi=self.phi, wires=wires)
    
class RYGate(QubitGate):
    """ Generalized Y rotation gate. """
    def __new__(cls, phi):
        obj = super().__new__(cls, [[np.cos(phi / 2), -np.sin(phi / 2)], 
                                     [np.sin(phi / 2), np.cos(phi / 2)]])
        obj.phi = phi
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.RYGate:
        return qiskit.circuit.library.RYGate(self.phi)
    
    def to_pennylane(self, wires: list[int] | int) -> qml.RY:
        return qml.RY(phi=self.phi, wires=wires)
    
class RZGate(QubitGate):
    """ Generalized Z rotation gate. """
    def __new__(cls, phi):
        obj = super().__new__(cls, [[np.exp(-1j * phi / 2), 0], 
                                     [0, np.exp(1j * phi / 2)]])
        obj.phi = phi
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.RZGate:
        return qiskit.circuit.library.RZGate(self.phi)
    
    def to_pennylane(self, wires: list[int] | int) -> qml.RZ:
        return qml.RZ(phi=self.phi, wires=wires)

class U3Gate(QubitGate):
    """ 
    Rotation around 3-axes. Single qubit gate.
    :param theta: Rotation angle around X-axis
    :param phi: Rotation angle around Y-axis
    :param delta: Rotation angle around Z-axis
    """
    def __new__(cls, theta: float, phi: float, delta: float):
        # Rotation matrices
        Rx = RXGate(theta)
        Ry = RYGate(phi)
        Rz = RZGate(delta)
        combined_matrix = Rz @ Ry @ Rx
        
        obj = super().__new__(cls, combined_matrix)
        obj.theta = theta
        obj.phi = phi
        obj.delta = delta
        return obj
    
    def to_qiskit(self) -> qiskit.circuit.library.U3Gate:
        return qiskit.circuit.library.U3Gate(self.theta, self.phi, self.delta)
    
    def to_pennylane(self, wires: list[int] | int) -> qml.U3:
        return qml.U3(theta=self.theta, phi=self.phi, delta=self.delta, wires=wires)
    

