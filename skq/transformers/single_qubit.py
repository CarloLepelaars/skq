from skq.gates.base import Gate
from skq.gates.single_qubit import *
from skq.transformers.base import BaseQubitTransformer


class SingleQubitTransformer(BaseQubitTransformer):
    """
    Transformer using a single qubit gate to transform input state vectors.
    :param gate: A valid skq quantum Gate object
    :param qubit: The qubit index to apply the gate to.
    """
    def __init__(self, gate: Gate, *, qubits: int):
        assert gate.shape == (2, 2), "Single Qubit Gate must be a 2x2 matrix"
        assert isinstance(qubits, int), "Single Qubit Transformer must have a single qubit integer index."
        super().__init__(gate=gate, qubits=qubits)

class ITransformer(SingleQubitTransformer):
    """ Identity gate """
    def __init__(self, *, qubits: int):
        super().__init__(IGate(), qubits=qubits)

class XTransformer(SingleQubitTransformer):
    """ Pauli X (NOT) gate"""
    def __init__(self, *, qubits: int):
        super().__init__(XGate(), qubits=qubits)

class YTransformer(SingleQubitTransformer):
    """ Pauli Y gate """
    def __init__(self, *, qubits: int):
        super().__init__(YGate(), qubits=qubits)

class ZTransformer(SingleQubitTransformer):
    """ Pauli Z gate """
    def __init__(self, *, qubits: int):
        super().__init__(ZGate(), qubits=qubits)

class HTransformer(SingleQubitTransformer):
    """ Hadamard gate """
    def __init__(self, *, qubits: int):
        super().__init__(HGate(), qubits=qubits)

class PhaseTransformer(SingleQubitTransformer):
    """ Generalized phase shift gate """
    def __init__(self, theta: float, *, qubits: int):
        self.theta = theta
        super().__init__(PhaseGate(theta), qubits=qubits)

class TTransformer(SingleQubitTransformer):
    """ T gate: phase shift gate with theta = pi/4 """
    def __init__(self, *, qubits: int):
        super().__init__(TGate(), qubits=qubits)

class STransformer(SingleQubitTransformer):
    """ S gate: phase shift gate with theta = pi/2 """
    def __init__(self, *, qubits: int):
        super().__init__(SGate(), qubits=qubits)

class RXTransformer(SingleQubitTransformer):
    """ Rotation around the X-axis """
    def __init__(self, theta: float, *, qubits: int):
        super().__init__(RXGate(theta), qubits=qubits)

class RYTransformer(SingleQubitTransformer):
    """ Rotation around the Y-axis """
    def __init__(self, theta: float, *, qubits: int):
        self.theta = theta
        super().__init__(RYGate(theta), qubits=qubits)

class RZTransformer(SingleQubitTransformer):
    """ Rotation around the Z-axis """
    def __init__(self, theta: float, *, qubits: int):
        self.theta = theta
        super().__init__(RZGate(theta), qubits=qubits)

class U3Transformer(SingleQubitTransformer):
    """ Rotation around 3-axes """
    def __init__(self, theta: float, phi: float, lam: float, *, qubits: int):
        self.theta = theta
        self.phi = phi
        self.lam = lam
        super().__init__(U3Gate(theta, phi, lam), qubits=qubits)
