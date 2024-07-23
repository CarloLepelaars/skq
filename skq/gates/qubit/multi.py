import qiskit
import numpy as np

from skq.gates.qubit.base import QubitGate
from skq.gates.qubit.single import XGate, YGate, ZGate, HGate


class CXGate(QubitGate):
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

class CYGate(QubitGate):
    """ Controlled-Y gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, -1j], 
                                     [0, 0, 1j, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CYGate:
        return qiskit.circuit.library.CYGate()
    
class CZGate(QubitGate):
    """ Controlled-Z gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 0, 0, -1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CZGate:
        return qiskit.circuit.library.CZGate()
    
class CHGate(QubitGate):
    """ Controlled-Hadamard gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], 
                                     [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CHGate:
        return qiskit.circuit.library.CHGate()

class CPhaseGate(QubitGate):
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
    
class SWAPGate(QubitGate):
    """ Swap gate. Swaps the states of two qubits. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.SwapGate:
        return qiskit.circuit.library.SwapGate()
    
class CSwapGate(QubitGate):
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
    
class CCXGate(QubitGate):
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
    
class CCYGate(QubitGate):
    """ A 3-qubit controlled-controlled-Y (CCY) gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, -1j], 
                                     [0, 0, 0, 0, 0, 0, 1j, 0]])
    
    def to_qiskit(self) -> qiskit.circuit.ControlledGate:
        # There is no native CCY gate in Qiskit so we construct it.
        return qiskit.circuit.ControlledGate(name="ccy",
                                                     num_qubits=3,
                                                     params=[],
                                                     num_ctrl_qubits=2, 
                                                     base_gate=YGate())
    
class CCZGate(QubitGate):
    """ A 3-qubit controlled-controlled-Z (CCZ) gate. """
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 1, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, -1]])
    
    def to_qiskit(self) -> qiskit.circuit.library.CCZGate:
        return qiskit.circuit.library.CCZGate()
    
class MCXGate(QubitGate):
    """ 
    Multi controlled-X (MCX) gate. 
    :param num_ctrl_qubits: Number of control qubits.
    """
    def __new__(cls, num_ctrl_qubits: int):
        assert num_ctrl_qubits >= 1, "MCX gate must have at least one control qubit."
        cls.num_ctrl_qubits = num_ctrl_qubits
        levels = 2 ** (num_ctrl_qubits + 1)
        gate = np.identity(levels)
        gate[-2:, -2:] = XGate()
        return super().__new__(cls, gate)
    
    def to_qiskit(self) -> qiskit.circuit.library.CXGate | qiskit.circuit.library.CCXGate | qiskit.circuit.library.C3XGate | qiskit.circuit.library.C4XGate | qiskit.circuit.library.MCXGate:
        match self.num_ctrl_qubits:
            case 1:
                return qiskit.circuit.library.CXGate()
            case 2:
                return qiskit.circuit.library.CCXGate()
            case 3:
                return qiskit.circuit.library.C3XGate()
            case 4:
                return qiskit.circuit.library.C4XGate()
            case _:
                return qiskit.circuit.library.MCXGate(self.num_ctrl_qubits)
    
# Aliases for gates
ToffoliGate = CCXGate
FredkinGate = CSwapGate
