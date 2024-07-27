import qiskit
import numpy as np

from skq.quantum_info import Statevector
from skq.gates.qubit.base import QubitGate
from skq.gates.qubit.single import XGate, YGate, ZGate, HGate

class PhaseOracleGate(QubitGate):
    """
    Phase Oracle as used in Grover's search algorithm.
    :param target_state: The target state to mark.
    target_state is assumed to be in Big-Endian format.
    """
    def __new__(cls, target_state: np.ndarray):
        state = Statevector(target_state)
        n_qubits = state.num_qubits()
        identity = MIGate(n_qubits)
        target_index = np.argmax(target_state)
        phase_inversion = MIGate(n_qubits)
        phase_inversion[target_index, target_index] = -1
        oracle_matrix = identity - 2 * np.outer(target_state, target_state.conj())
        return super().__new__(cls, oracle_matrix)
    
    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        # Reverse the order of qubits for Qiskit's little-endian convention
        reversed_matrix = self[::-1, ::-1]
        return qiskit.circuit.library.UnitaryGate(reversed_matrix, label='PhaseOracle')
    
class EqualSuperpositionGate(QubitGate):
    """
    Equal Superposition Matrix Gate used in Grover's diffusion operator.
    :param n_qubits: The number of qubits in the system.
    """
    def __new__(cls, n_qubits: int):
        assert n_qubits >= 1, "EqualSuperpositionGate must have at least one qubit."
        size = 2 ** n_qubits
        H = np.ones((size, size)) / size
        return super().__new__(cls, H)
    
    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        # Reverse the order of qubits for Qiskit's little-endian convention
        reversed_matrix = self[::-1, ::-1]
        return qiskit.circuit.library.UnitaryGate(reversed_matrix, label='EqualSuperposition')
    
class GroverDiffusionGate(QubitGate):
    """
    Grover Diffusion Operator Gate as used in Grover's search algorithm.
    This gate amplifies the amplitude of the marked state.
    :param n_qubits: The number of qubits in the system.
    """
    def __new__(cls, n_qubits: int):
        assert n_qubits >= 1, "GroverDiffusionGate must have at least one qubit."
        diffusion_matrix = 2 * EqualSuperpositionGate(n_qubits) - MIGate(n_qubits)
        return super().__new__(cls, diffusion_matrix)
    
    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        # Reverse the order of qubits for Qiskit's little-endian convention
        reversed_matrix = self[::-1, ::-1]
        return qiskit.circuit.library.UnitaryGate(reversed_matrix, label='GroverDiffusion')
    
class MIGate(QubitGate):
    """ 
    Multi-qubit identity gate. 
    :param num_qubits: Number of qubits in the system.
    """
    def __new__(cls, num_qubits: int):
        assert num_qubits >= 1, "MultiIGate must have at least one qubit."
        return super().__new__(cls, np.eye(2 ** num_qubits))
    
    def to_qiskit(self) -> qiskit.circuit.library.UnitaryGate:
        # No endianness conversion needed for the identity gate
        return qiskit.circuit.library.UnitaryGate(self, label=f'I_{self.num_qubits()}q')

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
        if self.num_ctrl_qubits == 1:
            return qiskit.circuit.library.CXGate()
        elif self.num_ctrl_qubits == 2:
            return qiskit.circuit.library.CCXGate()
        elif self.num_ctrl_qubits == 3:
            return qiskit.circuit.library.C3XGate()
        elif self.num_ctrl_qubits == 4:
            return qiskit.circuit.library.C4XGate()
        else:
            return qiskit.circuit.library.MCXGate(num_ctrl_qubits=self.num_ctrl_qubits)
            
class MCYGate(QubitGate):
    """ Multi controlled-Y (MCY) gate. """
    def __new__(cls, num_ctrl_qubits: int):
        assert num_ctrl_qubits >= 1, "MCY gate must have at least one control qubit."
        cls.num_ctrl_qubits = num_ctrl_qubits
        levels = 2 ** (num_ctrl_qubits + 1)
        gate = np.identity(levels)
        gate[-2:, -2:] = YGate()
        return super().__new__(cls, gate)
    
    def to_qiskit(self) -> qiskit.circuit.library.CYGate | qiskit.circuit.ControlledGate:
        if self.num_ctrl_qubits == 1:
            return qiskit.circuit.library.CYGate()
        else:
            return qiskit.cicuit.ControlledGate(name="mcy",
                                                        num_qubits=self.num_qubits(),
                                                        params=[],
                                                        num_ctrl_qubits=self.num_ctrl_qubits,
                                                        base_gate=YGate())

class MCZGate(QubitGate):
    """ Multi controlled-Z (MCZ) gate. """
    def __new__(cls, num_ctrl_qubits: int):
        assert num_ctrl_qubits >= 1, "MCZ gate must have at least one control qubit."
        cls.num_ctrl_qubits = num_ctrl_qubits
        levels = 2 ** (num_ctrl_qubits + 1)
        gate = np.identity(levels)
        gate[-2:, -2:] = ZGate()
        return super().__new__(cls, gate)
    
    def to_qiskit(self) -> qiskit.circuit.library.CZGate | qiskit.circuit.ControlledGate:
        if self.num_ctrl_qubits == 1:
            return qiskit.circuit.library.CZGate()
        elif self.num_ctrl_qubits == 2:
            return qiskit.circuit.library.CCZGate()
        else:
            return qiskit.cicuit.ControlledGate(name="mcz",
                                                num_qubits=self.num_qubits(),
                                                params=[],
                                                num_ctrl_qubits=self.num_ctrl_qubits,
                                                base_gate=ZGate())
    
# Aliases for gates
ToffoliGate = CCXGate
FredkinGate = CSwapGate
