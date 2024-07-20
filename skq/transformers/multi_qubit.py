from skq.gates.qubit.base import QubitGate
from skq.gates.qubit.multi import *
from skq.transformers.base import BaseQubitTransformer


class MultiQubitTransformer(BaseQubitTransformer):
    """ 
    Transformer which involves multiple qubits, like CNOT, SWAP, etc. 
    :param gate: A valid skq quantum Gate object
    :param qubits: List of qubit indices to apply the gate to.
    """
    def __init__(self, gate: QubitGate, *, qubits: list[int]):
        assert gate.shape[0] >= 4 and gate.shape[1] >= 4, "Multi Qubit Gate must be a matrix of at least 4x4"
        super().__init__(gate=gate, qubits=qubits)
        assert isinstance(qubits, list), "MultiQubitTransformer must be provided with a list of qubit indices."
        assert len(qubits) == gate.num_qubits(), f"Number of qubits in gate ({gate.num_qubits()}) must match the number of defined qubits in MultiQubitTransformer ({len(qubits)})."

class CXTransformer(MultiQubitTransformer):
    """ Controlled-X (CNOT) gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "CXTransformer requires 2 defined qubits."
        super().__init__(CXGate(), qubits=qubits)

class CYTransformer(MultiQubitTransformer):
    """ Controlled-Y gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "CYTransformer requires 2 defined qubits."
        super().__init__(CYGate(), qubits=qubits)

class CZTransformer(MultiQubitTransformer):
    """ Controlled-Z gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "CZTransformer requires 2 defined qubits."
        super().__init__(CZGate(), qubits=qubits)

class CHTransformer(MultiQubitTransformer):
    """ Controlled-Hadamard gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "CHTransformer requires 2 defined qubits."
        super().__init__(CHGate(), qubits=qubits)

class CPhaseTransformer(MultiQubitTransformer):
    """ General controlled phase shift gate """
    def __init__(self, theta: float, *, qubits: list[int]):
        assert len(qubits) == 2, "CPhaseTransformer requires 2 defined qubits."
        super().__init__(CPhaseGate(theta), qubits=qubits)

class CSTransformer(MultiQubitTransformer):
    """ Controlled-S gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "CSTransformer requires 2 defined qubits."
        super().__init__(CSGate(), qubits=qubits)

class CTTransformer(MultiQubitTransformer):
    """ Controlled-T gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "CTTransformer requires 2 defined qubits."
        super().__init__(CTGate(), qubits=qubits)

class SWAPTransformer(MultiQubitTransformer):
    """ SWAP Gate. Swaps states of two qubits """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 2, "SWAPTransformer requires 2 defined qubits."
        super().__init__(SWAPGate(), qubits=qubits)

class CCXTransformer(MultiQubitTransformer):
    """ Toffoli gate: A 3-qubit controlled-controlled-X (CCX) gate. """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 3, "CCXTransformer requires 3 defined qubits."
        super().__init__(CCXGate(), qubits=qubits)

class CSwapTransformer(MultiQubitTransformer):
    """ Controlled-SWAP gate """
    def __init__(self, *, qubits: list[int]):
        assert len(qubits) == 3, "CSwapTransformer requires 3 defined qubits."
        super().__init__(CSwapGate(), qubits=qubits)
