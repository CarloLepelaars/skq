from skq.gates.base import Gate
from skq.transformers.base import BaseQubitTransformer


class MultiQubitTransformer(BaseQubitTransformer):
    """ 
    Transformer which involves multiple qubits, like CNOT, SWAP, etc. 
    :param gate: A valid skq quantum Gate object
    :param qubits: List of qubit indices to apply the gate to.
    """
    def __init__(self, gate: Gate, qubits: list):
        assert gate.shape[0] >= 4 and gate.shape[1] >= 4, "Multi Qubit Gate must be a matrix of at least 4x4"
        super().__init__(gate=gate, qubits=qubits)
        assert isinstance(qubits, list), "Multi Qubit Transformer must have a list of qubit indices."
        assert len(qubits) == gate.num_qubits(), f"Number of qubits in gate ({gate.num_qubits()}) must match the number of defined qubits in MultiQubitTransformer ({len(qubits)})."
        