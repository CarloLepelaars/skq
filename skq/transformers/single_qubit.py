from skq.gates.base import Gate
from skq.transformers.base import BaseQubitTransformer


class SingleQubitTransformer(BaseQubitTransformer):
    """
    Transformer using a single qubit gate to transform input state vectors.
    :param gate: A valid skq quantum Gate object
    :param qubits: The qubit index to apply the gate to.
    """
    def __init__(self, gate: Gate, qubits: int):
        assert gate.shape == (2, 2), "Single Qubit Gate must be a 2x2 matrix"
        assert isinstance(qubits, int), "Single Qubit Transformer must have a single qubit integer index."
        super().__init__(gate=gate, qubits=qubits)