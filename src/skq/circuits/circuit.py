import numpy as np
from qiskit import QuantumCircuit


from ..base import Operator


class Circuit(list):
    """Run multiple gates in sequence."""

    def encodes(self, x):
        for gate in self:
            x = gate.encodes(x)
        return x

    def decodes(self, x):
        for gate in reversed(self):
            x = gate.decodes(x)
        return x

    def __call__(self, x):
        return self.encodes(x)

    def convert(self, total_qubits: int, framework="qiskit"):
        """Convert the circuit to a given framework.
        :param framework: Framework to convert to.
        :param total_qubits: Total number of qubits in the circuit.
        :return: Converted circuit.
        """
        if framework == "qiskit":
            return QiskitConverter().convert(self, total_qubits)
        else:
            raise NotImplementedError(f"Conversion to framework '{framework}' is not supported.")


class Concat:
    """
    Combine multiple gates into a single gate.
    :param gates: List of gates to concatenate.
    """

    def __init__(self, gates: list[Operator]):
        assert len(gates) > 1, "Concat must have at least 2 gates."
        assert all(isinstance(g, Operator) for g in gates), "All gates must be instances of Operator."
        self.gates = gates
        self.encoding_matrix = np.kron(*[g for g in gates])

    def encodes(self, x: np.ndarray) -> np.ndarray:
        """
        Concatenate 2 or more gates.

        :param x: Quantum state to encode.
        :return: Quantum state after encoding.
        """
        return x @ self.encoding_matrix

    def decodes(self, x: np.ndarray) -> np.ndarray:
        """
        Reverse propagation for all gates.

        :param x: Quantum state to decode.
        :return: Quantum state after decoding.
        """
        for g in reversed(self.gates):
            x = x @ np.kron(g.conj().T, np.eye(len(x) // g.shape[0]))
        return x

    def __call__(self, x):
        return self.encodes(x)


class QiskitConverter:
    """Convert a skq Circuit into a Qiskit QuantumCircuit."""

    def convert(self, circuit: Circuit, total_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(total_qubits)
        for gate in circuit:
            if isinstance(gate, Concat):
                for i, sub_gate in enumerate(gate.gates):
                    if hasattr(sub_gate, "to_qiskit"):
                        qgate = sub_gate.to_qiskit()
                        qc.append(qgate, [i])
                    else:
                        raise ValueError(f"Gate {sub_gate.__class__.__name__} does not implement to_qiskit().")
            else:
                if hasattr(gate, "to_qiskit"):
                    qgate = gate.to_qiskit()
                    n = gate.num_qubits() if hasattr(gate, "num_qubits") else 1
                    if n == 1:
                        qc.append(qgate, [0])
                    else:
                        qc.append(qgate, list(range(n)))
                else:
                    raise ValueError(f"Gate {gate.__class__.__name__} does not implement to_qiskit().")
        return qc
