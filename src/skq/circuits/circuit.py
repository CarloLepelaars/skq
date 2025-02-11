import numpy as np
from qiskit import QuantumCircuit


from ..base import Operator


class Circuit(list):
    """Run multiple qubit gates in sequence."""

    @property
    def num_qubits(self) -> int:
        return max(g.num_qubits for g in self)

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

    def convert(self, framework="qiskit"):
        """Convert circuit to a given framework.
        :param framework: Framework to convert to.
        :return: Converter Circuit object.
        For Qiskit -> QuantumCircuit object.
        For QASM -> OpenQASM string.
        """
        converter_mapping = {"qiskit": convert_to_qiskit, "qasm": convert_to_qasm}
        assert framework in converter_mapping, f"Invalid framework. Supported frameworks: {converter_mapping.keys()}."
        return converter_mapping[framework](self)
    
    def draw(self, **kwargs):
        """Draw circuit using Qiskit."""
        return self.convert(framework="qiskit").draw(**kwargs)


class Concat:
    """
    Combine multiple qubit gates into a single gate.
    :param gates: List of gates to concatenate.
    """

    def __init__(self, gates: list[Operator]):
        assert len(gates) > 1, "Concat must have at least 2 gates."
        assert all(isinstance(g, Operator) for g in gates), "All gates must be instances of Operator."
        self.gates = gates
        self.encoding_matrix = np.kron(*[g for g in gates])
        self.num_qubits = sum(g.num_qubits for g in gates)

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


def convert_to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """Convert a skq Circuit into a Qiskit QuantumCircuit."""
    qc = QuantumCircuit(circuit.num_qubits)
    for gate in circuit:
        if isinstance(gate, Concat):
            for i, sub_gate in enumerate(gate.gates):
                qgate = sub_gate.to_qiskit()
                qc.append(qgate, [i])
        else:
            qgate = gate.to_qiskit()
            qc.append(qgate, list(range(gate.num_qubits)))
    return qc


def convert_to_qasm(circuit: Circuit) -> str:
    """Convert a skq Circuit into an OpenQASM string."""
    qasm_lines = []
    for gate in circuit:
        if isinstance(gate, Concat):
            for i, sub_gate in enumerate(gate.gates):
                qasm_lines.append(sub_gate.to_qasm([i]))
        else:
            qasm_lines.append(gate.to_qasm(list(range(gate.num_qubits))))
    return "\n".join(qasm_lines)
