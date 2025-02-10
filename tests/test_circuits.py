import numpy as np
import pytest

from qiskit.circuit.library import HGate, IGate
from qiskit.quantum_info import Operator as QiskitOperator

from skq.circuits.circuit import Circuit, Concat
from skq.circuits.entangled_states import BellStates
from skq.gates.qubit import I, H
from skq.gates.base import BaseGate


def test_circuit_basic_operation():
    """Test that a Circuit can execute a simple sequence of gates"""
    circuit = Circuit([H()])
    initial_state = np.array([1, 0])  # |0⟩ state
    result = circuit(initial_state)
    expected = np.array([1, 1]) / np.sqrt(2)  # |+⟩ state
    np.testing.assert_array_almost_equal(result, expected)


def test_concat_two_gates():
    """Test that Concat correctly combines two single-qubit gates"""
    concat = Concat([I(), I()])
    initial_state = np.array([1, 0, 0, 0])  # |00⟩ state
    result = concat.encodes(initial_state)
    expected = initial_state  # Should remain unchanged for identity gates
    np.testing.assert_array_almost_equal(result, expected)


def test_bell_state_omega_plus():
    """Test creation of the first Bell state (Φ+)"""
    bell = BellStates()
    circuit = bell.get_bell_state(1)
    initial_state = np.array([1, 0, 0, 0])  # |00⟩ state
    result = circuit(initial_state)
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
    np.testing.assert_array_almost_equal(result, expected)


def test_convert_single_gate():
    """Conversion of a circuit with a single gate (H) into a Qiskit circuit."""
    circuit = Circuit([H()])
    qc = circuit.convert(total_qubits=1, framework="qiskit")
    assert qc.num_qubits == 1, "Expected 1 qubit in the Qiskit circuit."
    assert len(qc.data) == 1, "Expected a single instruction in the Qiskit circuit."
    instr = qc.data[0].operation
    assert isinstance(instr, HGate), "Converted gate should be an instance of HGate."


def test_convert_concat_gate():
    """Conversion of a circuit with a Concat gate, here combining two I (identity) gates."""
    circuit = Circuit([Concat([H(), I()])])
    qc = circuit.convert(total_qubits=2, framework="qiskit")
    assert qc.num_qubits == 2, "Expected 2 qubits in the Qiskit circuit from Concat."
    assert len(qc.data) == 2, "Expected 2 instructions from the Concat gate."
    for datum in qc.data:
        instr = datum.operation
        assert isinstance(instr, (IGate, HGate)), "Each converted gate should be an instance of IGate or HGate."


def test_convert_bell_state():
    """Conversion of a Bell state circuit to Qiskit."""
    bell = BellStates()
    circuit = bell.get_bell_state(1)
    qc = circuit.convert(total_qubits=2, framework="qiskit")
    assert qc.num_qubits == 2, "Expected Bell state circuit to operate on 2 qubits."
    qc_matrix = QiskitOperator(qc).data
    initial_state = np.array([1, 0, 0, 0], dtype=complex)
    result = qc_matrix @ initial_state
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_array_almost_equal(result, expected)


def test_convert_error_for_missing_to_qiskit():
    """Conversion raises a clear error when a gate does not implement to_qiskit."""

    class DummyGate(BaseGate):
        def __new__(cls, matrix):
            return super().__new__(cls, matrix)

    dummy_gate = DummyGate(np.eye(2))
    circuit = Circuit([dummy_gate])
    with pytest.raises(NotImplementedError):
        circuit.convert(total_qubits=1, framework="qiskit")


def test_convert_unsupported_framework():
    """
    Test that requesting conversion to an unsupported framework (e.g. 'pennylane')
    raises a NotImplementedError.
    """
    circuit = Circuit([H()])
    with pytest.raises(NotImplementedError):
        circuit.convert(total_qubits=1, framework="pennylane")
