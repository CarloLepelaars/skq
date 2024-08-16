from qiskit import QuantumCircuit

from skq.circuits.entangled_states import BellStates
from skq.converters.qiskit import pipeline_to_qiskit_circuit


def test_bell_state_omega_plus():
    bell_states = BellStates()
    pipeline = bell_states.get_bell_state(1)
    circuit = pipeline_to_qiskit_circuit(pipeline, num_qubits=2)

    # Expected Qiskit circuit for |Φ+⟩ = |00> + |11> / sqrt(2)
    expected_circuit = QuantumCircuit(2)
    expected_circuit.h(0)
    expected_circuit.cx(0, 1)
    assert circuit == expected_circuit, "The generated circuit does not match the expected |Φ+⟩ Bell state circuit."

def test_bell_state_omega_minus():
    bell_states = BellStates()
    pipeline = bell_states.get_bell_state(2)
    circuit = pipeline_to_qiskit_circuit(pipeline, num_qubits=2)

    # Expected Qiskit circuit for |Φ-⟩ = |00> - |11> / sqrt(2)
    expected_circuit = QuantumCircuit(2)
    expected_circuit.h(0)
    expected_circuit.cx(0, 1)
    expected_circuit.z(0)
    assert circuit == expected_circuit, "The generated circuit does not match the expected |Φ-⟩ Bell state circuit."

def test_bell_state_phi_plus():
    bell_states = BellStates()
    pipeline = bell_states.get_bell_state(3)
    circuit = pipeline_to_qiskit_circuit(pipeline, num_qubits=2)

    # Expected Qiskit circuit for |Ψ+⟩ = |01> + |10> / sqrt(2)
    expected_circuit = QuantumCircuit(2)
    expected_circuit.h(0)
    expected_circuit.cx(0, 1)
    expected_circuit.x(1)
    assert circuit == expected_circuit, "The generated circuit does not match the expected |Ψ+⟩ Bell state circuit."

def test_bell_state_phi_minus():
    bell_states = BellStates()
    pipeline = bell_states.get_bell_state(4)
    circuit = pipeline_to_qiskit_circuit(pipeline, num_qubits=2)

    # Expected Qiskit circuit for |Ψ-⟩ = |01> - |10> / sqrt(2)
    expected_circuit = QuantumCircuit(2)
    expected_circuit.h(0)
    expected_circuit.cx(0, 1)
    expected_circuit.z(0)
    expected_circuit.x(1)
    assert circuit == expected_circuit, "The generated circuit does not match the expected |Ψ-⟩ Bell state circuit."
