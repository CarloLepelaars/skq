from qiskit import QuantumCircuit

# from src.circuits.entangled_states import BellStates
# from src.converters.qiskit import pipeline_to_qiskit_circuit
# from src.converters.openqasm import pipeline_to_qasm

# TODO Test BellStates conversion with fastcore
# def test_qiskit_bell_state():
#     bell_states = BellStates()
#     pipeline = bell_states.get_bell_state(4)
#     circuit = pipeline_to_qiskit_circuit(pipeline, num_qubits=2)

#     # Expected Qiskit circuit for |Ψ-⟩ = |01> - |10> / sqrt(2)
#     expected_circuit = QuantumCircuit(2)
#     expected_circuit.h(0)
#     expected_circuit.cx(0, 1)
#     expected_circuit.z(0)
#     expected_circuit.x(1)
#     assert circuit == expected_circuit, "The generated circuit does not match the expected Phi Minus circuit."

# def test_qasm_bell_state():
#     bell_states = BellStates()
#     pipeline = bell_states.get_bell_state(3)
#     qasm_code = pipeline_to_qasm(pipeline, num_qubits=2, measure_all=True)
#     # Expected OpenQASM representation for |Ψ+⟩ = |01> + |10> / sqrt(2)
#     expected_qasm_code = """OPENQASM 3.0;
# include "stdgates.inc";
# qreg q[2];
# creg c[2];
# h q[0];
# cx q[0], q[1];
# x q[1];
# measure q -> c;"""
#     print(qasm_code)
#     assert qasm_code == expected_qasm_code, "The generated QASM code does not match the expected Phi Plus Bell state QASM code."
