from qiskit import QuantumCircuit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from skq.transformers import SingleQubitTransformer, MultiQubitTransformer, MeasurementTransformer

def pipeline_to_qiskit_circuit(pipeline: Pipeline, num_qubits: int, measure_all=False) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    def append_gate(transformer, circuit):
        if isinstance(transformer, ColumnTransformer):
            raise NotImplementedError("The usage of ColumnTransformer for converting to Qiskit circuit is not supported yet.")
        # Recurse over underlying Pipeline steps
        elif isinstance(transformer, Pipeline):
            for _, step in transformer.steps:
                append_gate(step, circuit)
            return
        # Recurse over underlying FeatureUnion steps. This includes QuantumFeatureUnion objects.
        elif isinstance(transformer, (FeatureUnion)):
            for _, step in transformer.transformer_list:
                append_gate(step, circuit)
            return
        elif isinstance(transformer, (SingleQubitTransformer, MultiQubitTransformer)):
            # Identity operations can be skipped in the circuit
            if transformer.__class__.__name__ == "ITransformer":
                return
            # Valid transformer. Add gate to Qiskit circuit
            qiskit_gate = transformer.gate.to_qiskit()
            circuit.append(qiskit_gate, transformer.qubits)
        # Not a valid transformer. Skip operation.
        else:
            return
        
    for _, step in pipeline.steps:
        # Add measurement to Qiskit circuit if defined
        if isinstance(step, MeasurementTransformer):
            circuit.measure_all()
        # Recursively add gates to Qiskit circuit
        else:
            append_gate(step, circuit)

    if measure_all:
        circuit.measure_all()

    return circuit
