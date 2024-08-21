import qiskit
from sklearn.pipeline import Pipeline

from skq.converters.base import QuantumCircuitConverter


class QiskitCircuitConverter(QuantumCircuitConverter):
    def __init__(self, pipeline: Pipeline, num_qubits: int, measure_all=False):
        super().__init__(pipeline)
        self.circuit = qiskit.QuantumCircuit(num_qubits)
        self.measure_all = measure_all
    
    def handle_gate(self, transformer, *args, **kwargs):
        qiskit_gate = transformer.gate.to_qiskit()
        self.circuit.append(qiskit_gate, transformer.qubits)
    
    def handle_measurement(self, *args, **kwargs):
        self.circuit.measure_all()

    def finalize_conversion(self, *args, **kwargs):
        if self.measure_all:
            self.circuit.measure_all()

    def get_circuit(self):
        return self.circuit

def pipeline_to_qiskit_circuit(pipeline: Pipeline, num_qubits: int, measure_all=False) -> qiskit.QuantumCircuit:
    converter = QiskitCircuitConverter(pipeline, num_qubits, measure_all)
    converter.convert()
    return converter.get_circuit()
