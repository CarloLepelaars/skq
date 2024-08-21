from sklearn.pipeline import Pipeline

from skq.converters.base import QuantumCircuitConverter


class QasmCircuitConverter(QuantumCircuitConverter):
    """ 
    Converts a pipeline to an OpenQASM 3.0 circuit.
    :param pipeline: scikit-learn Pipeline
    :num_qubits: Integer representing the number of qubits in the circuit.
    :measure_all: Measure all qubits at the end of the circuit or not.
    """
    def __init__(self, pipeline: Pipeline, num_qubits: int, measure_all=False):
        super().__init__(pipeline)
        self.qasm_code = f"OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
        self.qasm_code += f"qreg q[{num_qubits}];\n"
        self.measure_all = measure_all
        if self.measure_all:
            self.qasm_code += f"creg c[{num_qubits}];\n"
        self.num_qubits = num_qubits
    
    def handle_gate(self, transformer, *args, **kwargs):
        qasm_gate = transformer.gate.to_qasm(qubits=transformer.qubits)
        self.qasm_code += f"{qasm_gate}\n"
    
    def handle_measurement(self, *args, **kwargs):
        self.qasm_code += "measure q -> c;\n"
    
    def finalize_conversion(self, *args, **kwargs):
        if self.measure_all:
            self.handle_measurement(*args, **kwargs)
    
    def get_qasm(self):
        return self.qasm_code.strip()
    
def pipeline_to_qasm(pipeline: Pipeline, num_qubits: int, measure_all=False) -> str:
    converter = QasmCircuitConverter(pipeline, num_qubits, measure_all)
    converter.convert()
    return converter.get_qasm()
    