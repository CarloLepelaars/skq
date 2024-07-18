import numpy as np
from skq.transformers import ITransformer
from skq.gates import IGate
from sklearn.pipeline import _name_estimators, Pipeline, FeatureUnion

class QuantumFeatureUnion(FeatureUnion):
    def __init__(self, transformer_list, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        super().__init__(transformer_list, **kwargs)

    def transform(self, X):
        combined_gate = np.eye(2**self.n_qubits, dtype=complex)
        
        for _, transformer in self.transformer_list:
            # Construct the full gate for this transformer
            full_gate = np.eye(1, dtype=complex)
            for i in range(self.n_qubits):
                if i in transformer.qubits:
                    full_gate = np.kron(full_gate, transformer.gate)
                else:
                    full_gate = np.kron(full_gate, IGate())
            
            # Combine with the overall gate
            combined_gate = np.dot(combined_gate, full_gate)
        
        # Apply the combined gate to the input state vectors
        transformed_X = np.array([combined_gate @ x for x in X])
        return transformed_X


def make_quantum_union(*transformers, n_qubits: int , n_jobs=None, verbose=False) -> QuantumFeatureUnion:
    """
    Convenience function for creating a QuantumFeatureUnion.
    :param transformers: List of (name, transform) tuples (implementing fit/transform) that are concatenated.
    :param n_qubits: Number of qubits in the quantum circuit.
    :param n_jobs: Number of jobs to run in parallel.
    :param verbose: If True, the time elapsed while fitting each transformer will be printed.
    :return: QuantumFeatureUnion object
    """
    return QuantumFeatureUnion(_name_estimators(transformers), n_qubits=n_qubits, n_jobs=n_jobs, verbose=verbose)
