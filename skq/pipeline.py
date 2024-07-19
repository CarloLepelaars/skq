import numpy as np
from sklearn.pipeline import _name_estimators, FeatureUnion

from skq.gates import Gate, IGate
from skq.utils import _check_quantum_state_array
from skq.transformers import SingleQubitTransformer, MultiQubitTransformer


class QuantumFeatureUnion(FeatureUnion):
    def __init__(self, transformer_list, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        super().__init__(transformer_list, **kwargs)

    def fit(self, X, y=None):
        _check_quantum_state_array(X)
        self._validate_transformers()
        return self

    def transform(self, X) -> np.array:
        _check_quantum_state_array(X)
        self._validate_transformers()
        combined_gate = np.eye(2**self.n_qubits, dtype=complex)
        for _, transformer in self.transformer_list:
            full_gate = self._construct_full_gate(transformer)
            combined_gate = np.dot(combined_gate, full_gate)
        return np.array([combined_gate @ x for x in X])
    
    def _validate_transformers(self):
        used_qubits = set()
        for _, transformer in self.transformer_list:
            if not hasattr(transformer, 'qubits'):
                raise ValueError(f"Transformers in QuantumFeatureUnion must be either type `SingleQubitTransformer` or `MultiQubitTransformer`. Got '{type(transformer)}'")
            # Check for transformers trying to use the same qubit
            for qubit in transformer.qubits:
                if qubit in used_qubits:
                    raise ValueError(f"Qubit {qubit} is used by multiple transformers.")
                used_qubits.add(qubit)


    def _construct_full_gate(self, transformer):
        if isinstance(transformer, SingleQubitTransformer):
            return self._expand_single_qubit_gate(transformer.gate, transformer.qubits[0])
        elif isinstance(transformer, MultiQubitTransformer):
            return self._expand_multi_qubit_gate(transformer.gate, transformer.qubits)
        else:
            raise ValueError(f"Transformers in QuantumFeatureUnion must be either type `SingleQubitTransformer` or `MultiQubitTransformer`. Got '{type(transformer)}'")

    def _expand_single_qubit_gate(self, gate: Gate, qubit: int) -> np.array:
        """
        Expands a single qubit gate to a full gate.
        :param gate: 2x2 numpy array representing the gate.
        :param qubit: Index of the qubit the gate is applied to.
        :return: 2**n_qubits x 2**n_qubits numpy array representing the full gate.
        """
        full_gate = np.eye(1, dtype=complex)
        for i in range(self.n_qubits):
            if i == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, IGate())
        return full_gate

    def _expand_multi_qubit_gate(self, gate: Gate, qubits: list[int]) -> np.array:
        """
        Expands a multi qubit gate to a full gate.
        :param gate: numpy array representing the gate.
        :param qubits: List of qubits the gate is applied to.
        :return: 2**n_qubits x 2**n_qubits numpy array representing the full gate.
        """
        full_gate = np.eye(2**self.n_qubits, dtype=complex)
        num_controls = len(qubits) - 1
        for i in range(2**self.n_qubits):
            binary = f"{i:0{self.n_qubits}b}"
            if all(binary[q] == '1' for q in qubits[:-1]):
                target = qubits[-1]
                for k in range(2**num_controls):
                    control_state = f"{k:0{num_controls}b}"
                    control_index = int(''.join(control_state), 2)
                    full_gate[i ^ (control_index << target), i ^ (control_index << target)] = gate[control_index, control_index]
                    full_gate[i ^ (control_index << target), i] = gate[control_index, 0]
                    full_gate[i, i ^ (control_index << target)] = gate[0, control_index]
        return full_gate

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
