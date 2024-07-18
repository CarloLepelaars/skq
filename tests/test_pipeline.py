import numpy as np

from skq.pipeline import QuantumFeatureUnion
from skq.transformers import HTransformer, ZTransformer

def test_quantum_feature_union():
    H_transformer = HTransformer(qubits=[0])
    Z_transformer = ZTransformer(qubits=[1])
    
    qfu = QuantumFeatureUnion([('hadamard', H_transformer), ('pauli-z', Z_transformer)], n_qubits=2)
    X = np.array([[1, 0, 0, 0]], dtype=complex)
    transformed_X = qfu.transform(X)
    
    expected_combined_gate = np.kron(H_transformer.gate, Z_transformer.gate)
    assert np.allclose(transformed_X, [expected_combined_gate @ x for x in X])
