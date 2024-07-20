import pytest
import numpy as np

from skq.pipeline import QuantumLayer
from skq.transformers import HTransformer, ZTransformer, CXTransformer, TTransformer, U3Transformer, SWAPTransformer

def test_quantum_feature_union_single_qubits():
    H_transformer = HTransformer(qubits=[0])
    Z_transformer = ZTransformer(qubits=[1])
    
    qfu = QuantumLayer([('hadamard', H_transformer), ('pauli-z', Z_transformer)], n_qubits=2)
    X = np.array([[1, 0, 0, 0]], dtype=complex)
    transformed_X = qfu.transform(X)
    
    expected_combined_gate = np.kron(H_transformer.gate, Z_transformer.gate)
    assert np.allclose(transformed_X, [expected_combined_gate @ x for x in X])

def test_quantum_feature_union_single_multi_qubit():
    CX_transformer = CXTransformer(qubits=[0, 1])
    T_transformer = TTransformer(qubits=[2])
    
    qfu = QuantumLayer([('cnot', CX_transformer), ('t-gate', T_transformer)], n_qubits=3)
    # Initial state |110>
    X = np.array([[0, 0, 0, 0, 0, 1, 0, 0]], dtype=complex)
    transformed_X = qfu.transform(X)

    # Expected final state |101> with phase e^(i*pi/4) -> [0, 0, 0, 0, 0.70710678 + 0.70710678j, 0, 0, 0]
    expected_output = np.array([[0, 0, 0, 0, 0, 0.70710678 + 0.70710678j, 0, 0]], dtype=complex)
    assert np.allclose(transformed_X, expected_output)

def test_quantum_feature_union_single_multi_qubit_swap():
    SWAP_transformer = SWAPTransformer(qubits=[0, 1])
    T_transformer = TTransformer(qubits=[2])

    qfu = QuantumLayer([('swap', SWAP_transformer), ('t-gate', T_transformer)], n_qubits=3)
    
    # Initial state |100> -> [0, 0, 0, 0, 0, 0, 0, 1]
    X = np.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    transformed_X = qfu.transform(X)
    
    # Expected final state |011> with phase e^(i*pi/4) -> [0, 0, 0, 0, 0, 0, 0, 0.70710678+0.70710678j]
    expected_output = np.array([[0, 0, 0, 0, 0, 0, 0, 0.70710678 + 0.70710678j]], dtype=complex)
    assert np.allclose(transformed_X, expected_output)

def test_quantum_feature_union_multiple_single_qubits():
    H_transformer = HTransformer(qubits=[0])
    T_transformer = TTransformer(qubits=[1])
    U3_transformer = U3Transformer(theta=np.pi, phi=np.pi, lam=np.pi, qubits=[2])
    
    qfu = QuantumLayer([
        ('hadamard', H_transformer),
        ('t-gate', T_transformer),
        ('u3', U3_transformer)
    ], n_qubits=3)
    X = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
    transformed_X = qfu.transform(X)
    
    expected_combined_gate = np.kron(H_transformer.gate, np.kron(T_transformer.gate, U3_transformer.gate))
    assert np.allclose(transformed_X, [expected_combined_gate @ x for x in X])

def test_clashing_qubits():
    CX_transformer = CXTransformer(qubits=[0, 1])
    SWAP_transformer = SWAPTransformer(qubits=[1, 2])

    qfu = QuantumLayer([('cnot', CX_transformer), ('swap', SWAP_transformer)], n_qubits=3)
    
    initial_state = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
    
    # Expect ValueError due to clashing qubits
    with pytest.raises(ValueError):
        qfu.transform(initial_state)

def test_invalid_transformer():
    class InvalidTransformer:
        pass

    invalid_transformer = InvalidTransformer()
    qfu = QuantumLayer([('invalid', invalid_transformer)], n_qubits=1)
    
    initial_state = np.array([[1, 0]], dtype=complex)
    
    # Expect ValueError due to invalid transformer
    with pytest.raises(ValueError):
        qfu.transform(initial_state)
