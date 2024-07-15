import pytest
import numpy as np
from skq.gates import *

def test_base_gate():
    gate = BaseGate([[1, 0], [0, 1]])
    assert gate.is_unitary()
    assert gate.is_hermitian()
    assert isinstance(gate.trace(), complex)
    assert isinstance(gate.determinant(), complex)
    np.testing.assert_array_equal(gate.eigenvalues(), [1, 1])
    np.testing.assert_array_equal(gate.eigenvectors(), [[1, 0], [0, 1]])

def test_single_qubit_pauli_gates():
    """ Single qubit Pauli gates"""
    for GateClass in [XGate, YGate, ZGate]:
        gate = GateClass()
        assert gate.is_unitary(), f"{GateClass.__name__} should be unitary"
        assert gate.is_single_qubit_pauli(), f"{GateClass.__name__} should be a single-qubit Pauli gate"
        assert gate.is_single_qubit_clifford(), f"{GateClass.__name__} should be a single-qubit Clifford gate"

def test_single_qubit_clifford_gates():
    """ Single qubit Clifford gates"""
    for gate in [IdentityGate, XGate, YGate, ZGate, HadamardGate, SGate]:
        gate = gate()
        assert gate.is_unitary()
        assert gate.num_qubits() == 1
        assert not gate.is_multi_qubit()
        assert gate.is_single_qubit_clifford()

def test_t_gate():
    t_gate = TGate()
    assert t_gate.is_unitary()
    assert t_gate.num_qubits() == 1
    assert not t_gate.is_multi_qubit()
    assert hasattr(t_gate, "phi")

def test_rotation_gates():
    thetas = [0, np.pi / 2, np.pi, 2 * np.pi]
    for theta in thetas:
        for GateClass in [RotXGate, RotYGate, RotZGate]:
            gate = GateClass(theta)
            assert gate.is_unitary(), f"{GateClass.__name__} with theta={theta} should be unitary"
            assert gate.num_qubits() == 1, f"{GateClass.__name__} should operate on 1 qubit"
            assert not gate.is_multi_qubit(), f"{GateClass.__name__} should not be a multi-qubit gate"
            assert gate.frobenius_norm() == pytest.approx(np.sqrt(2)), f"{GateClass.__name__} Frobenius norm should be sqrt(2)"
            assert hasattr(gate, "theta")

def test_rotation_eigenvalues():
    theta = np.pi / 2
    for GateClass in [RotXGate, RotYGate, RotZGate]:
        gate = GateClass(theta)
        eigenvalues = gate.eigenvalues()
        expected_eigenvalues = np.exp([1j * theta / 2, -1j * theta / 2])
        assert np.allclose(np.sort(np.abs(eigenvalues)), np.sort(np.abs(expected_eigenvalues))), f"{GateClass.__name__} eigenvalues should match expected values"

def test_generalized_rotation_gate():
    thetas = [0, np.pi / 2, np.pi, 2 * np.pi]
    for theta_x in thetas:
        for theta_y in thetas:
            for theta_z in thetas:
                gate = GeneralizedRotationGate(theta_x, theta_y, theta_z)
                assert gate.is_unitary(), f"GeneralizedRotationGate with thetas=({theta_x}, {theta_y}, {theta_z}) should be unitary"
                assert gate.num_qubits() == 1, f"GeneralizedRotationGate should operate on 1 qubit"
                assert not gate.is_multi_qubit(), f"GeneralizedRotationGate should not be a multi-qubit gate"
                assert gate.frobenius_norm() == pytest.approx(np.sqrt(2)), "GeneralizedRotationGate Frobenius norm should be sqrt(2)"
                eigenvalues, eigenvectors = np.linalg.eig(gate)
                assert np.allclose(gate @ eigenvectors[:, 0], eigenvalues[0] * eigenvectors[:, 0]), "Eigenvector calculation is incorrect"
                assert np.allclose(gate @ eigenvectors[:, 1], eigenvalues[1] * eigenvectors[:, 1]), "Eigenvector calculation is incorrect"

def test_standard_multi_qubit_gates():
    for gate in [CXGate, CYGate, CZGate, CHGate, CSGate, CTGate, SWAPGate, ToffoliGate, FredkinGate]:
        gate = gate()
        assert gate.is_unitary()
        assert gate.num_qubits() >= 2
        assert gate.is_multi_qubit()

def test_cphase_gate():
    theta = np.pi / 2
    cphase = CPhaseGate(theta)
    assert cphase.is_unitary(), "CPhaseGate should be unitary"
    assert cphase.num_qubits() == 2, "CPhaseGate should operate on 2 qubits"
    assert cphase.is_multi_qubit(), "CPhaseGate should be a multi-qubit gate"
    eigenvalues, eigenvectors = np.linalg.eig(cphase)
    assert np.allclose(cphase @ eigenvectors[:, 0], eigenvalues[0] * eigenvectors[:, 0]), "Eigenvector calculation is incorrect"
    assert np.allclose(cphase @ eigenvectors[:, 1], eigenvalues[1] * eigenvectors[:, 1]), "Eigenvector calculation is incorrect"

def test_gate_commutation():
    theta = np.pi / 2
    rx = RotXGate(theta)
    rz = RotZGate(theta)
    assert not np.allclose(rx @ rz, rz @ rx), "Rx and Rz should not commute"

def test_inverse_gate():
    theta = np.pi / 2
    rx = RotXGate(theta)
    rx_inv = RotXGate(-theta)
    identity = rx @ rx_inv
    assert np.allclose(identity, np.eye(identity.shape[0])), "Rx and its inverse should result in the identity matrix"

def test_custom_gate_unitary():
    # S-gate
    s_matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
    s_gate = CustomGate(s_matrix)
    assert s_gate.is_unitary(), "S-gate should be unitary"
    assert s_gate.determinant() == pytest.approx(1j), "S-gate determinant should be 1j"

def test_custom_gate_non_unitary():
    # Non-unitary gate
    non_unitary_matrix = np.array([[1, 2], [3, 4]], dtype=complex)
    with pytest.raises(AssertionError, match="Custom gate must be unitary"):
        CustomGate(non_unitary_matrix)

def test_custom_gate_composition():
    hadamard_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hadamard_gate = CustomGate(hadamard_matrix)
    s_matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
    s_gate = CustomGate(s_matrix)
    composed_gate = CustomGate(hadamard_gate @ s_gate)
    assert composed_gate.is_unitary(), "Composed gate should be unitary"

def test_hermitian_gates():
    hermitian_gates = [
        XGate(), YGate(), ZGate(), 
        HadamardGate(), CXGate(), CZGate(), 
        SWAPGate(), ToffoliGate(), FredkinGate()
    ]
    for gate in hermitian_gates:
        assert gate.is_hermitian(), f"{gate.__class__.__name__} should be Hermitian"

def test_non_hermitian_gates():
    non_hermitian_gates = [
        TGate(), SGate(), CPhaseGate(np.pi / 4), 
        RotXGate(np.pi / 2), RotYGate(np.pi / 2), RotZGate(np.pi / 2), 
        GeneralizedRotationGate(np.pi / 2, np.pi / 2, np.pi / 2)
    ]
    for gate in non_hermitian_gates:
        assert not gate.is_hermitian(), f"{gate.__class__.__name__} should not be Hermitian"
