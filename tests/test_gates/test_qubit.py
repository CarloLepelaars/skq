import pytest
import qiskit
import numpy as np
import pennylane as qml

from skq.gates.qubit import *

import pytest
import numpy as np

from skq.gates.qubit import *


def test_base_gate():
    gate = QubitGate([[1, 0], [0, 1]]) # Identity gate
    assert gate.num_qubits() == 1, "Identity Gate should have 1 qubit"
    assert not gate.is_multi_qubit(), "Identity Gate is not a multi-qubit gate"
    assert gate.is_pauli(), "Identity Gate is a Pauli Gate"
    assert gate.is_clifford(), "Identity Gate is a Clifford Gate"
    assert isinstance(gate.trace(), complex), "Trace should be a complex number"
    np.testing.assert_array_equal(gate.eigenvalues(), [1, 1])
    np.testing.assert_array_equal(gate.eigenvectors(), [[1, 0], [0, 1]])

def test_single_qubit_pauli_gates():
    """ Single qubit Pauli gates"""
    for GateClass in [XGate, YGate, ZGate]:
        gate = GateClass()
        assert gate.is_unitary(), f"{GateClass.__name__} should be unitary"
        assert gate.is_pauli(), f"{GateClass.__name__} should be a single-qubit Pauli gate"
        assert gate.is_clifford(), f"{GateClass.__name__} should be a single-qubit Clifford gate"

def test_single_qubit_clifford_gates():
    """ Single qubit Clifford gates"""
    for GateClass in [IGate, XGate, YGate, ZGate, HGate, SGate]:
        gate = GateClass()
        assert gate.is_unitary(), f"{GateClass.__name__} should be unitary"
        assert gate.num_qubits() == 1, f"{GateClass.__name__} should operate on 1 qubit"
        assert not gate.is_multi_qubit(), f"{GateClass.__name__} should not be a multi-qubit gate"
        assert gate.is_clifford(), f"{GateClass.__name__} should be a single-qubit Clifford gate"

    # Algebraic equivalences
    # S = T^2 = P(pi/2)
    np.testing.assert_almost_equal(SGate(), TGate()**2)
    np.testing.assert_almost_equal(SGate(), PhaseGate(np.pi / 2))
    # H = (X + Z) / sqrt(2)
    np.testing.assert_almost_equal(HGate(), (XGate() + ZGate()) / np.sqrt(2))
    # Z = HXH = P(pi)
    np.testing.assert_almost_equal(ZGate(), HGate() @ XGate() @ HGate())
    np.testing.assert_almost_equal(ZGate(), PhaseGate(np.pi))
    # X = HZH
    np.testing.assert_almost_equal(XGate(), HGate() @ ZGate() @ HGate())

def test_t_gate():
    t_gate = TGate()
    assert isinstance(t_gate, PhaseGate), "TGate should be an instance of PhaseGate"
    assert t_gate.is_unitary(), "TGate should be unitary"
    assert t_gate.num_qubits() == 1, "TGate should operate on 1 qubit"
    assert not t_gate.is_multi_qubit(), "TGate should not be a multi-qubit gate"
    assert not t_gate.is_pauli(), "TGate should not be a Pauli gate"
    assert not t_gate.is_clifford(), "TGate should not be a Clifford gate"
    assert hasattr(t_gate, "theta"), "TGate should have a theta attribute"
    # T = P(pi/4)
    np.testing.assert_almost_equal(t_gate, PhaseGate(np.pi / 4))

def test_rotation_gates():
    thetas = [0, np.pi / 2, np.pi, 2 * np.pi]
    for theta in thetas:
        for GateClass in [RXGate, RYGate, RZGate]:
            gate = GateClass(theta)
            assert gate.is_unitary(), f"{GateClass.__name__} with theta={theta} should be unitary"
            assert gate.num_qubits() == 1, f"{GateClass.__name__} should operate on 1 qubit"
            assert not gate.is_multi_qubit(), f"{GateClass.__name__} should not be a multi-qubit gate"
            assert gate.frobenius_norm() == pytest.approx(np.sqrt(2)), f"{GateClass.__name__} Frobenius norm should be sqrt(2)"
            assert hasattr(gate, "theta")

def test_rotation_eigenvalues():
    theta = np.pi / 2
    for GateClass in [RXGate, RYGate, RZGate]:
        gate = GateClass(theta)
        eigenvalues = gate.eigenvalues()
        expected_eigenvalues = np.exp([1j * theta / 2, -1j * theta / 2])
        assert np.allclose(np.sort(np.abs(eigenvalues)), np.sort(np.abs(expected_eigenvalues))), f"{GateClass.__name__} eigenvalues should match expected values"

def test_u3gate():
    thetas = [0, np.pi / 2, np.pi, 2 * np.pi]
    for theta_x in thetas:
        for theta_y in thetas:
            for theta_z in thetas:
                gate = U3Gate(theta_x, theta_y, theta_z)
                assert gate.is_unitary(), f"U3Gate with thetas=({theta_x}, {theta_y}, {theta_z}) should be unitary"
                assert gate.num_qubits() == 1, f"U3Gate should operate on 1 qubit"
                assert not gate.is_multi_qubit(), f"U3Gate should not be a multi-qubit gate"
                assert gate.frobenius_norm() == pytest.approx(np.sqrt(2)), "U3Gate Frobenius norm should be sqrt(2)"
                eigenvalues, eigenvectors = np.linalg.eig(gate)
                assert np.allclose(gate @ eigenvectors[:, 0], eigenvalues[0] * eigenvectors[:, 0]), "Eigenvector calculation is incorrect"
                assert np.allclose(gate @ eigenvectors[:, 1], eigenvalues[1] * eigenvectors[:, 1]), "Eigenvector calculation is incorrect"

def test_standard_multi_qubit_gates():
    for GateClass in [CXGate, CYGate, CZGate, CHGate, CSGate, CTGate, SWAPGate, CCXGate, CSwapGate,
                      ToffoliGate, FredkinGate]:
        gate = GateClass()
        assert gate.is_unitary()
        assert gate.num_qubits() >= 2
        assert gate.is_multi_qubit()
        if gate.num_qubits() == 2 and not isinstance(gate, CTGate):
            assert gate.is_clifford(), f"{GateClass.__name__} should be a two-qubit Clifford gate"

def test_multi_pauli_gates():
    XX, YY, ZZ, II = XGate().kron(XGate()), YGate().kron(YGate()), ZGate().kron(ZGate()), IGate().kron(IGate())
    IX, IY, IZ = IGate().kron(XGate()), IGate().kron(YGate()), IGate().kron(ZGate())
    XI, XY, XZ = XGate().kron(IGate()), XGate().kron(YGate()), XGate().kron(ZGate())
    YX, YI, YZ = YGate().kron(XGate()), YGate().kron(IGate()), YGate().kron(ZGate())
    ZX, ZY, ZI = ZGate().kron(XGate()), ZGate().kron(YGate()), ZGate().kron(IGate())
    for gate in [XX, YY, ZZ, II, IX, IY, IZ, XI, XY, XZ, YX, YI, YZ, ZX, ZY, ZI]:
        assert gate.is_unitary(), f"{gate.__class__.__name__} should be unitary"
        assert gate.is_pauli(), f"{gate.__class__.__name__} should be a multi-qubit Pauli gate"
        assert gate.is_clifford(), f"{gate.__class__.__name__} should be a multi-qubit Clifford gate"

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
    rx = RXGate(theta)
    rz = RZGate(theta)
    assert not np.allclose(rx @ rz, rz @ rx), "Rx and Rz should not commute"

def test_inverse_gate():
    theta = np.pi / 2
    rx = RXGate(theta)
    rx_inv = RXGate(-theta)
    identity = rx @ rx_inv
    assert np.allclose(identity, np.eye(identity.shape[0])), "Rx and its inverse should result in the identity matrix"

def test_custom_gate_unitary():
    # S-gate
    s_matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
    s_gate = CustomQubitGate(s_matrix)
    assert s_gate.is_unitary(), "S-gate should be unitary"
    assert s_gate.is_clifford(), "S-gate should be a single-qubit Clifford gate"

def test_custom_gate_non_unitary():
    # Non-unitary gate
    non_unitary_matrix = np.array([[1, 2], [3, 4]], dtype=complex)
    with pytest.raises(AssertionError):
        CustomQubitGate(non_unitary_matrix)

def test_custom_gate_composition():
    hadamard_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hadamard_gate = CustomQubitGate(hadamard_matrix)
    s_matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
    s_gate = CustomQubitGate(s_matrix)
    composed_gate = CustomQubitGate(hadamard_gate @ s_gate)
    assert composed_gate.is_unitary(), "Composed gate should be unitary"

def test_hermitian_gates():
    hermitian_gates = [
        XGate(), YGate(), ZGate(), 
        HGate(), CXGate(), CZGate(), 
        SWAPGate(), CCXGate(), CSwapGate()
    ]
    for gate in hermitian_gates:
        assert gate.is_unitary(), f"{gate.__class__.__name__} should be unitary"
        assert gate.is_hermitian(), f"{gate.__class__.__name__} should be Hermitian"

def test_non_hermitian_gates():
    non_hermitian_gates = [
        TGate(), SGate(), CPhaseGate(np.pi / 4), 
        RXGate(np.pi / 2), RYGate(np.pi / 2), RZGate(np.pi / 2), 
        U3Gate(np.pi / 2, np.pi / 2, np.pi / 2)
    ]
    for gate in non_hermitian_gates:
        assert not gate.is_hermitian(), f"{gate.__class__.__name__} should not be Hermitian"

def test_sqrt():
    gate = XGate()
    # Construct Sqrt(X) gate
    sqrt_x = gate.sqrt()
    assert sqrt_x.is_unitary()
    assert not sqrt_x.is_pauli()
    assert not sqrt_x.is_clifford()
    assert isinstance(sqrt_x, QubitGate)
    np.testing.assert_array_almost_equal(sqrt_x @ sqrt_x, gate)
    expected_matrix = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
    np.testing.assert_array_almost_equal(sqrt_x, expected_matrix)

def test_kron():
    hgate = HGate()
    igate = IGate()
    h_i = hgate.kron(igate)
    np.testing.assert_array_almost_equal(h_i, 1 / np.sqrt(2) * np.array([[1, 0, 1, 0], 
                                                                         [0, 1, 0, 1], 
                                                                         [1, 0, -1, 0], 
                                                                         [0, 1, 0, -1]])
                                         )
    np.testing.assert_array_almost_equal(h_i, np.kron(hgate, igate))
    assert h_i.is_unitary()
    assert h_i.num_qubits() == 2
    assert h_i.is_multi_qubit()
    assert isinstance(h_i, CustomQubitGate)

    # Test |00> state
    state = np.array([1, 0, 0, 0])
    transformed_state = h_i @ state
    expected_state = np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])
    np.testing.assert_array_almost_equal(transformed_state, expected_state)

def test_kernel_density():
    X = XGate()
    Z = ZGate() 

    # Orthogonal gates should have a zero kernel density
    xz_kernel_density = X.kernel_density(Z)
    assert isinstance(xz_kernel_density, complex)
    assert xz_kernel_density == 0j

    # I with I should have kernel density of 2
    I = IGate() 
    ii_kernel_density = I.kernel_density(I)
    assert isinstance(ii_kernel_density, complex)
    assert ii_kernel_density == 2+0j

    # T with Z should have a kernel density of 0.2929-0.7071j
    T = TGate()
    tz_kernel_density = T.kernel_density(Z)
    assert isinstance(tz_kernel_density, complex)
    assert np.isclose(tz_kernel_density, 0.2928932188134524-0.7071067811865475j)

def test_hilbert_schmidt_inner_product():
    # Orthogonal gates should have a zero inner product
    X = XGate()
    Z = ZGate()
    xz_inner_product = X.hilbert_schmidt_inner_product(Z)
    assert isinstance(xz_inner_product, complex)
    assert xz_inner_product == 0j

    # I with I should have Hilbert-Schmidt inner product of 2
    I = IGate()
    ii_inner_product = I.hilbert_schmidt_inner_product(I)
    assert isinstance(ii_inner_product, complex)
    assert ii_inner_product == 2+0j

    # T with Z should have a Hilbert-Schmidt inner product of 0.2929+0.7071j
    T = TGate()
    tz_inner_product = T.hilbert_schmidt_inner_product(Z)
    assert isinstance(tz_inner_product, complex)
    assert tz_inner_product == 0.2928932188134524+0.7071067811865475j

def test_convert_endianness():
    # Hadamard
    gate = HGate()
    little_endian_gate = gate.convert_endianness()
    # Little endian is the same as big endian
    expected_matrix = np.array([[1, 1], 
                                [1, -1]]) / np.sqrt(2)
    np.testing.assert_array_almost_equal(little_endian_gate, expected_matrix)

    # CNOT
    gate = CXGate()
    little_endian_gate = gate.convert_endianness()
    # Permuted matrix for little endian
    expected_matrix = np.array([[1, 0, 0, 0], 
                                [0, 0, 0, 1], 
                                [0, 0, 1, 0], 
                                [0, 1, 0, 0]])
    np.testing.assert_array_almost_equal(little_endian_gate, expected_matrix)

def test_to_qiskit():
    # Hadamard
    gate = HGate()
    qiskit_gate = gate.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.library.HGate)
    np.testing.assert_array_equal(gate.convert_endianness(), qiskit_gate.to_matrix())

    # CNOT
    gate = CXGate()
    qiskit_gate = gate.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.library.CXGate)
    np.testing.assert_array_equal(gate.convert_endianness(), qiskit_gate.to_matrix())


def test_from_qiskit():
    # Hadamard
    qiskit_gate = qiskit.circuit.library.HGate()
    gate = QubitGate.from_qiskit(gate=qiskit_gate)
    assert isinstance(gate, QubitGate)
    np.testing.assert_array_equal(gate.convert_endianness(), qiskit_gate.to_matrix())
    assert gate.is_unitary()
    assert gate.num_qubits() == 1
    assert gate.is_clifford()    

    # CNOT
    qiskit_gate = qiskit.circuit.library.CXGate()
    gate = QubitGate.from_qiskit(gate=qiskit_gate)
    assert isinstance(gate, QubitGate)
    np.testing.assert_array_equal(gate.convert_endianness(), qiskit_gate.to_matrix())
    assert gate.is_unitary()
    assert gate.num_qubits() == 2
    assert gate.is_clifford()

def test_to_pennylane():
    gate = XGate()
    pennylane_gate = gate.to_pennylane(wires=0)
    assert isinstance(pennylane_gate, qml.operation.Operation)
    np.testing.assert_array_equal(pennylane_gate.matrix(), gate)

    gate = IGate()
    pennylane_gate = gate.to_pennylane(wires=0)
    assert isinstance(pennylane_gate, qml.operation.Operation)
    np.testing.assert_array_equal(pennylane_gate.matrix(), gate)

def test_from_pennylane():
    pennylane_gate = qml.PauliX(wires=[0])
    gate = QubitGate.from_pennylane(gate=pennylane_gate)
    assert isinstance(gate, QubitGate)
    np.testing.assert_array_equal(gate, pennylane_gate.matrix())
    assert gate.num_qubits() == 1
    assert gate.is_pauli()
    assert gate.is_clifford()
    