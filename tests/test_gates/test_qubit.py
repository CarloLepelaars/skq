import pytest
import qiskit
import numpy as np
import pennylane as qml

from skq.gates.qubit import *


def test_base_gate():
    gate = QubitGate([[1, 0], [0, 1]])  # Identity gate
    assert gate.num_qubits == 1, "Identity Gate should have 1 qubit"
    assert not gate.is_multi_qubit(), "Identity Gate is not a multi-qubit gate"
    assert gate.is_pauli(), "Identity Gate is a Pauli Gate"
    assert gate.is_clifford(), "Identity Gate is a Clifford Gate"
    assert isinstance(gate.trace(), complex), "Trace should be a complex number"
    np.testing.assert_array_equal(gate.eigenvalues(), [1, 1])
    np.testing.assert_array_equal(gate.eigenvectors(), [[1, 0], [0, 1]])


def test_single_qubit_pauli_gates():
    """Single qubit Pauli gates"""
    for GateClass in [X, Y, Z]:
        gate = GateClass()
        assert gate.is_unitary(), f"{GateClass.__name__} should be unitary"
        assert gate.is_pauli(), f"{GateClass.__name__} should be a single-qubit Pauli gate"
        assert gate.is_clifford(), f"{GateClass.__name__} should be a single-qubit Clifford gate"


def test_single_qubit_clifford_gates():
    """Single qubit Clifford gates"""
    for GateClass in [I, X, Y, Z, H, S]:
        gate = GateClass()
        assert gate.is_unitary(), f"{GateClass.__name__} should be unitary"
        assert gate.num_qubits == 1, f"{GateClass.__name__} should operate on 1 qubit"
        assert not gate.is_multi_qubit(), f"{GateClass.__name__} should not be a multi-qubit gate"
        assert gate.is_clifford(), f"{GateClass.__name__} should be a single-qubit Clifford gate"

    # Algebraic equivalences
    # S = T^2 = P(pi/2)
    np.testing.assert_almost_equal(S(), T() ** 2)
    np.testing.assert_almost_equal(S(), Phase(np.pi / 2))
    # H = (X + Z) / sqrt(2)
    np.testing.assert_almost_equal(H(), (X() + Z()) / np.sqrt(2))
    # Z = HXH = P(pi)
    np.testing.assert_almost_equal(Z(), H() @ X() @ H())
    np.testing.assert_almost_equal(Z(), Phase(np.pi))
    # X = HZH
    np.testing.assert_almost_equal(X(), H() @ Z() @ H())


def test_t_gate():
    t_gate = T()
    assert isinstance(t_gate, Phase), "TGate should be an instance of PhaseGate"
    assert t_gate.is_unitary(), "TGate should be unitary"
    assert t_gate.num_qubits == 1, "TGate should operate on 1 qubit"
    assert not t_gate.is_multi_qubit(), "TGate should not be a multi-qubit gate"
    assert not t_gate.is_pauli(), "TGate should not be a Pauli gate"
    assert not t_gate.is_clifford(), "TGate should not be a Clifford gate"
    assert hasattr(t_gate, "phi"), "TGate should have a phi attribute"
    # T = P(pi/4)
    np.testing.assert_almost_equal(t_gate, Phase(np.pi / 4))


def test_rotation_gates():
    thetas = [0, np.pi / 2, np.pi, 2 * np.pi]
    for theta in thetas:
        for GateClass in [RX, RY, RZ]:
            gate = GateClass(theta)
            assert gate.is_unitary(), f"{GateClass.__name__} with theta={theta} should be unitary"
            assert gate.num_qubits == 1, f"{GateClass.__name__} should operate on 1 qubit"
            assert not gate.is_multi_qubit(), f"{GateClass.__name__} should not be a multi-qubit gate"
            assert gate.frobenius_norm() == pytest.approx(np.sqrt(2)), f"{GateClass.__name__} Frobenius norm should be sqrt(2)"
            assert hasattr(gate, "phi")


def test_rotation_eigenvalues():
    theta = np.pi / 2
    for GateClass in [RX, RY, RZ]:
        gate = GateClass(theta)
        eigenvalues = gate.eigenvalues()
        expected_eigenvalues = np.exp([1j * theta / 2, -1j * theta / 2])
        assert np.allclose(np.sort(np.abs(eigenvalues)), np.sort(np.abs(expected_eigenvalues))), f"{GateClass.__name__} eigenvalues should match expected values"


def test_u3gate():
    thetas = [0, np.pi / 2, np.pi, 2 * np.pi]
    for theta_x in thetas:
        for theta_y in thetas:
            for theta_z in thetas:
                gate = U3(theta_x, theta_y, theta_z)
                assert gate.is_unitary(), f"U3Gate with thetas=({theta_x}, {theta_y}, {theta_z}) should be unitary"
                assert gate.num_qubits == 1, "U3Gate should operate on 1 qubit"
                assert not gate.is_multi_qubit(), "U3Gate should not be a multi-qubit gate"
                assert gate.frobenius_norm() == pytest.approx(np.sqrt(2)), "U3Gate Frobenius norm should be sqrt(2)"
                eigenvalues, eigenvectors = np.linalg.eig(gate)
                assert np.allclose(gate @ eigenvectors[:, 0], eigenvalues[0] * eigenvectors[:, 0]), "Eigenvector calculation is incorrect"
                assert np.allclose(gate @ eigenvectors[:, 1], eigenvalues[1] * eigenvectors[:, 1]), "Eigenvector calculation is incorrect"


def test_standard_multi_qubit_gates():
    for gate_class in [CX, CY, CZ, CH, CS, CT, SWAP, CCX, CSwap]:
        gate = gate_class()
        assert gate.is_unitary()
        assert gate.num_qubits >= 2
        assert gate.is_multi_qubit()
        if gate.num_qubits == 2 and not isinstance(gate, CT):
            assert gate.is_clifford(), f"{gate_class.__name__} should be a two-qubit Clifford gate"

        cr_gate = CR(np.pi / 2)
        assert cr_gate.is_unitary()
        assert cr_gate.num_qubits == 2
        assert cr_gate.is_multi_qubit()
        np.testing.assert_array_equal(CR(np.pi) @ CR(-np.pi), I().kron(I()))
        sym_ecr_gate = SymmetricECR(np.pi / 2)
        assert sym_ecr_gate.is_unitary()
        assert sym_ecr_gate.num_qubits == 2
        assert sym_ecr_gate.is_multi_qubit()
        np.testing.assert_array_equal(sym_ecr_gate @ sym_ecr_gate, I().kron(I()))
        # ECR is its own inverse
        np.testing.assert_array_equal(np.linalg.inv(sym_ecr_gate), sym_ecr_gate)
        asym_ecr_gate = AsymmetricECR(np.pi / 2, np.pi / 3)
        assert asym_ecr_gate.is_unitary()
        assert asym_ecr_gate.num_qubits == 2
        assert asym_ecr_gate.is_multi_qubit()
        np.testing.assert_array_almost_equal(abs(np.linalg.inv(asym_ecr_gate)), abs(asym_ecr_gate))


def test_multi_pauli_gates():
    XX, YY, ZZ, II = X().kron(X()), Y().kron(Y()), Z().kron(Z()), I().kron(I())
    IX, IY, IZ = I().kron(X()), I().kron(Y()), I().kron(Z())
    XI, XY, XZ = X().kron(I()), X().kron(Y()), X().kron(Z())
    YX, YI, YZ = Y().kron(X()), Y().kron(I()), Y().kron(Z())
    ZX, ZY, ZI = Z().kron(X()), Z().kron(Y()), Z().kron(I())
    for gate in [XX, YY, ZZ, II, IX, IY, IZ, XI, XY, XZ, YX, YI, YZ, ZX, ZY, ZI]:
        assert gate.is_unitary(), f"{gate.__class__.__name__} should be unitary"
        assert gate.is_pauli(), f"{gate.__class__.__name__} should be a multi-qubit Pauli gate"
        assert gate.is_clifford(), f"{gate.__class__.__name__} should be a multi-qubit Clifford gate"


def test_cphase_gate():
    theta = np.pi / 2
    cphase = CPhase(theta)
    assert cphase.is_unitary(), "CPhase should be unitary"
    assert cphase.num_qubits == 2, "CPhase should operate on 2 qubits"
    assert cphase.is_multi_qubit(), "CPhase should be a multi-qubit gate"
    eigenvalues, eigenvectors = np.linalg.eig(cphase)
    assert np.allclose(cphase @ eigenvectors[:, 0], eigenvalues[0] * eigenvectors[:, 0]), "Eigenvector calculation is incorrect"
    assert np.allclose(cphase @ eigenvectors[:, 1], eigenvalues[1] * eigenvectors[:, 1]), "Eigenvector calculation is incorrect"


def test_gate_commutation():
    theta = np.pi / 2
    rx = RX(theta)
    rz = RZ(theta)
    assert not np.allclose(rx @ rz, rz @ rx), "Rx and Rz should not commute"


def test_inverse_gate():
    theta = np.pi / 2
    rx = RX(theta)
    rx_inv = RX(-theta)
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
    hermitian_gates = [X(), Y(), Z(), H(), CX(), CZ(), SWAP(), CCX(), CSwap()]
    for gate in hermitian_gates:
        assert gate.is_unitary(), f"{gate.__class__.__name__} should be unitary"
        assert gate.is_hermitian(), f"{gate.__class__.__name__} should be Hermitian"


def test_non_hermitian_gates():
    non_hermitian_gates = [T(), S(), CPhase(np.pi / 4), RX(np.pi / 2), RY(np.pi / 2), RZ(np.pi / 2), U3(np.pi / 2, np.pi / 2, np.pi / 2)]
    for gate in non_hermitian_gates:
        assert not gate.is_hermitian(), f"{gate.__class__.__name__} should not be Hermitian"


def test_sqrt():
    gate = X()
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
    hgate = H()
    igate = I()
    h_i = hgate.kron(igate)
    np.testing.assert_array_almost_equal(h_i, 1 / np.sqrt(2) * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]]))
    np.testing.assert_array_almost_equal(h_i, np.kron(hgate, igate))
    assert h_i.is_unitary()
    assert h_i.num_qubits == 2
    assert h_i.is_multi_qubit()
    assert isinstance(h_i, CustomQubitGate)

    # Test |00> state
    state = np.array([1, 0, 0, 0])
    transformed_state = h_i @ state
    expected_state = np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])
    np.testing.assert_array_almost_equal(transformed_state, expected_state)


def test_kernel_density():
    # Orthogonal gates should have a zero kernel density
    xz_kernel_density = X().kernel_density(Z())
    assert isinstance(xz_kernel_density, complex)
    assert xz_kernel_density == 0j

    # I with I should have kernel density of 2
    ii_kernel_density = I().kernel_density(I())
    assert isinstance(ii_kernel_density, complex)
    assert ii_kernel_density == 2 + 0j

    # T with Z should have a kernel density of 0.2929-0.7071j
    tz_kernel_density = T().kernel_density(Z())
    assert isinstance(tz_kernel_density, complex)
    assert np.isclose(tz_kernel_density, 0.2928932188134524 - 0.7071067811865475j)


def test_hilbert_schmidt_inner_product():
    # Orthogonal gates should have a zero inner product
    xz_inner_product = X().hilbert_schmidt_inner_product(Z())
    assert isinstance(xz_inner_product, complex)
    assert xz_inner_product == 0j

    # I with I should have Hilbert-Schmidt inner product of 2

    ii_inner_product = I().hilbert_schmidt_inner_product(I())
    assert isinstance(ii_inner_product, complex)
    assert ii_inner_product == 2 + 0j

    # T with Z should have a Hilbert-Schmidt inner product of 0.2929+0.7071j
    tz_inner_product = T().hilbert_schmidt_inner_product(Z())
    assert isinstance(tz_inner_product, complex)
    assert tz_inner_product == 0.2928932188134524 + 0.7071067811865475j


def test_convert_endianness():
    # Hadamard
    gate = H()
    little_endian_gate = gate.convert_endianness()
    # Little endian is the same as big endian
    expected_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    np.testing.assert_array_almost_equal(little_endian_gate, expected_matrix)

    # CNOT
    gate = CX()
    little_endian_gate = gate.convert_endianness()
    # Permuted matrix for little endian
    expected_matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    np.testing.assert_array_almost_equal(little_endian_gate, expected_matrix)


def test_to_qiskit():
    # Hadamard
    gate = H()
    qiskit_gate = gate.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.library.HGate)
    np.testing.assert_array_equal(gate.convert_endianness(), qiskit_gate.to_matrix())

    # CNOT
    gate = CX()
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
    assert gate.num_qubits == 1
    assert gate.is_clifford()

    # CNOT
    qiskit_gate = qiskit.circuit.library.CXGate()
    gate = QubitGate.from_qiskit(gate=qiskit_gate)
    assert isinstance(gate, QubitGate)
    np.testing.assert_array_equal(gate.convert_endianness(), qiskit_gate.to_matrix())
    assert gate.is_unitary()
    assert gate.num_qubits == 2
    assert gate.is_clifford()


def test_to_pennylane():
    gate = X()
    pennylane_gate = gate.to_pennylane(wires=0)
    assert isinstance(pennylane_gate, qml.operation.Operation)
    np.testing.assert_array_equal(pennylane_gate.matrix(), gate)

    gate = I()
    pennylane_gate = gate.to_pennylane(wires=0)
    assert isinstance(pennylane_gate, qml.operation.Operation)
    np.testing.assert_array_equal(pennylane_gate.matrix(), gate)


def test_from_pennylane():
    pennylane_gate = qml.PauliX(wires=[0])
    gate = QubitGate.from_pennylane(gate=pennylane_gate)
    assert isinstance(gate, QubitGate)
    np.testing.assert_array_equal(gate, pennylane_gate.matrix())
    assert gate.num_qubits == 1
    assert gate.is_pauli()
    assert gate.is_clifford()


def test_to_qasm_single():
    # Simple H gate example
    gate = H()
    qasm = gate.to_qasm([2])
    expected_qasm = "h q[2];"
    assert qasm == expected_qasm

    # U3
    gate = U3(np.pi / 2, np.pi / 2, np.pi / 2)
    qasm = gate.to_qasm([0])
    expected_qasm = """rx(1.5707963267948966) q[0];\nry(1.5707963267948966) q[0];\nrz(1.5707963267948966) q[0];"""
    assert qasm == expected_qasm


def test_to_qasm_multi():
    # CCX
    gate = CCX()
    qasm = gate.to_qasm([0, 1, 2])
    expected_qasm = "ccx q[0], q[1], q[2];"
    assert qasm == expected_qasm

    # CCZ custom spec
    gate = CCZ()
    qasm = gate.to_qasm([0, 1, 2])
    # H, CCX, H
    expected_qasm = """h q[2];\nccx q[0], q[1], q[2];\nh q[2];"""
    assert qasm == expected_qasm


def test_measurement_gate():
    """Test measurement gate functionality"""
    measure = Measure()

    # Basic properties
    assert measure.num_qubits == 1
    assert measure.is_unitary()
    assert not measure.is_multi_qubit()

    # Single qubit states
    state_0 = np.array([1, 0], dtype=complex)  # |0⟩
    state_1 = np.array([0, 1], dtype=complex)  # |1⟩
    state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩

    np.testing.assert_array_almost_equal(measure(state_0), [1, 0])
    np.testing.assert_array_almost_equal(measure(state_1), [0, 1])
    np.testing.assert_array_almost_equal(measure(state_plus), [0.5, 0.5])

    # Two-qubit states
    bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
    np.testing.assert_array_almost_equal(measure(bell_state), [0.5, 0, 0, 0.5])


def test_measurement_conversions():
    """Test measurement gate conversions to different frameworks"""
    measure = Measure()

    # QASM conversion
    qasm_str = measure.to_qasm([0])
    assert qasm_str == "measure q[0] -> c[0];"

    # Qiskit conversion
    qiskit_gate = measure.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.library.Measure)

    # PennyLane conversion
    pennylane_gate = measure.to_pennylane(wires=0)
    assert isinstance(pennylane_gate, qml.measurements.mid_measure.MeasurementValue)
