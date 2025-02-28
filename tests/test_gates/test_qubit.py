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


def test_deutsch_jozsa_oracle():
    """Test Deutsch-Jozsa Oracle functionality"""
    # 2-qubit constant function f(x) = 0
    constant_zero = lambda x: 0
    oracle_constant_zero = DeutschJozsaOracle(constant_zero, n_bits=2)
    assert oracle_constant_zero.is_unitary()
    assert oracle_constant_zero.num_qubits == 2
    assert oracle_constant_zero.is_multi_qubit()
    np.testing.assert_array_equal(oracle_constant_zero, np.eye(4))

    # 2-qubit constant function f(x) = 1
    constant_one = lambda x: 1
    oracle_constant_one = DeutschJozsaOracle(constant_one, n_bits=2)
    assert oracle_constant_one.is_unitary()
    expected_matrix = -np.eye(4)
    np.testing.assert_array_equal(oracle_constant_one, expected_matrix)

    # 2-qubit balanced function (half 0s, half 1s)
    balanced = lambda x: 1 if x < 2 else 0
    oracle_balanced = DeutschJozsaOracle(balanced, n_bits=2)
    assert oracle_balanced.is_unitary()
    expected_matrix = np.diag([-1, -1, 1, 1])
    np.testing.assert_array_equal(oracle_balanced, expected_matrix)

    # Invalid function (not constant or balanced)
    invalid_func = lambda x: 1 if x == 0 else 0
    with pytest.raises(ValueError):
        DeutschJozsaOracle(invalid_func, n_bits=2)

    # Invalid output values
    invalid_output = lambda x: 2
    with pytest.raises(ValueError):
        DeutschJozsaOracle(invalid_output, n_bits=2)


def test_qft_gate():
    """Test Quantum Fourier Transform gate functionality"""
    # Test 1-qubit QFT (should be equivalent to Hadamard)
    qft_1 = QFT(n_qubits=1)
    assert qft_1.is_unitary()
    assert qft_1.num_qubits == 1
    assert not qft_1.is_multi_qubit()
    np.testing.assert_array_almost_equal(qft_1, H())

    # Test 2-qubit QFT
    qft_2 = QFT(n_qubits=2)
    assert qft_2.is_unitary()
    assert qft_2.num_qubits == 2
    assert qft_2.is_multi_qubit()

    # Expected matrix for 2-qubit QFT
    omega = np.exp(2j * np.pi / 4)  # 4th root of unity
    expected_matrix = np.array([[1, 1, 1, 1], [1, omega, omega**2, omega**3], [1, omega**2, omega**4, omega**6], [1, omega**3, omega**6, omega**9]]) / 2  # Normalize by 1/sqrt(4)
    np.testing.assert_array_almost_equal(qft_2, expected_matrix)

    # Test inverse 2-qubit QFT
    inv_qft_2 = QFT(n_qubits=2, inverse=True)
    assert inv_qft_2.is_unitary()
    assert inv_qft_2.num_qubits == 2
    assert inv_qft_2.is_multi_qubit()

    # Test 3-qubit QFT
    qft_3 = QFT(n_qubits=3)
    assert qft_3.is_unitary()
    assert qft_3.num_qubits == 3
    assert qft_3.is_multi_qubit()

    # Verify dimensions
    assert qft_3.shape == (8, 8)

    # Expected matrix for 3-qubit QFT
    omega_8 = np.exp(2j * np.pi / 8)  # 8th root of unity
    expected_matrix_3 = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        for j in range(8):
            expected_matrix_3[i, j] = omega_8 ** (i * j) / np.sqrt(8)
    np.testing.assert_array_almost_equal(qft_3, expected_matrix_3)

    # Since omega_8^8 = 1, we need to take modulo 8 for the exponents
    normalized_explicit_matrix = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  # |000⟩
            [1, omega_8, omega_8**2, omega_8**3, omega_8**4, omega_8**5, omega_8**6, omega_8**7],  # |001⟩
            [1, omega_8**2, omega_8**4, omega_8**6, omega_8**0, omega_8**2, omega_8**4, omega_8**6],  # |010⟩
            [1, omega_8**3, omega_8**6, omega_8**1, omega_8**4, omega_8**7, omega_8**2, omega_8**5],  # |011⟩
            [1, omega_8**4, omega_8**0, omega_8**4, omega_8**0, omega_8**4, omega_8**0, omega_8**4],  # |100⟩
            [1, omega_8**5, omega_8**2, omega_8**7, omega_8**4, omega_8**1, omega_8**6, omega_8**3],  # |101⟩
            [1, omega_8**6, omega_8**4, omega_8**2, omega_8**0, omega_8**6, omega_8**4, omega_8**2],  # |110⟩
            [1, omega_8**7, omega_8**6, omega_8**5, omega_8**4, omega_8**3, omega_8**2, omega_8**1],  # |111⟩
        ]
    ) / np.sqrt(8)

    # Verify the explicit matrix matches our QFT implementation
    np.testing.assert_array_almost_equal(qft_3, normalized_explicit_matrix)

    # Test inverse 3-qubit QFT
    inv_qft_3 = QFT(n_qubits=3, inverse=True)
    assert inv_qft_3.is_unitary()
    assert inv_qft_3.num_qubits == 3
    assert inv_qft_3.is_multi_qubit()

    # Verify specific matrix elements for 3-qubit QFT
    # First row should be all 1/sqrt(8)
    np.testing.assert_array_almost_equal(qft_3[0, :], np.ones(8) / np.sqrt(8))
    # First column should be all 1/sqrt(8)
    np.testing.assert_array_almost_equal(qft_3[:, 0], np.ones(8) / np.sqrt(8))
    # Check specific phase relationships
    np.testing.assert_almost_equal(qft_3[1, 1], omega_8 / np.sqrt(8))
    np.testing.assert_almost_equal(qft_3[1, 2], omega_8**2 / np.sqrt(8))
    np.testing.assert_almost_equal(qft_3[2, 1], omega_8**2 / np.sqrt(8))


def test_qft_conversions():
    """Test QFT gate conversions to different frameworks"""
    # Test Qiskit conversion
    qft_2 = QFT(n_qubits=2)
    qiskit_circuit = qft_2.to_qiskit()
    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert qiskit_circuit.name == "QFT"
    assert qiskit_circuit.num_qubits == 2

    # Test inverse QFT Qiskit conversion
    inv_qft_2 = QFT(n_qubits=2, inverse=True)
    inv_qiskit_circuit = inv_qft_2.to_qiskit()
    assert isinstance(inv_qiskit_circuit, qiskit.QuantumCircuit)
    assert inv_qiskit_circuit.name == "QFT†"
    assert inv_qiskit_circuit.num_qubits == 2

    # Test QASM conversion
    qasm_str = qft_2.to_qasm([0, 1])
    assert "// QFT circuit (big-endian convention)" in qasm_str
    assert "h q[0];" in qasm_str
    assert "h q[1];" in qasm_str
    assert "cu1" in qasm_str  # Should contain controlled phase gates
    assert "swap" in qasm_str  # Should contain swap gates
    assert "// End of QFT circuit" in qasm_str

    # Test inverse QFT QASM conversion
    inv_qasm_str = inv_qft_2.to_qasm([0, 1])
    assert "// QFT† circuit (big-endian convention)" in inv_qasm_str
    assert "h q[0];" in inv_qasm_str
    assert "h q[1];" in inv_qasm_str
    assert "cu1" in inv_qasm_str  # Should contain controlled phase gates
    assert "swap" in inv_qasm_str  # Should contain swap gates
    assert "// End of QFT† circuit" in inv_qasm_str


def test_qft_state_transformation():
    """Test QFT transformation of quantum states"""
    # Test transformation of |0⟩ state (should remain |0⟩)
    qft_1 = QFT(n_qubits=1)
    state_0 = np.array([1, 0])
    transformed_state = qft_1 @ state_0
    expected_state = np.array([1, 1]) / np.sqrt(2)  # |+⟩ state
    np.testing.assert_array_almost_equal(transformed_state, expected_state)

    # Test transformation of |00⟩ state with 2-qubit QFT
    qft_2 = QFT(n_qubits=2)
    state_00 = np.array([1, 0, 0, 0])
    transformed_state = qft_2 @ state_00
    # Should be equal superposition of all states
    expected_state = np.ones(4) / 2
    np.testing.assert_array_almost_equal(transformed_state, expected_state)

    # Test transformation of |01⟩ state with 2-qubit QFT
    state_01 = np.array([0, 1, 0, 0])
    transformed_state = qft_2 @ state_01
    omega = np.exp(2j * np.pi / 4)
    expected_state = np.array([1, omega, omega**2, omega**3]) / 2
    np.testing.assert_array_almost_equal(transformed_state, expected_state)

    # Test inverse QFT transformation
    inv_qft_2 = QFT(n_qubits=2, inverse=True)
    # Transform the already transformed state back
    original_state = inv_qft_2 @ transformed_state
    np.testing.assert_array_almost_equal(original_state, state_01)

    # Test 3-qubit QFT state transformations
    qft_3 = QFT(n_qubits=3)

    # Test |000⟩ state transformation
    state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    transformed_000 = qft_3 @ state_000
    # Should be equal superposition of all states
    expected_state_000 = np.ones(8) / np.sqrt(8)
    np.testing.assert_array_almost_equal(transformed_000, expected_state_000)

    # Test |001⟩ state transformation
    state_001 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    transformed_001 = qft_3 @ state_001
    omega_8 = np.exp(2j * np.pi / 8)
    expected_state_001 = np.array([1, omega_8, omega_8**2, omega_8**3, omega_8**4, omega_8**5, omega_8**6, omega_8**7]) / np.sqrt(8)
    np.testing.assert_array_almost_equal(transformed_001, expected_state_001)

    # Test |010⟩ state transformation
    state_010 = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    transformed_010 = qft_3 @ state_010
    expected_state_010 = np.array([1, omega_8**2, omega_8**4, omega_8**6, omega_8**8, omega_8**10, omega_8**12, omega_8**14]) / np.sqrt(8)
    np.testing.assert_array_almost_equal(transformed_010, expected_state_010)

    # Test |100⟩ state transformation
    state_100 = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    transformed_100 = qft_3 @ state_100
    expected_state_100 = np.array([1, omega_8**4, omega_8**8, omega_8**12, omega_8**16, omega_8**20, omega_8**24, omega_8**28]) / np.sqrt(8)
    np.testing.assert_array_almost_equal(transformed_100, expected_state_100)

    # Test inverse QFT for 3-qubit states
    inv_qft_3 = QFT(n_qubits=3, inverse=True)
    # Transform back to original state
    original_state_001 = inv_qft_3 @ transformed_001
    np.testing.assert_array_almost_equal(original_state_001, state_001)

    # Test a superposition state
    superposition = (state_000 + state_001 + state_010) / np.sqrt(3)
    transformed_superposition = qft_3 @ superposition
    original_superposition = inv_qft_3 @ transformed_superposition
    np.testing.assert_array_almost_equal(original_superposition, superposition)


def test_controlled_unitary_gate():
    """Test the Controlled-Unitary (CU) gate functionality"""
    # Test with X gate (should be equivalent to CX)
    x_gate = X()
    cx_gate = CU(x_gate)
    assert cx_gate.is_unitary(), "CU(X) should be unitary"
    assert cx_gate.num_qubits == 2, "CU(X) should operate on 2 qubits"
    assert cx_gate.is_multi_qubit(), "CU(X) should be a multi-qubit gate"
    assert cx_gate.n_target_qubits == 1, "CU(X) should have 1 target qubit"
    assert np.allclose(cx_gate, CX()), "CU(X) should be equivalent to CX"

    # Test with Y gate (should be equivalent to CY)
    y_gate = Y()
    cy_gate = CU(y_gate)
    assert cy_gate.is_unitary(), "CU(Y) should be unitary"
    assert cy_gate.num_qubits == 2, "CU(Y) should operate on 2 qubits"
    assert cy_gate.n_target_qubits == 1, "CU(Y) should have 1 target qubit"
    assert np.allclose(cy_gate, CY()), "CU(Y) should be equivalent to CY"

    # Test with Z gate (should be equivalent to CZ)
    z_gate = Z()
    cz_gate = CU(z_gate)
    assert cz_gate.is_unitary(), "CU(Z) should be unitary"
    assert cz_gate.num_qubits == 2, "CU(Z) should operate on 2 qubits"
    assert cz_gate.n_target_qubits == 1, "CU(Z) should have 1 target qubit"
    assert np.allclose(cz_gate, CZ()), "CU(Z) should be equivalent to CZ"

    # Test with H gate (should be equivalent to CH)
    h_gate = H()
    ch_gate = CU(h_gate)
    assert ch_gate.is_unitary(), "CU(H) should be unitary"
    assert ch_gate.num_qubits == 2, "CU(H) should operate on 2 qubits"
    assert ch_gate.n_target_qubits == 1, "CU(H) should have 1 target qubit"
    assert np.allclose(ch_gate, CH()), "CU(H) should be equivalent to CH"

    rx_gate = RX(np.pi / 4)
    crx_gate = CU(rx_gate)
    assert crx_gate.is_unitary(), "CU(RX) should be unitary"
    assert crx_gate.num_qubits == 2, "CU(RX) should operate on 2 qubits"
    assert crx_gate.n_target_qubits == 1, "CU(RX) should have 1 target qubit"

    # Verify matrix structure
    expected_matrix = np.eye(4, dtype=complex)
    expected_matrix[2:, 2:] = rx_gate
    assert np.allclose(crx_gate, expected_matrix), "CU(RX) matrix structure is incorrect"

    # Test with Phase gate
    phase_gate = Phase(np.pi / 4)
    cphase_gate = CU(phase_gate)
    assert cphase_gate.is_unitary(), "CU(Phase) should be unitary"
    assert cphase_gate.num_qubits == 2, "CU(Phase) should operate on 2 qubits"
    assert cphase_gate.n_target_qubits == 1, "CU(Phase) should have 1 target qubit"

    # Verify matrix structure
    expected_matrix = np.eye(4, dtype=complex)
    expected_matrix[2:, 2:] = phase_gate
    assert np.allclose(cphase_gate, expected_matrix), "CU(Phase) matrix structure is incorrect"

    # Test with SWAP gate (should create a controlled-SWAP, or Fredkin gate)
    swap_gate = SWAP()
    cswap_gate = CU(swap_gate)
    assert cswap_gate.is_unitary(), "CU(SWAP) should be unitary"
    assert cswap_gate.num_qubits == 3, "CU(SWAP) should operate on 3 qubits"
    assert cswap_gate.n_target_qubits == 2, "CU(SWAP) should have 2 target qubits"
    assert np.allclose(cswap_gate, CSwap()), "CU(SWAP) should be equivalent to CSwap"

    # Test CU(X) on |00⟩ state - should remain |00⟩
    state_00 = np.array([1, 0, 0, 0])
    assert np.allclose(cx_gate @ state_00, np.array([1, 0, 0, 0])), "CU(X) should leave |00⟩ unchanged"

    # Test CU(X) on |10⟩ state - should become |11⟩
    state_10 = np.array([0, 0, 1, 0])
    assert np.allclose(cx_gate @ state_10, np.array([0, 0, 0, 1])), "CU(X) should transform |10⟩ to |11⟩"

    # Test CU(X) on |01⟩ state - should remain |01⟩
    state_01 = np.array([0, 1, 0, 0])
    assert np.allclose(cx_gate @ state_01, np.array([0, 1, 0, 0])), "CU(X) should leave |01⟩ unchanged"

    # Test CU(X) on |11⟩ state - should become |10⟩
    state_11 = np.array([0, 0, 0, 1])
    assert np.allclose(cx_gate @ state_11, np.array([0, 0, 1, 0])), "CU(X) should transform |11⟩ to |10⟩"

    # |100⟩ should become |100⟩ (no swap when control is 1, targets are 00)
    state_100 = np.zeros(8)
    state_100[4] = 1
    assert np.allclose(cswap_gate @ state_100, state_100), "CU(SWAP) should leave |100⟩ unchanged"

    # |101⟩ should become |110⟩ (swap when control is 1)
    state_101 = np.zeros(8)
    state_101[5] = 1
    expected_101 = np.zeros(8)
    expected_101[6] = 1
    assert np.allclose(cswap_gate @ state_101, expected_101), "CU(SWAP) should transform |101⟩ to |110⟩"

    # |110⟩ should become |101⟩ (swap when control is 1)
    state_110 = np.zeros(8)
    state_110[6] = 1
    expected_110 = np.zeros(8)
    expected_110[5] = 1
    assert np.allclose(cswap_gate @ state_110, expected_110), "CU(SWAP) should transform |110⟩ to |101⟩"

    qiskit_gate = cx_gate.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.Instruction), "CU(X) should convert to a Qiskit Instruction"

    pennylane_gate = cx_gate.to_pennylane(wires=[0, 1])
    assert isinstance(pennylane_gate, qml.operation.Operation), "CU(X) should convert to a PennyLane Operation"

    # Custom unitary
    custom_matrix = np.array([[np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)], [-1j * np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    custom_gate = CustomQubitGate(custom_matrix)
    cu_custom = CU(custom_gate)
    assert cu_custom.is_unitary(), "CU with custom unitary should be unitary"
    assert cu_custom.num_qubits == 2, "CU with custom unitary should operate on 2 qubits"
    assert cu_custom.n_target_qubits == 1, "CU with custom unitary should have 1 target qubit"
    expected_matrix = np.eye(4, dtype=complex)
    expected_matrix[2:, 2:] = custom_matrix
    assert np.allclose(cu_custom, expected_matrix), "CU with custom unitary matrix structure is incorrect"
    qiskit_gate = cu_custom.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.Instruction), "CU should convert to a Qiskit Instruction"

    pennylane_gate = cu_custom.to_pennylane(wires=[0, 1])
    assert isinstance(pennylane_gate, qml.operation.Operation), "CU should convert to a PennyLane Operation"


def test_mcu():
    """Test the Multi-Controlled Unitary (MCU) gate functionality"""
    x_gate = X()
    ccx_gate = MCU(x_gate, num_ctrl_qubits=2)
    assert ccx_gate.is_unitary(), "MCU(X, 2) should be unitary"
    assert ccx_gate.num_qubits == 3, "MCU(X, 2) should operate on 3 qubits"
    assert ccx_gate.num_ctrl_qubits == 2, "MCU(X, 2) should have 2 control qubits"
    assert ccx_gate.n_target_qubits == 1, "MCU(X, 2) should have 1 target qubit"
    assert np.allclose(ccx_gate, CCX()), "MCU(X, 2) should be equivalent to CCX"

    cccx_gate = MCU(x_gate, num_ctrl_qubits=3)
    assert cccx_gate.is_unitary(), "MCU(X, 3) should be unitary"
    assert cccx_gate.num_qubits == 4, "MCU(X, 3) should operate on 4 qubits"
    assert cccx_gate.num_ctrl_qubits == 3, "MCU(X, 3) should have 3 control qubits"
    assert cccx_gate.n_target_qubits == 1, "MCU(X, 3) should have 1 target qubit"

    h_gate = H()
    cch_gate = MCU(h_gate, num_ctrl_qubits=2)
    assert cch_gate.is_unitary(), "MCU(H, 2) should be unitary"
    assert cch_gate.num_qubits == 3, "MCU(H, 2) should operate on 3 qubits"
    assert cch_gate.num_ctrl_qubits == 2, "MCU(H, 2) should have 2 control qubits"
    assert cch_gate.n_target_qubits == 1, "MCU(H, 2) should have 1 target qubit"

    # |000⟩ should remain |000⟩
    state_000 = np.zeros(8)
    state_000[0] = 1
    assert np.allclose(ccx_gate @ state_000, state_000), "MCU(X, 2) should leave |000⟩ unchanged"

    # |001⟩ should remain |001⟩
    state_001 = np.zeros(8)
    state_001[1] = 1
    assert np.allclose(ccx_gate @ state_001, state_001), "MCU(X, 2) should leave |001⟩ unchanged"

    # |010⟩ should remain |010⟩
    state_010 = np.zeros(8)
    state_010[2] = 1
    assert np.allclose(ccx_gate @ state_010, state_010), "MCU(X, 2) should leave |010⟩ unchanged"

    # |011⟩ should remain |011⟩
    state_011 = np.zeros(8)
    state_011[3] = 1
    assert np.allclose(ccx_gate @ state_011, state_011), "MCU(X, 2) should leave |011⟩ unchanged"

    # |100⟩ should remain |100⟩
    state_100 = np.zeros(8)
    state_100[4] = 1
    assert np.allclose(ccx_gate @ state_100, state_100), "MCU(X, 2) should leave |100⟩ unchanged"

    # |101⟩ should remain |101⟩
    state_101 = np.zeros(8)
    state_101[5] = 1
    assert np.allclose(ccx_gate @ state_101, state_101), "MCU(X, 2) should leave |101⟩ unchanged"

    # |110⟩ should become |111⟩ (X applied when both controls are 1)
    state_110 = np.zeros(8)
    state_110[6] = 1
    expected_110 = np.zeros(8)
    expected_110[7] = 1
    assert np.allclose(ccx_gate @ state_110, expected_110), "MCU(X, 2) should transform |110⟩ to |111⟩"

    # |111⟩ should become |110⟩ (X applied when both controls are 1)
    state_111 = np.zeros(8)
    state_111[7] = 1
    expected_111 = np.zeros(8)
    expected_111[6] = 1
    assert np.allclose(ccx_gate @ state_111, expected_111), "MCU(X, 2) should transform |111⟩ to |110⟩"

    theta = np.pi / 4
    rx_gate = RX(theta)
    ccrx_gate = MCU(rx_gate, num_ctrl_qubits=2)
    assert ccrx_gate.is_unitary(), "MCU(RX, 2) should be unitary"
    assert ccrx_gate.num_qubits == 3, "MCU(RX, 2) should operate on 3 qubits"

    custom_matrix = np.array([[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]])
    custom_gate = CustomQubitGate(custom_matrix)
    cc_custom = MCU(custom_gate, num_ctrl_qubits=2)
    assert cc_custom.is_unitary(), "MCU with custom unitary should be unitary"
    assert cc_custom.num_qubits == 3, "MCU with custom unitary should operate on 3 qubits"
    assert cc_custom.num_ctrl_qubits == 2, "MCU with custom unitary should have 2 control qubits"
    assert cc_custom.n_target_qubits == 1, "MCU with custom unitary should have 1 target qubit"

    # Test framework conversions
    qiskit_gate = ccx_gate.to_qiskit()
    assert qiskit_gate is not None, "MCU(X, 2) should convert to a Qiskit gate"

    pennylane_gate = ccx_gate.to_pennylane(wires=[0, 1, 2])
    assert pennylane_gate is not None, "MCU(X, 2) should convert to a PennyLane gate"
