import inspect
import pytest
import numpy as np

from src.gates.qubit import *


def test_base_attrs():
    import src.gates.qubit as all_qubit_gates
    import src.gates.qutrit as all_qutrit_gates
    import src.gates.ququart as all_ququart_gates
    import src.gates.qupent as all_qupent_gates

    for module in [all_qubit_gates, all_qutrit_gates, all_ququart_gates, all_qupent_gates]:
        all_objs = [obj for _, obj in inspect.getmembers(module)]
        for obj in all_objs:
            if inspect.isclass(obj) and obj not in [BaseGate, CustomQubitGate]:
                assert hasattr(obj, "to_qiskit"), f"{obj} does not have to_qiskit method."
                assert hasattr(obj, "to_pennylane"), f"{obj} does not have to_pennylane method."
                assert hasattr(obj, "from_qiskit"), f"{obj} does not have from_qiskit method."
                assert hasattr(obj, "from_pennylane"), f"{obj} does not have from_pennylane method."
                assert hasattr(obj, "to_pyquil"), f"{obj} does not have to_pyquil method."
                assert hasattr(obj, "from_pyquil"), f"{obj} does not have from_pyquil method."


def test_base_gate():
    # Z Gate
    gate = BaseGate([[1, 0], [0, -1]])  # Identity gate
    assert gate.dtype == complex, "Gate should be complex"
    assert gate.is_unitary(), "Gate should be unitary"
    assert gate.is_hermitian(), "Z Gate should be Hermitian"
    assert gate.num_levels() == 2, "Z should have 2 levels"
    assert gate.is_2d(), "Gate should be 2D"
    assert gate.is_at_least_nxn(1), "Gate should be at least 1x1"
    assert gate.is_at_least_nxn(2), "Gate should be at least 2x2"
    assert gate.is_power_of_n_shape(2), "Gate should be a power of 2"
    assert not gate.is_identity(), "Z Gate should not be the identity"
    assert gate.is_equal(gate), "Gate should be equal to itself"
    assert gate.is_equal(ZGate()), "Gate should be equal to ZGate"
    assert not gate.is_equal(XGate()), "Gate should not be equal to XGate"
    trace = gate.trace()
    assert trace == 0, "Trace of Z Gate should be 0"
    assert isinstance(trace, complex), "Trace should be a complex number"
    np.testing.assert_array_equal(gate.eigenvalues(), [-1, 1])
    np.testing.assert_array_equal(gate.eigenvectors(), [[0, 1], [1, 0]])

    # Identity commutes with any gate
    identity = IGate()
    pauli_x = XGate()
    pauli_h = HGate()
    assert identity.commute(pauli_x)
    assert pauli_x.commute(identity)
    assert identity.commute(pauli_h)
    assert pauli_h.commute(identity)

    # X commutes with itself but not with Z
    pauli_x = XGate()
    assert pauli_x.commute(pauli_x)
    pauli_x = XGate()
    pauli_z = ZGate()
    assert not pauli_x.commute(pauli_z)

    # Rotation gates can commute
    theta1 = np.pi / 4
    theta2 = np.pi / 6
    rz1 = RZGate(theta1)
    rz2 = RZGate(theta2)
    assert rz1.commute(rz2)

    # Gate must be the same shape for commutation check
    cx = CXGate()
    with pytest.raises(AssertionError):
        rz1.commute(cx)
