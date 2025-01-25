import pytest
import qiskit
import numpy as np
import pennylane as qml

from src.gates.global_phase import *


def test_base_gate():
    qu_scalar = GlobalPhase(np.pi / 4)
    assert qu_scalar.scalar == pytest.approx(np.exp(1j * np.pi / 4))
    assert qu_scalar.scalar == pytest.approx(np.sqrt(2) / 2 + 1j * np.sqrt(2) / 2)
    assert qu_scalar.phase == pytest.approx(np.pi / 4)
    assert qu_scalar.inverse().phase == pytest.approx(-np.pi / 4)
    assert qu_scalar.combine(GlobalPhase(-np.pi)).phase == pytest.approx(-3 * np.pi / 4)
    assert qu_scalar.multiply(GlobalPhase(np.pi / 2)).phase == pytest.approx(3 * np.pi / 4)
    # Qiskit conversion comparisons
    qiskit_gate = qu_scalar.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.library.GlobalPhaseGate)
    assert qiskit_gate.params[0] == qu_scalar.phase
    assert qiskit_gate.inverse().params[0] == qu_scalar.inverse().phase


def test_identity():
    identity = Identity()
    assert identity.scalar == 1
    assert identity.phase == 0
    assert identity.inverse().phase == 0
    assert identity.combine(Identity()).phase == pytest.approx(0)
    assert identity.multiply(Identity()).phase == pytest.approx(0)


def test_quarter_phase():
    quarter_phase = QuarterPhase()
    assert quarter_phase.scalar == pytest.approx(1j)
    assert quarter_phase.phase == pytest.approx(np.pi / 2)
    assert quarter_phase.inverse().phase == pytest.approx(-np.pi / 2)
    assert quarter_phase.combine(QuarterPhase()).phase == pytest.approx(np.pi)
    assert quarter_phase.multiply(QuarterPhase()).phase == pytest.approx(np.pi)


def test_half_phase():
    half_phase = HalfPhase()
    assert half_phase.scalar == pytest.approx(-1)
    assert half_phase.phase == pytest.approx(np.pi)
    assert half_phase.inverse().phase == pytest.approx(-np.pi)
    assert half_phase.combine(HalfPhase()).phase == pytest.approx(0)
    assert half_phase.multiply(HalfPhase()).phase == pytest.approx(0)


def test_three_quarter_phase():
    three_quarter_phase = ThreeQuarterPhase()
    assert three_quarter_phase.scalar == pytest.approx(-1j)
    assert three_quarter_phase.phase == pytest.approx(-np.pi / 2)
    assert three_quarter_phase.inverse().phase == pytest.approx(np.pi / 2)
    assert three_quarter_phase.combine(ThreeQuarterPhase()).phase == pytest.approx(np.pi)
    assert three_quarter_phase.multiply(ThreeQuarterPhase()).phase == pytest.approx(np.pi)


def test_full_phase():
    full_phase = FullPhase()
    assert full_phase.scalar == pytest.approx(1)
    assert full_phase.phase == pytest.approx(0)
    assert full_phase.inverse().phase == pytest.approx(0)
    assert full_phase.combine(FullPhase()).phase == pytest.approx(0)
    assert full_phase.multiply(FullPhase()).phase == pytest.approx(0)


def test_to_qiskit():
    qu_scalar = GlobalPhase(np.pi / 4)
    qiskit_gate = qu_scalar.to_qiskit()
    assert isinstance(qiskit_gate, qiskit.circuit.library.GlobalPhaseGate)
    assert qiskit_gate.params[0] == qu_scalar.phase
    assert qiskit_gate.inverse().params[0] == qu_scalar.inverse().phase


def test_from_qiskit():
    qiskit_gate = qiskit.circuit.library.GlobalPhaseGate(np.pi / 4)
    qu_scalar = GlobalPhase.from_qiskit(qiskit_gate)
    assert qu_scalar.phase == pytest.approx(np.pi / 4)


def test_to_pennylane():
    qu_scalar = GlobalPhase(np.pi / 4)
    pennylane_gate = qu_scalar.to_pennylane()
    assert isinstance(pennylane_gate, qml.GlobalPhase)
    assert pennylane_gate.parameters[0] == qu_scalar.phase


def test_from_pennylane():
    pennylane_gate = qml.GlobalPhase(np.pi / 4)
    qu_scalar = GlobalPhase.from_pennylane(pennylane_gate)
    assert qu_scalar.phase == pytest.approx(np.pi / 4)
