import numpy as np
from skq.gates.qupent import *


def test_base_qupent_gate():
    gate = QupentGate(np.eye(5, dtype=complex))
    assert gate.dtype == complex, "Gate must have complex dtype."
    assert gate.is_unitary(), "Gate must be unitary."
    assert gate.num_qupents() == 1, "Gate must involve 1 qupent."
    assert not gate.is_multi_qupent(), "Gate must not involve multiple qupents."


def test_qupent_i_gate():
    gate = QupentI()
    state = np.array([1, 0, 0, 0, 0], dtype=complex)
    assert np.allclose(gate @ state, state), "I gate must not change the state."


def test_qupent_x_gate():
    gate = QupentX()
    zero_state = np.array([1, 0, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1, 0], dtype=complex)
    four_state = np.array([0, 0, 0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, one_state), "X gate must transform |0> to |1>."
    assert np.allclose(gate @ one_state, two_state), "X gate must transform |1> to |2>."
    assert np.allclose(gate @ two_state, three_state), "X gate must transform |2> to |3>."
    assert np.allclose(gate @ three_state, four_state), "X gate must transform |3> to |4>."
    assert np.allclose(gate @ four_state, zero_state), "X gate must transform |4> to |0>."


def test_qupent_z_gate():
    gate = QupentZ()
    zero_state = np.array([1, 0, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1, 0], dtype=complex)
    four_state = np.array([0, 0, 0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, zero_state), "Z gate must not change the |0> state."
    assert np.allclose(gate @ one_state, one_state * np.exp(2j * np.pi / 5)), "Z gate must correctly phase shift the |1> state."
    assert np.allclose(gate @ two_state, two_state * np.exp(4j * np.pi / 5)), "Z gate must correctly phase shift the |2> state."
    assert np.allclose(gate @ three_state, three_state * np.exp(6j * np.pi / 5)), "Z gate must correctly phase shift the |3> state."
    assert np.allclose(gate @ four_state, four_state * np.exp(8j * np.pi / 5)), "Z gate must correctly phase shift the |4> state."


def test_qupent_h_gate():
    gate = QupentH()
    zero_state = np.array([1, 0, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1, 0], dtype=complex)
    four_state = np.array([0, 0, 0, 0, 1], dtype=complex)

    # Expected transformations
    expected_zero = np.array([1, 1, 1, 1, 1]) / np.sqrt(5)
    expected_one = np.array([1, np.exp(2j * np.pi / 5), np.exp(4j * np.pi / 5), np.exp(6j * np.pi / 5), np.exp(8j * np.pi / 5)]) / np.sqrt(5)
    expected_two = np.array([1, np.exp(4j * np.pi / 5), np.exp(8j * np.pi / 5), np.exp(2j * np.pi / 5), np.exp(6j * np.pi / 5)]) / np.sqrt(5)
    expected_three = np.array([1, np.exp(6j * np.pi / 5), np.exp(2j * np.pi / 5), np.exp(8j * np.pi / 5), np.exp(4j * np.pi / 5)]) / np.sqrt(5)
    expected_four = np.array([1, np.exp(8j * np.pi / 5), np.exp(6j * np.pi / 5), np.exp(4j * np.pi / 5), np.exp(2j * np.pi / 5)]) / np.sqrt(5)

    assert np.allclose(gate @ zero_state, expected_zero), "H gate transformation on |0> is incorrect."
    assert np.allclose(gate @ one_state, expected_one), "H gate transformation on |1> is incorrect."
    assert np.allclose(gate @ two_state, expected_two), "H gate transformation on |2> is incorrect."
    assert np.allclose(gate @ three_state, expected_three), "H gate transformation on |3> is incorrect."
    assert np.allclose(gate @ four_state, expected_four), "H gate transformation on |4> is incorrect."


def test_qupent_t_gate():
    gate = QupentT()
    zero_state = np.array([1, 0, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1, 0], dtype=complex)
    four_state = np.array([0, 0, 0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, zero_state), "T gate must not change the |0> state."
    assert np.allclose(gate @ one_state, one_state * np.exp(1j * np.pi / 10)), "T gate must correctly phase shift the |1> state."
    assert np.allclose(gate @ two_state, two_state * np.exp(2j * np.pi / 10)), "T gate must correctly phase shift the |2> state."
    assert np.allclose(gate @ three_state, three_state * np.exp(3j * np.pi / 10)), "T gate must correctly phase shift the |3> state."
    assert np.allclose(gate @ four_state, four_state * np.exp(4j * np.pi / 10)), "T gate must correctly phase shift the |4> state."
