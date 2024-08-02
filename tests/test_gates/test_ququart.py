import numpy as np

from skq.gates.ququart.single import *


def test_base_ququart_gate():
    gate = QuquartGate(np.eye(4, dtype=complex))
    assert gate.dtype == complex, "Gate must have complex dtype."
    assert gate.is_unitary(), "Gate must be unitary."
    assert gate.num_ququarts() == 1, "Gate must involve 1 ququart."
    assert not gate.is_multi_ququart(), "Gate must not involve multiple ququarts."

def test_ququart_x_gate():
    gate = QuquartXGate()
    zero_state = np.array([1, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, one_state), "X gate must transform |0> to |1>."
    assert np.allclose(gate @ one_state, two_state), "X gate must transform |1> to |2>."
    assert np.allclose(gate @ two_state, three_state), "X gate must transform |2> to |3>."
    assert np.allclose(gate @ three_state, zero_state), "X gate must transform |3> to |0>."

def test_ququart_z_gate():
    gate = QuquartZGate()
    zero_state = np.array([1, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, zero_state)
    assert np.allclose(gate @ one_state, [0, -1, 0, 0])
    assert np.allclose(gate @ two_state, [0, 0, 1, 0])
    assert np.allclose(gate @ three_state, [0, 0, 0, -1])

def test_ququart_h_gate():
    gate = QuquartHGate()
    zero_state = np.array([1, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1], dtype=complex)

    # Expected transformations
    expected_zero = np.array([1, 1, 1, 1]) / 2
    expected_one = np.array([1, -1, 1, -1]) / 2
    expected_two = np.array([1, 1, -1, -1]) / 2
    expected_three = np.array([1, -1, -1, 1]) / 2

    assert np.allclose(gate @ zero_state, expected_zero), "H gate transformation on |0> is incorrect."
    assert np.allclose(gate @ one_state, expected_one), "H gate transformation on |1> is incorrect."
    assert np.allclose(gate @ two_state, expected_two), "H gate transformation on |2> is incorrect."
    assert np.allclose(gate @ three_state, expected_three), "H gate transformation on |3> is incorrect."

def test_ququart_t_gate():
    gate = QuquartTGate()
    zero_state = np.array([1, 0, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0, 0], dtype=complex)
    two_state = np.array([0, 0, 1, 0], dtype=complex)
    three_state = np.array([0, 0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, [1, 0, 0, 0])
    assert np.allclose(gate @ one_state, [0, np.exp(1j * np.pi / 4), 0, 0])
    assert np.allclose(gate @ two_state, [0, 0, 1, 0])
    assert np.allclose(gate @ three_state, [0, 0, 0, 1])
