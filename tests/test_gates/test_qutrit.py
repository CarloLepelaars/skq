import numpy as np
from skq.gates.qutrit import *

def test_base_qutrit_gate():
    # QutritXGate
    gate = QutritGate(np.array([[1, 0, 0], 
                                [0, 0, 1], 
                                [0, 1, 0]]))
    assert gate.dtype == complex, "Gate must have complex dtype."
    assert gate.is_unitary(), "Gate must be unitary."
    assert gate.num_qutrits() == 1, "Gate must involve 1 qutrit."
    assert not gate.is_multi_qutrit(), "Gate must not involve multiple qutrits."

def test_qutritxgate():
    gate = QutritXGate()
    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, one_state), "X gate must transform |0> to |1>."
    assert np.allclose(gate @ one_state, two_state), "X gate must transform |1> to |2>."
    assert np.allclose(gate @ two_state, zero_state), "X gate must transform |2> to |0>."

def test_qutritygate():
    gate = QutritYGate()
    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, [0, -1j, 0])
    assert np.allclose(gate @ one_state, [0, 0, -1j])
    assert np.allclose(gate @ two_state, [-1j, 0, 0])

def test_qutritzgate():
    gate = QutritZGate()
    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, zero_state)
    assert np.allclose(gate @ one_state, [0, np.exp(2*np.pi*1j/3), 0])
    assert np.allclose(gate @ two_state, [0, 0, np.exp(-2*np.pi*1j/3)])

def test_qutrithgate():
    gate = QutritHGate()
    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    # Expected transformations
    expected_zero = np.array([1, 1, 1]) / np.sqrt(3)
    expected_one = np.array([1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]) / np.sqrt(3)
    expected_two = np.array([1, np.exp(4j * np.pi / 3), np.exp(2j * np.pi / 3)]) / np.sqrt(3)

    assert np.allclose(gate @ zero_state, expected_zero), "QutritHGate transformation on |0> is incorrect."
    assert np.allclose(gate @ one_state, expected_one), "QutritHGate transformation on |1> is incorrect."
    assert np.allclose(gate @ two_state, expected_two), "QutritHGate transformation on |2> is incorrect."

def test_qutrittgate():
    gate = QutritTGate()
    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    assert np.allclose(gate @ zero_state, [1, 0, 0])
    assert np.allclose(gate @ one_state, [0, np.exp(2*np.pi*1j/9), 0])
    assert np.allclose(gate @ two_state, [0, 0, np.exp(-2*np.pi*1j/9)])

def test_qutrit_phase_gate():
    phi_0, phi_1, phi_2 = np.pi/3, np.pi/2, np.pi
    gate = QutritPhaseGate(phi_0, phi_1, phi_2)

    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    # Expected transformations
    expected_zero = np.array([np.exp(1j * phi_0), 0, 0])
    expected_one = np.array([0, np.exp(1j * phi_1), 0])
    expected_two = np.array([0, 0, np.exp(1j * phi_2)])

    assert np.allclose(gate @ zero_state, expected_zero), "QutritPhaseGate transformation on |0> is incorrect."
    assert np.allclose(gate @ one_state, expected_one), "QutritPhaseGate transformation on |1> is incorrect."
    assert np.allclose(gate @ two_state, expected_two), "QutritPhaseGate transformation on |2> is incorrect."

def test_qutrit_s_gate():
    # Special case of phase gate with phi_0=0, phi_1=2pi/3, phi_2=4pi/3
    gate = QutritSGate()

    zero_state = np.array([1, 0, 0], dtype=complex)
    one_state = np.array([0, 1, 0], dtype=complex)
    two_state = np.array([0, 0, 1], dtype=complex)

    # Expected transformations
    expected_zero = np.array([1, 0, 0])
    expected_one = np.array([0, np.exp(2j * np.pi / 3), 0])
    expected_two = np.array([0, 0, np.exp(4j * np.pi / 3)])

    assert np.allclose(gate @ zero_state, expected_zero), "QutritSGate transformation on |0> is incorrect."
    assert np.allclose(gate @ one_state, expected_one), "QutritSGate transformation on |1> is incorrect."
    assert np.allclose(gate @ two_state, expected_two), "QutritSGate transformation on |2> is incorrect."

def test_qutrit_swap_gate():
    swap_gate = QutritSWAPGate()
    # |01>
    initial_state = np.zeros((9,), dtype=complex)
    initial_state[1] = 1 
    # |10>
    expected_state = np.zeros((9,), dtype=complex)
    expected_state[3] = 1
    # |01> -> |10>
    assert np.allclose(swap_gate @ initial_state, expected_state), "SWAP gate did not perform correctly."
    # |10> -> |01>
    assert np.allclose(swap_gate @ expected_state, initial_state), "SWAP gate did not perform correctly."

    # |12>
    initial_state = np.zeros((9,), dtype=complex)
    initial_state[5] = 1
    # |21>
    expected_state = np.zeros((9,), dtype=complex)
    expected_state[7] = 1
    # |12> -> |21>
    assert np.allclose(swap_gate @ initial_state, expected_state), "SWAP gate did not perform correctly."
    # |21> -> |12>
    assert np.allclose(swap_gate @ expected_state, initial_state), "SWAP gate did not perform correctly."

    # |20>
    initial_state = np.zeros((9,), dtype=complex)
    initial_state[6] = 1
    # |02>
    expected_state = np.zeros((9,), dtype=complex)
    expected_state[2] = 1
    # |20> -> |02>
    assert np.allclose(swap_gate @ initial_state, expected_state), "SWAP gate did not perform correctly."
    # |02> -> |20>
    assert np.allclose(swap_gate @ expected_state, initial_state), "SWAP gate did not perform correctly."
