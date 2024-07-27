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
    