import numpy as np

from src.gates.qudit.base import QuditGate


class QuditI(QuditGate):
    """Identity gate for qudits."""

    def __new__(cls):
        obj = super().__new__(cls, np.eye(4))
        return obj


class QuditX(QuditGate):
    """X gate for qudits."""

    def __new__(cls):
        obj = super().__new__(cls, np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
        return obj


class QuditZ(QuditGate):
    """Z gate for qudits."""

    def __new__(cls):
        obj = super().__new__(cls, np.array([[1, 0, 0, 0], [0, 1j, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1j]]))
        return obj


class QuditH(QuditGate):
    """Hadamard gate for qudits."""

    def __new__(cls):
        obj = super().__new__(cls, np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]]) / 2)
        return obj


class QuditT(QuditGate):
    """T gate for qudits."""

    def __new__(cls):
        obj = super().__new__(cls, np.diag([1, np.exp(1j * np.pi / 4), 1, 1]))
        return obj
