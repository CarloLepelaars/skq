import numpy as np

from src.gates.qutrit.base import QutritGate


class QutritMIGate(QutritGate):
    """
    Multi-qutrit Identity gate.
    :param num_qutrits: Number of qutrits in the gate.
    """

    def __new__(cls, num_qutrits: int):
        assert num_qutrits >= 1, "Number of qutrits must be at least 1."
        return super().__new__(cls, np.eye(3**num_qutrits))


class QutritCXAGate(QutritGate):
    """
    CNOT gate for qutrits.
    More information on Qutrit CNOT: https://www.iosrjournals.org/iosr-jap/papers/Vol10-issue6/Version-2/D1006021619.pdf
    Control on |1>
    """

    def __new__(cls):
        return super().__new__(
            cls, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]])
        )


class QutritCXBGate(QutritGate):
    """
    CNOT gate for qutrits.
    More information on Qutrit CNOT: https://www.iosrjournals.org/iosr-jap/papers/Vol10-issue6/Version-2/D1006021619.pdf
    Control on |2>
    """

    def __new__(cls):
        return super().__new__(
            cls, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]])
        )


class QutritCXCGate(QutritGate):
    """
    CNOT gate for qutrits.
    More information on Qutrit CNOT: https://www.iosrjournals.org/iosr-jap/papers/Vol10-issue6/Version-2/D1006021619.pdf
    Control on |0>
    """

    def __new__(cls):
        return super().__new__(
            cls, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0]])
        )


class QutritSWAPGate(QutritGate):
    """
    SWAP gate for qutrits.
    |01> -> |10>
    |10> -> |01>
    """

    def __new__(cls):
        swap_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return super().__new__(cls, swap_matrix)
