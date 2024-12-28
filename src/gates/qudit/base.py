import numpy as np
from src.gates.base import BaseGate


class QuditGate(BaseGate):
    """
    Base class for Qudit gates.
    These are quantum systems with a basis of d states. |0>, |1>, |2>, ..., |d-1>.
    Models spin-d/2 particles like baryons.
    """

    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_at_least_nxn(n=4), "Gate must be at least a 4x4 matrix"
        assert obj.is_power_of_n_shape(n=4), "Gate shape must be a power of 4"
        return obj

    def num_qudits(self) -> int:
        """Return the number of qudits involved in the gate."""
        return int(np.log(self.shape[0]) / np.log(4))

    def is_multi_qudit(self) -> bool:
        """Check if the gate involves multiple qudits."""
        return self.num_qudits() > 1
