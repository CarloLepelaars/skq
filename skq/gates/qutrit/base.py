import numpy as np
from skq.gates.base import BaseGate


class QutritGate(BaseGate):
    """ Base class for Qutrit gates. """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_at_least_nxn(n=3), "Gate must be at least a 3x3 matrix"
        assert obj.is_power_of_n_shape(n=3), "Gate shape must be a power of 3"
        return obj
    
    def num_qutrits(self) -> int:
        """ Return the number of qutrits involved in the gate. """
        return int(np.log(self.shape[0]) / np.log(3))
    
    def is_multi_qutrit(self) -> bool:
        """ Check if the gate involves multiple qutrits. """
        return self.num_qutrits() > 1
