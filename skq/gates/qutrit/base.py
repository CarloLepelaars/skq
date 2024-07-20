import numpy as np
from skq.gates.base import BaseGate

class QutritGate(BaseGate):
    """ Base class for Qutrit gates. """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_at_least_3x3(), "Gate must be at least a 3x3 matrix"
        assert obj.is_power_of_three_shape(), "Gate shape must be a power of 2"
        return obj

    def is_at_least_3x3(self) -> bool:
        """ Check if the gate is at least a 3x3 matrix. """
        return self.shape[0] >= 3 and self.shape[1] >= 3
    
    def is_power_of_three_shape(self) -> bool:
        """ 
        Check if the gate shape is a power of 3. 
        1 Qutrit system = 3x3
        2 Qutrit system = 9x9
        3 Qutrit system = 27x27
        """
        rows, cols = self.shape
        return rows == cols and (rows & (rows - 1) == 0) and rows > 0 and (rows % 3 == 0)
    
    def num_qutrits(self) -> int:
        """ Return the number of qutrits involved in the gate. """
        return int(np.log2(self.shape[0] / 3))
    
    def is_multi_qutrit(self) -> bool:
        """ Check if the gate involves multiple qutrits. """
        return self.num_qutrits() > 1
