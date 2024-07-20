import numpy as np
from skq.gates.quscalar.base import QuScalarGate

class Identity(QuScalarGate):
    """ No phase shift. """
    def __new__(cls) -> 'Identity':
        return super().__new__(cls, 0.0)
    
class QuarterPhase(QuScalarGate):
    """ Quarter phase shift (π/2) """
    def __new__(cls) -> 'QuarterPhase':
        return super().__new__(cls, np.pi/2)
    
class HalfPhase(QuScalarGate):
    """ Half phase shift (π) """
    def __new__(cls) -> 'HalfPhase':
        return super().__new__(cls, np.pi)
    
class ThreeQuarterPhase(QuScalarGate):
    """ Three quarters phase shift (3π/2) """
    def __new__(cls) -> 'ThreeQuarterPhase':
        return super().__new__(cls, 3*np.pi/2)
    
class FullPhase(QuScalarGate):
    """ Full phase shift (2π) """
    def __new__(cls) -> 'FullPhase':
        return super().__new__(cls, 2*np.pi)
