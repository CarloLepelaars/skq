import qiskit
import numpy as np

from skq.gates.base import BaseGate

class QuScalarGate(BaseGate):
    """
    Class representing a quantum scalar gate.
    This intends to simulate a spin 0 particle. In practice it acts as a global phase operator.
    :param phase: Phase of the scalar gate
    """
    def __new__(cls, phase: float) -> 'QuScalarGate':
        input_array = np.array([[np.exp(1j * phase)]], dtype=complex)
        obj = super().__new__(cls, input_array)
        assert obj.is_1x1(), "Quscalar must be a 1x1 matrix"
        return obj
    
    @property
    def scalar(self) -> complex:
        """ Get the scalar value of the gate. """
        return self[0, 0]
    
    @property
    def phase(self) -> float:
        """ Get the phase of the scalar gate. """
        return np.angle(self.scalar)
    
    def is_1x1(self) -> bool:
        """ Check if the gate is a 1x1 matrix. """
        return self.shape == (1, 1)
    
    def inverse(self) -> "QuScalarGate":
        inverse_phase = -self.phase
        return QuScalarGate(inverse_phase)
    
    def combine(self, other: "QuScalarGate") -> "QuScalarGate":
        combined_phase = self.phase + other.phase
        return QuScalarGate(combined_phase)

    def multiply(self, other: "QuScalarGate") -> "QuScalarGate":
        multiplied_phase = np.angle(self.scalar * other.scalar)
        return QuScalarGate(multiplied_phase)
    
    def to_qiskit(self) -> qiskit.circuit.library.GlobalPhaseGate:
        """ Convert QuScalar to a Qiskit GlobalPhaseGate object. """
        return qiskit.circuit.library.GlobalPhaseGate(self.phase)
    
    @staticmethod
    def from_qiskit(qiskit_gate: qiskit.circuit.library.GlobalPhaseGate) -> "QuScalarGate":
        """ 
        Convert a Qiskit GlobalPhaseGate to a QuScalar object. 
        :param qiskit_gate: Qiskit GlobalPhaseGate object
        :return: A QuScalar object
        """
        if not isinstance(qiskit_gate, qiskit.circuit.library.GlobalPhaseGate):
            raise ValueError(f"Expected GlobalPhaseGate, got {type(qiskit_gate)}.")
        phase = qiskit_gate.params[0]
        return QuScalarGate(phase)
    