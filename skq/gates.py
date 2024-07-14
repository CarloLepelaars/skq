import numpy as np


class BaseGate(np.ndarray):
    """ Base class for quantum gates with NumPy. """
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=complex)
        obj = arr.view(cls)
        return obj

    def is_unitary(self) -> bool:
        """Check if the gate is unitary: U*U^dagger = I"""
        identity = np.eye(self.shape[0])
        return np.allclose(self @ self.conjugate_transpose(), identity)

    def is_hermitian(self) -> bool:
        """Check if the gate is Hermitian: U = U^dagger"""
        return np.allclose(self, self.conjugate_transpose())

    def eigenvalues(self) -> np.ndarray:
        """Compute and return the eigenvalues of the gate"""
        return np.linalg.eigvals(self)

    def eigenvectors(self) -> np.ndarray:
        """Compute and return the eigenvectors of the gate"""
        _, vectors = np.linalg.eig(self)
        return vectors
    
    def matrix_trace(self) -> complex:
        """Compute the trace of the gate"""
        return np.trace(self)

    def determinant(self) -> complex:
        """Compute the determinant of the gate"""
        return np.linalg.det(self)

    def conjugate_transpose(self) -> np.ndarray:
        """Return the conjugate transpose (Hermitian adjoint) of the gate"""
        return self.conj().T

    def frobenius_norm(self) -> float:
        """Compute the Frobenius norm of the gate"""
        return np.linalg.norm(self)
    
    def num_qubits(self) -> int:
        """Return the number of qubits involved in the gate."""
        return int(np.log2(self.shape[0]))
    
    def is_multi_qubit(self) -> bool:
        """Check if the gate involves multiple qubits."""
        return self.num_qubits() > 1
    
class CustomGate(BaseGate):
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        assert obj.is_unitary(), "Custom gate must be unitary"
        return obj
    
class IdentityGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, np.eye(2))
    
class PauliXGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[0, 1], 
                                     [1, 0]])
    
class PauliYGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[0, -1j], 
                                     [1j, 0]])
    
class PauliZGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0], 
                                     [0, -1]])
    
class HadamardGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 1], 
                                     [1, -1]]) / np.sqrt(2)
    
class PhaseGate(BaseGate):
    def __new__(cls, phi):
        obj = super().__new__(cls, [[1, 0], 
                                    [0, np.exp(1j * phi)]])
        obj.phi = phi
        return obj
    
class TGate(PhaseGate):
    def __new__(cls):
        phi = np.pi / 4
        return super().__new__(cls, phi=phi)
    
class SGate(PhaseGate):
    def __new__(cls):
        phi = np.pi / 2
        return super().__new__(cls, phi=phi)

class CXGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 1], 
                                     [0, 0, 1, 0]])

class CYGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, -1j], 
                                     [0, 0, 1j, 0]])
    
class CZGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 0, 0, -1]])
    
class CPhaseGate(BaseGate):
    def __new__(cls, phi):
        obj = super().__new__(cls, [[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, np.exp(1j * phi)]])
        obj.phi = phi
        return obj
    
class SWAPGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0], 
                                     [0, 0, 1, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 0, 1]])
    
class ToffoliGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, 1], 
                                     [0, 0, 0, 0, 0, 0, 1, 0]])
    
class FredkinGate(BaseGate):
    def __new__(cls):
        return super().__new__(cls, [[1, 0, 0, 0, 0, 0, 0, 0], 
                                     [0, 1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 1, 0, 0, 0, 0, 0], 
                                     [0, 0, 0, 1, 0, 0, 0, 0], 
                                     [0, 0, 0, 0, 1, 0, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 1, 0], 
                                     [0, 0, 0, 0, 0, 1, 0, 0], 
                                     [0, 0, 0, 0, 0, 0, 0, 1]])
    
class RotXGate(BaseGate):
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.cos(theta / 2), -1j * np.sin(theta / 2)], 
                                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
        obj.theta = theta
        return obj
    
class RotYGate(BaseGate):
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.cos(theta / 2), -np.sin(theta / 2)], 
                                     [np.sin(theta / 2), np.cos(theta / 2)]])
        obj.theta = theta
        return obj
    
class RotZGate(BaseGate):
    def __new__(cls, theta):
        obj = super().__new__(cls, [[np.exp(-1j * theta / 2), 0], 
                                     [0, np.exp(1j * theta / 2)]])
        obj.theta = theta
        return obj

class GeneralizedRotationGate(BaseGate):
    """ Also known as a U3 Gate. """
    def __new__(cls, theta_x, theta_y, theta_z):
        # Rotation matrices
        Rx = RotXGate(theta_x)
        Ry = RotYGate(theta_y)
        Rz = RotZGate(theta_z)
        combined_matrix = Rz @ Ry @ Rx
        
        obj = super().__new__(cls, combined_matrix)
        obj.theta_x = theta_x
        obj.theta_y = theta_y
        obj.theta_z = theta_z
        return obj
