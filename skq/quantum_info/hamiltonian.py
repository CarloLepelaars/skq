import qiskit
import numpy as np
import scipy.linalg
import pennylane as qml

from skq.base import Operator


class Hamiltonian(Operator):
    """
    Class representing a Hamiltonian in quantum computing.

    :param input_array: The input array representing the Hamiltonian. Will be converted to a complex numpy array.
    :param hbar: The reduced Planck constant. Default is 1.0 (natural units). 
    If you want to use the actual physical value, set hbar to 1.0545718e-34.
    """
    def __new__(cls, input_array, hbar: float = 1.0):
        assert hbar > 0, "The reduced Planck constant must be greater than zero."
        obj = super().__new__(cls, input_array)
        assert obj.is_at_least_nxn(n=2), "Hamiltonian must be at least a 2x2 matrix."
        assert obj.is_hermitian(), "Hamiltonian must be Hermitian."
        obj.hbar = hbar
        return obj
    
    def num_qubits(self) -> int:
        """ Return the number of qubits in the Hamiltonian. """
        return int(np.log2(self.shape[0]))
    
    def eigenvalues(self) -> np.ndarray:
        """ 
        Return the eigenvalues of the Hamiltonian. 
        Optimized for Hermitian matrices.
        :return: Array of eigenvalues.
        """
        return np.linalg.eigvalsh(self)

    def eigenvectors(self) -> np.ndarray:
        """ 
        Return the eigenvectors of the Hamiltonian.
        Optimized for Hermitian matrices.
        :return: Array of eigenvectors.
        """
        _, vectors = np.linalg.eigh(self)
        return vectors
    
    def is_multi_qubit(self) -> bool:
        """ Check if the gate involves multiple qubits. """
        return self.num_qubits() > 1
    
    def time_evolution_operator(self, t: float) -> np.ndarray:
        """ Time evolution operator U(t) = exp(-iHt/hbar). """
        return scipy.linalg.expm(-1j * self * t / self.hbar)

    def ground_state_energy(self) -> float:
        """ Ground state energy. i.e. the smallest eigenvalue. """
        eigenvalues = self.eigenvalues()
        return eigenvalues[0]

    def ground_state(self) -> np.ndarray:
        """ Compute the ground state. i.e. the eigenvector corresponding to the smallest eigenvalue. """
        _, eigenvectors = np.linalg.eigh(self)
        return eigenvectors[:, 0]
    
    def convert_endianness(self) -> 'Hamiltonian':
        """ Convert a Hamiltonian from big-endian to little-endian and vice versa. """
        num_qubits = self.num_qubits()
        perm = np.argsort([int(bin(i)[2:].zfill(num_qubits)[::-1], 2) for i in range(2**num_qubits)])
        return self[np.ix_(perm, perm)]

    def to_qiskit(self) -> qiskit.quantum_info.Operator:
        """
        Convert the scikit-q Hamiltonian to a Qiskit Operator object.
        Qiskit using little endian convention, so we permute the order of the qubits.
        :return: Qiskit Operator object
        """
        return qiskit.quantum_info.Operator(self.convert_endianness())

    @staticmethod
    def from_qiskit(operator: qiskit.quantum_info.Operator) -> 'Hamiltonian':
        """
        Create a scikit-q Hamiltonian object from a Qiskit Operator object.
        Qiskit using little endian convention, so we permute the order of the qubits.
        :param operator: Qiskit Operator object
        :return: Hamiltonian object
        """
        return Hamiltonian(operator.data).convert_endianness()

    def to_pennylane(self, wires: list[int] | int = None, **kwargs) -> 'qml.Hamiltonian':
        """
        Convert the scikit-q Hamiltonian to a PennyLane Hamiltonian.
        :param wires: List of wires to apply the Hamiltonian to
        kwargs are passed to the PennyLane Hamiltonian constructor.
        :return: PennyLane Hamiltonian object
        """
        coefficients = [1.0]
        wires = wires if wires is not None else list(range(self.num_qubits()))
        observables = [qml.Hermitian(self, wires=wires)]
        return qml.Hamiltonian(coefficients, observables, **kwargs)

    @staticmethod
    def from_pennylane(hamiltonian: qml.Hamiltonian) -> "Hamiltonian":
        """
        Convert a PennyLane Hamiltonian object to a scikit-q Hamiltonian object.
        :param hamiltonian: PennyLane Hamiltonian object
        :return: Hamiltonian object
        """
        assert len(hamiltonian.ops) == 1, "Only single-term Hamiltonians are supported."
        return Hamiltonian(hamiltonian.ops[0].matrix())
