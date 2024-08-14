import numpy as np
from scipy.linalg import sqrtm

from skq.quantum_info.superoperator import SuperOperator


class QuantumChannel(SuperOperator):
    """ Quantum Channel representation in choi, stinespring, or kraus form. """
    def __new__(cls, input_array: np.array, representation="choi"):
        cls.representation = representation
        obj = super().__new__(cls, input_array)
        if cls.representation == "choi":
            obj._validate_choi()
        elif cls.representation == "stinespring":
            obj._validate_stinespring()
        elif cls.representation == "kraus":
            obj._validate_kraus()
        else:
            raise ValueError("Invalid representation. Choose from 'choi', 'stinespring', or 'kraus'.")
        return obj
    
    def _validate_choi(self):
        """ Validate if the input matrix is a valid choi matrix. """
        assert self.is_nd(2), "Choi matrix must be a 2D matrix."
        assert self.is_square(), "Choi matrix must be a square matrix."
        assert self.is_positive_semidefinite(), "Choi matrix must be positive semidefinite."
        assert self.is_trace_preserving(), "Choi matrix must be trace-preserving."
    
    def _validate_stinespring(self):
        """ Validate if the input matrix is a valid stinespring matrix. """
        assert self.is_nd(2), "Stringspring matrix must be a 2D matrix."
        assert self.is_isometry(), "Stinespring matrix must be an isometry."
        assert self.shape[0] > self.shape[1], "Stinespring matrix must have more rows than columns."

    def _validate_kraus(self):
        """ Validate if the input list of matrices is a valid kraus representation. """
        assert self.is_nd(3), "Kraus operators must be 3D matrices."
        for kraus_op in self:
            assert kraus_op.shape[0] == kraus_op.shape[1], "Each Kraus operator must be a square matrix."
        assert self.is_trace_preserving(), "Sum of outer products of kraus operators must be identity."
    
    def is_nd(self, n: int) -> bool:
        """ Channel is an n-dimensional matrix. """
        return self.ndim == n
    
    def is_square(self) -> bool:
        """ Check if the superoperator is represented by a square matrix. """
        return self.shape[0] == self.shape[1]
    
    def is_positive_semidefinite(self) -> bool:
        """ All eigenvalues are non-negative. Requirement for choi matrix. """
        eigenvalues = np.linalg.eigvalsh(self)
        return np.all(eigenvalues >= 0)
    
    def is_trace_preserving(self) -> bool:
        """ Check if the quantum channel is trace-preserving. """
        if self.representation == "kraus":
            d = self[0].shape[0]
            kraus_sum = sum([np.dot(k.conjugate_transpose(), k) for k in self])
            return np.allclose(kraus_sum, np.eye(d))
        elif self.representation == "choi":
            d = int(np.sqrt(self.shape[0]))
            partial_trace = np.trace(self.reshape(d, d, d, d), axis1=1, axis2=3)
            return np.allclose(partial_trace, np.eye(d))
        else:
            raise ValueError(f"Trace-preserving check is not implemented for representation '{self.representation}'.")
        
    def is_isometry(self) -> bool:
        """ 
        Check if the quantum channel is an isometry. 
        An isometry is a linear transformation that preserves distances (V^dagger V = I)
        """
        return np.allclose(self.conjugate_transpose() @ self, np.eye(self.shape[1]))
    
    def compose(self, other: "QuantumChannel") -> "QuantumChannel":
        """
        Compose quantum channels.
        :param other: QuantumChannel object to compose with.
        :return: Composed QuantumChannel object in the Kraus representation.
        """
        assert isinstance(other, QuantumChannel), "Input must be a QuantumChannel."
        self_kraus = self.to_kraus()
        other_kraus = other.to_kraus()
        composed_kraus = []

        for k1 in self_kraus:
            for k2 in other_kraus:
                composed_kraus.append(np.dot(k1, k2))

        composed_kraus = np.array(composed_kraus, dtype=np.complex128)
        kraus_sum = sum(np.dot(k.conj().T, k) for k in composed_kraus)
        normalization_factor = np.linalg.inv(sqrtm(kraus_sum).astype(np.complex128))
        normalized_kraus = [np.dot(normalization_factor, k) for k in composed_kraus]
        return QuantumChannel(np.array(normalized_kraus), representation="kraus")
    
    def tensor(self, other: "QuantumChannel") -> "QuantumChannel":
        """ 
        Tensor product with another channel. 
        :param other: QuantumChannel object to tensor with.
        :return: QuantumChannel object in the Choi representation
        """
        assert isinstance(other, QuantumChannel), "The other object must be a QuantumChannel."
        return QuantumChannel(np.kron(self.to_choi(), other.to_choi()), representation="choi")
    
    def fidelity(self, other: "QuantumChannel") -> float:
        """ 
        Fidelity between two quantum channels. 
        :param other: QuantumChannel object to calculate fidelity with.
        :return: Fidelity in range [0...1].
        """
        assert isinstance(other, QuantumChannel), "The other object must be a QuantumChannel."
        choi_self = self.to_choi()
        choi_other = other.to_choi()
        sqrt_choi_self = sqrtm(choi_self)
        product_matrix = sqrt_choi_self @ choi_other @ sqrt_choi_self
        fidelity_value = np.trace(sqrtm(product_matrix)) ** 2
        # Normalize fidelity
        d = np.sqrt(choi_self.shape[0])
        fidelity_value /= d ** 2
        return np.real(fidelity_value)
    
    def to_choi(self):
        """ Convert the channel to the choi matrix representation. """
        if self.representation == "choi":
            return self
        elif self.representation == "stinespring":
            return self._stinespring_to_choi()
        elif self.representation == "kraus":
            return self._kraus_to_choi()
    
    def to_stinespring(self):
        """ Convert the channel to the stinespring representation. """
        if self.representation == "stinespring":
            return self
        elif self.representation == "choi":
            return self._choi_to_stinespring()
        elif self.representation == "kraus":
            return self._kraus_to_stinespring()
    
    def to_kraus(self):
        """ Convert the channel to the kraus representation. """
        if self.representation == "kraus":
            return self
        elif self.representation == "choi":
            return self._choi_to_kraus()
        elif self.representation == "stinespring":
            return self._stinespring_to_kraus()
    
    def _stinespring_to_choi(self) -> "QuantumChannel":
        """ Convert stinespring representation to choi matrix. """
        return QuantumChannel(np.dot(self, self.conjugate_transpose()), representation="choi")
    
    def _choi_to_stinespring(self) -> "QuantumChannel":
        """ Convert choi matrix to stinespring representation. """
        d = int(np.sqrt(self.shape[0])) 
        w, v = np.linalg.eigh(self)
        # Filter out small eigenvalues to avoid numerical instability
        non_zero_indices = np.where(w > 1e-10)[0]
        sqrt_eigenvals = np.sqrt(w[non_zero_indices])
        v = v[:, non_zero_indices]
        r = len(non_zero_indices)
        stinespring_matrix = np.zeros((d * r, d), dtype=complex)
        for i in range(r):
            stinespring_matrix[i * d:(i + 1) * d, :] = sqrt_eigenvals[i] * v[:, i].reshape(d, d)
        return QuantumChannel(stinespring_matrix, representation="stinespring")
    
    def _kraus_to_choi(self) -> "QuantumChannel":
        """ Convert Kraus representation to Choi matrix using vectorization. """
        d = self[0].shape[0]  
        choi_matrix = np.zeros((d * d, d * d), dtype=complex)
        for k in self:
            choi_matrix += np.kron(k, k.conj())
        return QuantumChannel(choi_matrix / d, representation="choi")
    
    def _choi_to_kraus(self) -> "QuantumChannel":
        """ Convert choi matrix to kraus operators. """
        d = int(np.sqrt(self.shape[0]))
        kraus_operators = []
        w, v = np.linalg.eigh(self)
        for i in range(len(w)):
            if np.isclose(w[i], 0):
                continue
            kraus_operators.append(np.sqrt(w[i]) * v[:, i].reshape(d, d))
        return QuantumChannel(np.array(kraus_operators, dtype=np.complex128), representation="kraus")
    
    def _kraus_to_stinespring(self) -> "QuantumChannel":
        """ Convert kraus representation to stinespring representation. """
        d = self[0].shape[0]
        num_kraus = len(self)
        stinespring_matrix = np.zeros((d * num_kraus, d), dtype=np.complex128)
        for i, k in enumerate(self):
            stinespring_matrix[i * d:(i + 1) * d, :] = k
        return QuantumChannel(stinespring_matrix, representation="stinespring")
    
    def _stinespring_to_kraus(self) -> "QuantumChannel":
        """ Convert stinespring representation to kraus operators. """
        d = self.shape[1]
        num_kraus = self.shape[0] // d
        kraus_operators = []
        for i in range(num_kraus):
            kraus_operators.append(self[i * d:(i + 1) * d, :])
        return QuantumChannel(np.array(kraus_operators, dtype=np.complex128), representation="kraus")
    