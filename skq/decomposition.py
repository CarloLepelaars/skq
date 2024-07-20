import numpy as np
from scipy.linalg import svd


def schmidt_decomposition(state_vector: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Perform Schmidt decomposition on a bipartite quantum state.
    :param state_vector: Bipartite quantum state vector
    :return: Tuple of Schmidt coefficients, Basis A and Basis B
    """
    assert len(state_vector) > 2, "Invalid state vector: Schmidt decomposition is not applicable for single qubit states."
    assert len(state_vector) % 2 == 0, "Invalid state vector: Not a bipartite state"
    
    # Infer dimensions
    N = len(state_vector)
    dim_A = int(np.sqrt(N))
    dim_B = N // dim_A

    # SVD on the state matrix
    state_matrix = state_vector.reshape(dim_A, dim_B)
    U, S, Vh = svd(state_matrix)

    # Coefficients (S), Basis A (U) and Basis B (Vh^dagger)
    return S, U, Vh.conj().T
