import numpy as np

def _check_quantum_state_array(X):
    """ Ensure that input are normalized state vectors of the right size."""
    if len(X.shape) != 2:
        raise ValueError("Input must be a 2D array.")
        
    # Check if all inputs are complex
    if not np.iscomplexobj(X):
        raise ValueError("Input must be a complex array.")

    # Check if input is normalized.
    normalized = np.allclose(np.linalg.norm(X, axis=-1), 1)
    if not normalized:
        not_normalized = X[np.linalg.norm(X, axis=-1) != 1]
        raise ValueError(f"Input state vectors must be normalized. Got non-normalized vectors: '{not_normalized}'")
    return True
