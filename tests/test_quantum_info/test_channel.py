import pytest
import numpy as np

from skq.quantum_info.channel import *
from skq.quantum_info.density import DensityMatrix


def test_choi_validation():
    choi_matrix = np.array([[1, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]], dtype=complex) / 2
    channel = QuantumChannel(choi_matrix, representation="choi")
    assert channel.is_nd(2), "Choi matrix should be 2D"
    assert channel.is_square(), "Choi matrix should be square"
    assert channel.is_positive_semidefinite(), "Choi matrix should be positive semidefinite"
    assert channel.is_trace_preserving(), "Choi matrix should be trace-preserving"

def test_stinespring_validation():
    stinespring_matrix = np.array([[1, 0], 
                                   [0, 1], 
                                   [0, 0], 
                                   [0, 0]], dtype=complex)
    channel = QuantumChannel(stinespring_matrix, representation="stinespring")
    assert channel.is_nd(2), "Stinespring matrix should be 2D"
    assert channel.is_isometry(), "Stinespring matrix should be an isometry"
    assert channel.shape[0] > channel.shape[1], "Stinespring matrix should have more rows than columns"

def test_kraus_validation():
    # Correcting the Kraus operators to ensure trace-preserving condition
    kraus_operators = np.array([np.eye(2, dtype=complex) / np.sqrt(2), 
                                np.eye(2, dtype=complex) / np.sqrt(2)])
    channel = QuantumChannel(kraus_operators, representation="kraus")
    
    assert channel.is_nd(3), "Kraus operators should be 3D"
    assert channel.is_trace_preserving(), "Kraus representation should be trace-preserving"

def test_convert_to_choi():
    kraus_operators = np.array([np.eye(2, dtype=complex) / np.sqrt(2), 
                                np.eye(2, dtype=complex) / np.sqrt(2)])
    channel = QuantumChannel(kraus_operators, representation="kraus")
    choi_matrix = channel.to_choi()
    expected_choi = np.eye(4, dtype=complex) / 2
    assert choi_matrix.shape == (4, 4), "Converted Choi matrix should be 4x4"
    assert np.allclose(choi_matrix, expected_choi), "Converted Choi matrix should match expected scaled identity matrix"

def test_convert_to_stinespring():
    choi_matrix = np.eye(4, dtype=complex) / 2
    channel = QuantumChannel(choi_matrix, representation="choi")
    stinespring_matrix = channel.to_stinespring()
    assert stinespring_matrix.shape == (8, 2), "Converted Stinespring matrix should have correct shape"

def test_convert_to_kraus():
    stinespring_matrix = np.array([[1, 0], 
                                   [0, 1], 
                                   [0, 0], 
                                   [0, 0]], dtype=complex)
    channel = QuantumChannel(stinespring_matrix, representation="stinespring")
    kraus_operators = channel.to_kraus()
    
    assert len(kraus_operators) == 2, "Converted Kraus operators should have correct number"
    for k in kraus_operators:
        assert k.shape == (2, 2), "Each Kraus operator should be 2x2"

def test_invalid_choi_matrix():
    # Not positive semidefinite
    invalid_choi = np.array([[1, 0, 0, 0], 
                             [0, -1, 0, 0], 
                             [0, 0, 1, 0], 
                             [0, 0, 0, 1]], dtype=complex)
    
    with pytest.raises(AssertionError):
        QuantumChannel(invalid_choi, representation="choi")

def test_invalid_stinespring_matrix():
    # Not an isometry
    invalid_stinespring = np.array([[1, 0], 
                                    [0, 1], 
                                    [1, 0], 
                                    [0, 2]], dtype=complex)
    
    with pytest.raises(AssertionError):
        QuantumChannel(invalid_stinespring, representation="stinespring")

def test_invalid_kraus_operators():
    # Not trace-preserving
    invalid_kraus = np.array([np.eye(2, dtype=complex), 2 * np.eye(2, dtype=complex)])
    with pytest.raises(AssertionError):
        QuantumChannel(invalid_kraus, representation="kraus")

def test_compose_channels():
    kraus_operators_1 = np.array([np.eye(2, dtype=complex)]) 
    kraus_operators_2 = np.array([np.eye(2, dtype=complex)])  
    
    channel1 = QuantumChannel(kraus_operators_1, representation="kraus")
    channel2 = QuantumChannel(kraus_operators_2, representation="kraus")
    
    composed_channel = channel1.compose(channel2)
    expected_kraus = np.array([np.eye(2, dtype=complex)])  
    
    assert composed_channel.representation == "kraus"
    assert np.allclose(composed_channel.to_kraus(), expected_kraus), "Composed channel should be equivalent to an identity channel."

def test_tensor_channels():
    choi_matrix_1 = np.eye(4, dtype=complex) / 2
    choi_matrix_2 = np.eye(4, dtype=complex) / 2
    
    channel1 = QuantumChannel(choi_matrix_1, representation="choi")
    channel2 = QuantumChannel(choi_matrix_2, representation="choi")
    
    tensor_channel = channel1.tensor(channel2)
    expected_choi = np.eye(16, dtype=complex) / 4
    
    assert tensor_channel.representation == "choi"
    assert np.allclose(tensor_channel.to_choi(), expected_choi), "Tensor product channel should match expected Choi matrix."

def test_call_channel():
    # Amplitude damping
    gamma = 0.3
    kraus_operators = np.array([
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    ])
    channel = QuantumChannel(kraus_operators, representation="kraus")
    # Density matrix of qubit in |1>
    rho = DensityMatrix(np.array([[0, 0], 
                                  [0, 1]], dtype=complex))
    output = channel(rho)
    expected_output = np.array([[gamma, 0],
                                [0, 1 - gamma]], dtype=complex)
    assert isinstance(output, DensityMatrix), "Channel should return a DensityMatrix"
    assert np.allclose(output, expected_output), "Channel is not applied correctly to state"

def test_fidelity_channels():
    # Identity channel represented in Choi form
    choi_matrix_1 = np.array([[1, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]], dtype=complex) / 2
    choi_matrix_2 = np.array([[1, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]], dtype=complex) / 2
    channel1 = QuantumChannel(choi_matrix_1, representation="choi")
    channel2 = QuantumChannel(choi_matrix_2, representation="choi")
    
    fidelity = channel1.fidelity(channel2)
    assert np.isclose(fidelity, 1.0), "Fidelity between two identical channels should be 1."

def test_qubit_reset_channel():
    reset_channel = QubitResetChannel()
    assert reset_channel.representation == "kraus"
    assert reset_channel.shape == (2, 2, 2)

def test_depolarizing_channel():
    noise_channel = DepolarizingChannel(0.1)
    assert noise_channel.representation == "kraus"
    assert noise_channel.shape == (4, 2, 2)

    rho = np.array([[1, 0], 
                    [0, 0]], dtype=complex)
    output = noise_channel(rho)
    expected_output = np.array([[0.93333333, 0.],
                                [0., 0.06666667]], dtype=complex)
    assert isinstance(output, DensityMatrix), "Noise channel should return a numpy array"
    assert np.allclose(output, expected_output), "Noise channel is not applied correctly to state"

def test_pauli_noise_channel():
    p_x, p_y, p_z = 0.2, 0.3, 0.1
    channel = PauliNoiseChannel(p_x, p_y, p_z)
    assert channel.representation == "kraus"
    assert channel.shape == (4, 2, 2)

    rho = np.array([[1, 0], 
                    [0, 0]], dtype=complex)
    output = channel(rho)
    assert isinstance(output, DensityMatrix), "Pauli noise channel should return a DensityMatrix"
    expected_output = np.array([[0.5, 0.], 
                                [0., 0.5]], dtype=complex)
    assert np.allclose(output, expected_output), "Pauli noise channel is not applied correctly to state"

def test_completely_dephasing_channe():
    dephasing_channel = CompletelyDephasingChannel()
    assert dephasing_channel.representation == "kraus"
    assert dephasing_channel.shape == (2, 2, 2)
    # Each Kruas operator is diagonal
    for k in dephasing_channel:
        assert np.allclose(k, np.diag(np.diag(k))), "Kraus operator is not diagonal"

def test_amplitude_damping_channel():
    gamma = 0.2
    amplitude_damping_channel = AmplitudeDampingChannel(gamma)
    
    assert amplitude_damping_channel.representation == "kraus"
    assert amplitude_damping_channel.shape == (2, 2, 2)

def test_phase_flip_channel():
    phase_flip_channel = PhaseFlipChannel(0.2)
    assert phase_flip_channel.representation == "kraus"
    assert phase_flip_channel.shape == (2, 2, 2)
