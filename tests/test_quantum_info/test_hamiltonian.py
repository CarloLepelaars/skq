import numpy as np
from scipy.linalg import expm

from skq.quantum_info import Hamiltonian, IsingHamiltonian, HeisenbergHamiltonian


def test_hamiltonian_properties():
    h_matrix = np.array([
        [0, 1],
        [1, 0]
    ])
    H = Hamiltonian(h_matrix)

    assert H.is_2d(), "Hamiltonian should be 2D."
    assert H.is_at_least_nxn(2), "Hamiltonian should be at least 2x2."
    assert H.is_hermitian(), "Hamiltonian should be Hermitian."
    assert H.num_qubits() == 1, "Hamiltonian should correspond to 1 qubit."

def test_hamiltonian_eigenvalues():
    h_matrix = np.array([
        [0, 1],
        [1, 0]
    ])
    H = Hamiltonian(h_matrix)

    expected_eigenvalues = [-1, 1]
    np.testing.assert_array_almost_equal(H.eigenvalues(), expected_eigenvalues, decimal=5)

def test_hamiltonian_ground_state():
    h_matrix = np.array([
        [0, 1],
        [1, 0]
    ])
    H = Hamiltonian(h_matrix)

    ground_state_energy = H.ground_state_energy()
    ground_state = H.ground_state()

    expected_ground_state_energy = -1
    expected_ground_state = np.array([1, -1]) / np.sqrt(2)

    assert np.isclose(ground_state_energy, expected_ground_state_energy, atol=1e-5), \
        "Ground state energy should be -1."
    np.testing.assert_array_almost_equal(np.abs(ground_state), np.abs(expected_ground_state), decimal=5)

def test_hamiltonian_time_evolution_operator():
    h_matrix = np.array([
        [0, 1],
        [1, 0]
    ])
    H = Hamiltonian(h_matrix)

    t = 1.0
    U = H.time_evolution_operator(t)
    expected_U = expm(-1j * H * t)

    np.testing.assert_array_almost_equal(U, expected_U, decimal=5)

def test_hamiltonian_convert_endianness():
    h_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 2, 0],
        [0, 2, -1, 0],
        [0, 0, 0, 1]
    ])
    H = Hamiltonian(h_matrix)
    H_converted = H.convert_endianness()

    np.testing.assert_array_almost_equal(H, H_converted, decimal=5)

def test_hamiltonian_qiskit_conversion():
    h_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 2, 0],
        [0, 2, -1, 0],
        [0, 0, 0, 1]
    ])
    H = Hamiltonian(h_matrix)
    qiskit_op = H.to_qiskit()
    H_from_qiskit = Hamiltonian.from_qiskit(qiskit_op)

    np.testing.assert_array_almost_equal(H, H_from_qiskit, decimal=5)

def test_hamiltonian_pennylane_conversion():
    h_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 2, 0],
        [0, 2, -1, 0],
        [0, 0, 0, 1]
    ])
    H = Hamiltonian(h_matrix)
    wires = [0, 1]
    pennylane_h = H.to_pennylane(wires=wires)
    H_from_pennylane = Hamiltonian.from_pennylane(pennylane_h)

    np.testing.assert_array_almost_equal(H, H_from_pennylane, decimal=5)

def test_ising_hamiltonian():
    H = IsingHamiltonian(num_qubits=2, J=1.0, h=0.5)
    
    expected_h_matrix = np.array([
        [-1, -0.5, -0.5, 0.],
        [-0.5, 1., 0., -0.5],
        [-0.5, 0., 1., -0.5],
        [0., -0.5, -0.5, -1.]
    ])
    np.testing.assert_array_almost_equal(H, expected_h_matrix)

def test_heisenberg_hamiltonian():
    H = HeisenbergHamiltonian(num_qubits=2, J=1.0)
    
    expected_h_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 2.0, 0.0],
        [0.0, 2.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    np.testing.assert_array_almost_equal(H, expected_h_matrix, decimal=5)

def test_ising_hamiltonian_properties():
    H = IsingHamiltonian(num_qubits=2, J=1.0, h=0.5)
    assert H.is_2d(), "Hamiltonian should be 2D."
    assert H.is_at_least_nxn(4), "Hamiltonian should be at least 4x4."
    assert H.is_hermitian(), "Hamiltonian should be Hermitian."
    assert H.num_qubits() == 2, "Hamiltonian should correspond to 2 qubits."

def test_heisenberg_hamiltonian_properties():
    H = HeisenbergHamiltonian(num_qubits=2, J=1.0)

    assert H.is_2d(), "Hamiltonian should be 2D."
    assert H.is_at_least_nxn(4), "Hamiltonian should be at least 4x4."
    assert H.is_hermitian(), "Hamiltonian should be Hermitian."
    assert H.num_qubits() == 2, "Hamiltonian should correspond to 2 qubits."
