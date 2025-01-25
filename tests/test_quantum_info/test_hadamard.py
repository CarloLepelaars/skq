import pytest
import numpy as np

from skq.quantum_info import HadamardMatrix, generate_hadamard_matrix


def test_hadamard_basic_properties():
    # Define a 4x4 Hadamard matrix
    hadamard_matrix = HadamardMatrix(np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))

    assert isinstance(hadamard_matrix, HadamardMatrix)
    assert hadamard_matrix.is_binary()
    assert hadamard_matrix.is_orthogonal()
    assert hadamard_matrix.is_hadamard_order()
    assert np.isclose(hadamard_matrix.determinant(), 16.0)
    assert hadamard_matrix.num_levels() == 4


def test_hadamard_equivalence():
    hadamard_matrix = HadamardMatrix(np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))

    permuted_hadamard_matrix = HadamardMatrix(np.array([[1, 1, -1, -1], [1, -1, -1, 1], [1, 1, 1, 1], [1, -1, 1, -1]]))

    assert hadamard_matrix.equivalence(permuted_hadamard_matrix)


def test_hadamard_spectral_norm():
    hadamard_matrix = HadamardMatrix(np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))
    assert np.isclose(hadamard_matrix.spectral_norm(), 2.0)


def test_hadamard_permutations_and_sign_flips():
    hadamard_matrix = HadamardMatrix(np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))

    permutations_flips = hadamard_matrix.permutations_and_sign_flips()
    assert len(permutations_flips) > 0


def test_invalid_hadamard_matrix():
    # Invalid entry (0)
    with np.testing.assert_raises(AssertionError):
        HadamardMatrix(np.array([[1, 1, 1, 1], [1, 0, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))
    # Invalid order 3
    with np.testing.assert_raises(AssertionError):
        HadamardMatrix(np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]]))


def test_hadamard_identity():
    # Permutation identity
    hadamard_matrix = HadamardMatrix(np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))

    identity_matrix = HadamardMatrix(np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]))

    assert hadamard_matrix.equivalence(identity_matrix)


def test_generate_hadamard_matrix():
    # Order 2
    H2 = generate_hadamard_matrix(2)
    assert isinstance(H2, HadamardMatrix)
    assert H2.shape == (2, 2)
    assert H2.is_binary()
    assert H2.is_orthogonal()

    expected_H2 = np.array([[1, 1], [1, -1]])
    np.testing.assert_array_equal(H2, expected_H2)

    # Order 4
    H4 = generate_hadamard_matrix(4)
    assert isinstance(H4, HadamardMatrix)
    assert H4.shape == (4, 4)
    assert H4.is_binary()
    assert H4.is_orthogonal()

    expected_H4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    np.testing.assert_array_equal(H4, expected_H4)

    # Order 8
    H8 = generate_hadamard_matrix(8)
    assert isinstance(H8, HadamardMatrix)
    assert H8.shape == (8, 8)
    assert H8.is_binary()
    assert H8.is_orthogonal()

    expected_H8 = np.block([[expected_H4, expected_H4], [expected_H4, -expected_H4]])
    np.testing.assert_array_equal(H8, expected_H8)

    H16 = generate_hadamard_matrix(16)
    assert isinstance(H16, HadamardMatrix)
    assert H16.shape == (16, 16)
    assert H16.is_binary()
    assert H16.is_orthogonal()
    assert np.isclose(H16.spectral_norm(), 4.0)

    # Invalid order (not a power of 2)
    with pytest.raises(AssertionError):
        generate_hadamard_matrix(3)
