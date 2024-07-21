import pytest
import numpy as np
from skq.transformers import HTransformer, TTransformer, CXTransformer, CHTransformer, CCXTransformer, CSwapTransformer, MeasurementTransformer


def test_single_qubit_transformer():
    # Hadamard gate
    transformer = HTransformer(qubits=[0])
    # Single input
    single_test_arr = np.array([[1, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0.70710678+0.j, 0.70710678+0.j]])

    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Multi inputs
    multi_test_arr = np.array([[1, 0], [0, 1]], dtype=complex)
    transformed_arr = transformer.fit_transform(multi_test_arr)
    expected_output = np.array([[0.70710678+0.j, 0.70710678+0.j], [0.70710678+0.j, -0.70710678+0.j]])

    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Phase gate (T Gate)
    transformer = TTransformer(qubits=[0])

    # Single input
    single_test_arr = np.array([[1, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[1+0.j, 0+0.j]])

    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Multi inputs
    multi_test_arr = np.array([[1, 0], [0, 1]], dtype=complex)
    transformed_arr = transformer.fit_transform(multi_test_arr)
    expected_output = np.array([[1+0.j, 0+0.j], [0+0.j, 1/np.sqrt(2)+1j/np.sqrt(2)]])

    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

def test_single_qubit_transformer_invalid_input():
    transformer = HTransformer(qubits=[0])

    # Invalid 1D shape input
    test_arr = np.array([1, 0], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)

    # Non normalized input
    test_arr = np.array([[2, 1], [2, 1]], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)

    # 1 element in row
    test_arr = np.array([[1], [0]], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)

    # More than two elements in row
    test_arr = np.array([[1, 0, 1], [0, 1, 0]], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)

def test_multi_qubit_transformer():
    # Test Controlled NOT
    transformer = CXTransformer(qubits=[0, 1])

    # Two qubit state |01⟩ which should not change after applying CX gate
    single_test_arr = np.array([[0, 1, 0, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 1, 0, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Two qubit state |11⟩ which should change to |10⟩ after applying CX gate
    single_test_arr = np.array([[0, 0, 0, 1]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 0, 1, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Test Controlled Hadamard
    transformer = CHTransformer(qubits=[0, 1])

    # Two qubit state |01⟩ which should not change after applying CH gate
    single_test_arr = np.array([[0, 1, 0, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 1, 0, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Two qubit state |11⟩ which should change to |1-⟩ after applying CH gate
    single_test_arr = np.array([[0, 0, 0, 1]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Two qubit state |10⟩ which should change to |1+⟩ after applying CH gate
    single_test_arr = np.array([[0, 0, 1, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

def test_multi_qubit_transformer_toffoli_fredkin():
    # Test Toffoli Gate
    transformer = CCXTransformer(qubits=[0, 1, 2])

    # Three qubit state |000⟩ which should not change after applying Toffoli gate
    single_test_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Three qubit state |001⟩ which should not change after applying Toffoli gate
    single_test_arr = np.array([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Three qubit state |110> which should change to |111> after applying Toffoli gate
    single_test_arr = np.array([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Test Fredkin Gate
    transformer = CSwapTransformer(qubits=[0, 1, 2])

    # Three qubit state |000⟩ which should not change after applying Fredkin gate
    single_test_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

    # Three qubit state |110> which should change to |101> after applying Fredkin gate
    single_test_arr = np.array([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)
    transformed_arr = transformer.fit_transform(single_test_arr)
    expected_output = np.array([[0, 0, 0, 0, 0, 1, 0, 0]], dtype=complex)
    assert transformed_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(transformed_arr, expected_output)

def test_multi_qubit_transformer_invalid_input():
    transformer = CXTransformer(qubits=[0, 1])

    # Invalid 1D shape input
    test_arr = np.array([1, 0], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)
    
    # Non normalized input
    test_arr = np.array([[2, 1, 0, 0], [2, 1, 0, 0]], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)

    # 1 element in row
    test_arr = np.array([[1], [0]], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)
    
    # More than two elements in row
    test_arr = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=complex)
    with pytest.raises(ValueError):
        transformer.fit_transform(test_arr)

    # Invalid number of qubits
    with pytest.raises(AssertionError):
        transformer = CXTransformer(qubits=[0])

def test_measurement_transformer_single_measurement():
    # Create a MeasurementTransformer instance with 1 measurement
    transformer = MeasurementTransformer(repeat=1)

    # Define a 2-qubit state |00>
    X = np.array([[1, 0, 0, 0]], dtype=complex)
    transformer.fit(X)
    measurements = transformer.transform(X)

    assert measurements.shape == (1, 2)
    assert np.all(measurements == [0, 0])

def test_measurement_transformer_multiple_measurements():
    # Create a MeasurementTransformer instance with 10 measurements
    transformer = MeasurementTransformer(repeat=10)

    # Define a 2-qubit state |00>
    X = np.array([[1, 0, 0, 0]], dtype=complex)
    transformer.fit(X)
    measurements = transformer.transform(X)

    assert measurements.shape == (10, 2)
    assert np.all(measurements == [0, 0])

def test_measurement_transformer_bell_state():
    # Create a MeasurementTransformer instance with 1000 measurements
    transformer = MeasurementTransformer(repeat=1000)

    # Define a Bell state |Φ+> = (|00> + |11>) / √2
    bell_state = np.array([[1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]], dtype=complex)
    transformer.fit(bell_state)
    measurements = transformer.transform(bell_state)

    assert measurements.shape == (1000, 2)

    # Check the distribution of measurement results
    unique, counts = np.unique(measurements, axis=0, return_counts=True)
    counts_dict = dict(zip(map(tuple, unique), counts))

    # Check if the measurements are roughly 50% |00> and 50% |11>
    assert counts_dict.get((0, 0), 0) > 400 
    assert counts_dict.get((1, 1), 0) > 400 

def test_measurement_transformer_complex_state():
    transformer = MeasurementTransformer(repeat=1000)

    # Define a complex state ψ = 1/2 (|00> + |01> + |10> + |11>)
    complex_state = np.array([[1/2, 1/2, 1/2, 1/2]], dtype=complex)
    transformer.fit(complex_state)
    measurements = transformer.transform(complex_state)

    assert measurements.shape == (1000, 2)
    assert np.all((measurements == 0) | (measurements == 1))

    # Check the distribution of measurement results
    unique, counts = np.unique(measurements, axis=0, return_counts=True)
    counts_dict = dict(zip(map(tuple, unique), counts))

    # Check if the measurements are roughly 25% |00>, 25% |01>, 25% |10>, and 25% |11>
    assert counts_dict.get((0, 0), 0) > 200 
    assert counts_dict.get((0, 1), 0) > 200 
    assert counts_dict.get((1, 0), 0) > 200 
    assert counts_dict.get((1, 1), 0) > 200
    
def test_measurement_invalid_state_vector():
    transformer = MeasurementTransformer(repeat=1)
    # Non-normalized state vector
    X = np.array([[1, 1, 0, 0]], dtype=complex)

    with pytest.raises(ValueError, match="Input state vectors must be normalized"):
        transformer.fit(X)
        transformer.transform(X)
        