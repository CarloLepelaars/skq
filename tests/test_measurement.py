import numpy as np
import pytest
from skq.transformers.measurement import MeasurementTransformer

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
    
def test_invalid_state_vector():
    transformer = MeasurementTransformer(repeat=1)
    # Non-normalized state vector
    X = np.array([[1, 1, 0, 0]], dtype=complex)

    with pytest.raises(ValueError, match="Input state vectors must be normalized"):
        transformer.fit(X)
        transformer.transform(X)
