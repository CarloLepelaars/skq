import pytest
import numpy as np
from skq.gates import HGate, XGate, YGate, ZGate
from skq.encoders import AmplitudeEncoder, GateTransformer


def test_amplitude_encoder():
    encoder = AmplitudeEncoder()
    test_arr = np.array([0.1+2j, -0.6-2j, 1.0+1.3j, 0.-1.5j])

    encoded_arr = encoder.fit_transform(test_arr)
    expected_output = np.array([0.02741012+0.54820244j, -0.16446073-0.54820244j, 0.27410122+0.35633159j, 0.-0.41115183j])

    assert np.iscomplex(encoded_arr).all()
    assert encoded_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(encoded_arr, expected_output)

def test_gate_transformer():
    gate = HGate()
    transformer = GateTransformer(gate)
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

def test_gate_transformer_invalid_input():
    gate = HGate()
    transformer = GateTransformer(gate)

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
