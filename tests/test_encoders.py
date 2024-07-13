import numpy as np
from skq.encoders import AmplitudeEncoder


def test_amplitude_encoder():
    encoder = AmplitudeEncoder()
    test_arr = np.array([0.1+2j, -0.6-2j, 1.0+1.3j, 0.-1.5j])

    encoded_arr = encoder.fit_transform(test_arr)
    expected_output = np.array([0.02741012+0.54820244j, -0.16446073-0.54820244j, 0.27410122+0.35633159j, 0.-0.41115183j])

    assert np.iscomplex(encoded_arr).all()
    assert encoded_arr.shape == expected_output.shape
    np.testing.assert_almost_equal(encoded_arr, expected_output)
