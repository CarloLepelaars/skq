from skq.encoders import BaseEncoder, BasisEncoder

def test_base_encoder():
    encoder = BaseEncoder()
    assert encoder is not None
