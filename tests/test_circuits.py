import pytest
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline

from skq.circuits import BellStates
from skq.transformers import MeasurementTransformer

def test_bell_state_output_type():
    bell_states = BellStates()
    for i in range(1, 5):
        pipeline = bell_states.get_bell_state(i)
        assert isinstance(pipeline, Pipeline), f"Configuration {i} did not return a Pipeline"

def test_invalid_bell_state_configuration():
    bell_states = BellStates()
    with pytest.raises(AssertionError):
        bell_states.get_bell_state(0)
    with pytest.raises(AssertionError):
        bell_states.get_bell_state(5)

def test_bell_state_output():
    repeat = 10000
    # Bell State 1
    bell_states = BellStates()
    pipe = make_pipeline(bell_states.get_bell_state(configuration=1), MeasurementTransformer(repeat=repeat))
    X = np.array([[1, 0, 0, 0]], dtype=complex)

    output = pipe.transform(X)
    assert len(output) == repeat
    count_00 = np.sum(np.all(output == [0, 0], axis=1))
    count_11 = np.sum(np.all(output == [1, 1], axis=1))
    assert count_00 == pytest.approx(repeat//2, rel=0.2)
    assert count_11 == pytest.approx(repeat//2, rel=0.2)

    # Bell State 2: |Φ-⟩ = |00> - |11> / sqrt(2)
    pipe = make_pipeline(bell_states.get_bell_state(configuration=2), MeasurementTransformer(repeat=repeat))
    output = pipe.transform(X)
    assert len(output) == repeat
    count_00 = np.sum(np.all(output == [0, 0], axis=1))
    count_11 = np.sum(np.all(output == [1, 1], axis=1))
    assert count_00 == pytest.approx(repeat//2, rel=0.2)
    assert count_11 == pytest.approx(repeat//2, rel=0.2)

    # Bell State 3: |Ψ+⟩ = |01> + |10> / sqrt(2)
    pipe = make_pipeline(bell_states.get_bell_state(configuration=3), MeasurementTransformer(repeat=repeat))
    output = pipe.transform(X)
    assert len(output) == repeat
    count_01 = np.sum(np.all(output == [0, 1], axis=1))
    count_10 = np.sum(np.all(output == [1, 0], axis=1))
    assert count_01 == pytest.approx(repeat//2, rel=0.2)
    assert count_10 == pytest.approx(repeat//2, rel=0.2)

    # Bell State 4: |Ψ-⟩ = |01> - |10> / sqrt(2)
    pipe = make_pipeline(bell_states.get_bell_state(configuration=4), MeasurementTransformer(repeat=repeat))
    output = pipe.transform(X)
    assert len(output) == repeat
    count_01 = np.sum(np.all(output == [0, 1], axis=1))
    count_10 = np.sum(np.all(output == [1, 0], axis=1))
    assert count_01 == pytest.approx(repeat//2, rel=0.2)
    assert count_10 == pytest.approx(repeat//2, rel=0.2)
