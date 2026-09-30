import numpy as np

from src.codec_views import parent_shared_weights
from src.fourier_detector import logistic_probability, publisher_residual
from src.fusion import finalize_prediction, semantic_context
from src.selection import ads_selection_error, robust_component_error


def test_cps_is_copied_bit_exact():
    new = np.array([0.8, 0.9, 0.6, 0.1, 0.2], dtype=np.float64)
    old = np.array([0.4, 0.3, 0.5, 0.987654321, 0.123456789], dtype=np.float64)
    result = finalize_prediction(new, old, voice_v2_blend=0.5)
    assert np.array_equal(result[3:], old[3:])
    assert result[1] == 0.6


def test_semantic_context_is_bounded():
    singing, speech = semantic_context(
        {"pann_speech_logit": 2.0, "pann_singing_logit": -1.0}
    )
    assert 0.0 < singing < 1.0
    assert 0.0 < speech < 1.0
    assert speech > singing


def test_parent_views_share_unit_weight():
    parents = ["a", "a", "a", "a", "b", "b"]
    weights = parent_shared_weights(parents)
    assert np.isclose(weights[:4].sum(), 1.0)
    assert np.isclose(weights[4:].sum(), 1.0)


def test_robust_error_matches_ads_weights():
    file_error = robust_component_error([0.1, 0.2], [0.1, 0.3])
    errors = {"file": file_error, "voice": 0.2, "music": 0.3}
    assert np.isclose(ads_selection_error(errors), 0.5 * file_error + 0.13)


def test_fourier_shape_and_probability():
    profile = np.linspace(-40.0, -20.0, 3585)
    features = publisher_residual(profile)
    probability = logistic_probability(features, np.zeros(3585), 0.0)
    assert features.shape == (3585,)
    assert np.isfinite(features).all()
    assert probability == 0.5

