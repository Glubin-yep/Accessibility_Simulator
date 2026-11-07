import numpy as np
import pytest

from observer import LOST_FIELD_OBSERVER, NORMAL_OBSERVER, ObserverCSF


@pytest.mark.parametrize("logmar, logcs", [(0, 1.95), (0.3, 1.5), (1.0, 0.9), (1.6, 0.3)])
def test_csf_passes_through_clinical_points(logmar, logcs):
    csf = ObserverCSF(logmar, logcs)
    assert csf.log_gamma_max == pytest.approx(logcs + 0.35)
    assert csf.cutoff_cpd == pytest.approx(30 * 10 ** -logmar)
    assert csf.log_sensitivity(csf.cutoff_cpd) == pytest.approx(0.0, abs=1e-9)


def test_normal_observer_peak():
    assert NORMAL_OBSERVER.fmax_cpd == pytest.approx(2.52, abs=0.01)
    assert NORMAL_OBSERVER.log_sensitivity(NORMAL_OBSERVER.fmax_cpd) == pytest.approx(2.30)


def test_low_frequencies_are_truncated():
    csf = NORMAL_OBSERVER
    assert csf.log_sensitivity(0.05) == pytest.approx(csf.log_gamma_max - 0.5)


def test_sensitivity_drops_above_cutoff():
    assert NORMAL_OBSERVER.log_sensitivity(40) < 0


def test_array_input_matches_scalar_and_returns_array():
    freqs = np.array([0.05, 1.0, 2.52, 10.0, 40.0])
    result = NORMAL_OBSERVER.log_sensitivity(freqs)
    assert isinstance(result, np.ndarray)
    assert result.shape == freqs.shape
    for f, value in zip(freqs, result):
        assert value == pytest.approx(NORMAL_OBSERVER.log_sensitivity(float(f)))


def test_scalar_input_returns_float():
    assert isinstance(NORMAL_OBSERVER.log_sensitivity(3.0), float)


def test_lost_field_observer_parameters():
    assert LOST_FIELD_OBSERVER.log_gamma_max == pytest.approx(0.65)
    assert LOST_FIELD_OBSERVER.cutoff_cpd == pytest.approx(30 * 10 ** -1.6)


@pytest.mark.parametrize("f", [0.0, -1.0])
def test_non_positive_frequency_raises(f):
    with pytest.raises(ValueError):
        NORMAL_OBSERVER.log_sensitivity(f)


@pytest.mark.parametrize("logmar, logcs", [(0, 0), (0, -1), (-0.5, 1.95)])
def test_invalid_observer_raises(logmar, logcs):
    with pytest.raises(ValueError):
        ObserverCSF(logmar, logcs)
