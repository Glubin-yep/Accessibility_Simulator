import cv2
import numpy as np
import pytest

from color_utils import gamma_decode, gamma_encode
from observer import (
    LOST_FIELD_OBSERVER,
    NORMAL_OBSERVER,
    PYRAMID_F0,
    ObserverCSF,
    apply_observer,
)
from viewing import ViewingConditions


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


@pytest.mark.parametrize("f", [0.0, -1.0, float("nan")])
def test_non_positive_frequency_raises(f):
    with pytest.raises(ValueError):
        NORMAL_OBSERVER.log_sensitivity(f)


@pytest.mark.parametrize("logmar, logcs", [(0, 0), (0, -1), (-0.5, 1.95), (0, float("nan")), (float("nan"), 1.95), (float("inf"), 1.95), (0, float("inf"))])
def test_invalid_observer_raises(logmar, logcs):
    with pytest.raises(ValueError):
        ObserverCSF(logmar, logcs)


# --- метод Peli: apply_observer ---


def grating_image(contrast, cpd, viewing, height=256, width=512):
    """Вертикальна синусоїдальна ґратка навколо лінійної яскравості 0.5 (uint8 BGR)."""
    f_px = viewing.cpd_to_cycles_per_px(cpd)
    x = np.arange(width)
    row = 0.5 * (1.0 + contrast * np.sin(2.0 * np.pi * f_px * x))
    linear = np.broadcast_to(row, (height, width))
    return np.ascontiguousarray(np.repeat(gamma_encode(linear)[:, :, None], 3, axis=2))


def grating_amplitude(image_bgr, cpd, viewing):
    """Амплітуда вертикальної синусоїди частоти cpd у лінійній яскравості (проєкція на sin і cos)."""
    row = gamma_decode(image_bgr[:, :, 0]).mean(axis=0).astype(float)
    x = np.arange(row.size)
    phase = 2.0 * np.pi * viewing.cpd_to_cycles_per_px(cpd) * x
    centred = row - row.mean()
    return float(np.hypot(2.0 * np.mean(centred * np.sin(phase)), 2.0 * np.mean(centred * np.cos(phase))))


def _level0_gain(f_px, width=2048, height=16, margin=256):
    """Підсилення рівня 0 піраміди Лапласа для синусоїди частоти f_px (цикл/піксель)."""
    x = np.arange(width)
    img = np.broadcast_to(np.cos(2.0 * np.pi * f_px * x), (height, width)).astype(np.float32)
    band = img - cv2.pyrUp(cv2.pyrDown(img), dstsize=(width, height))
    xs = x[margin:-margin]
    design = np.stack([np.cos(2.0 * np.pi * f_px * xs), np.sin(2.0 * np.pi * f_px * xs)], axis=1)
    coef, *_ = np.linalg.lstsq(design, band[height // 2, margin:-margin].astype(float), rcond=None)
    return float(np.hypot(*coef))


def measured_f0():
    """F0 = sqrt(f_half · 0.5): геометричне середнє частоти половинного підсилення і Найквіста."""
    lo, hi = 0.02, 0.5
    assert _level0_gain(lo) < 0.5 < _level0_gain(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _level0_gain(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    f_half = 0.5 * (lo + hi)
    return float(np.sqrt(f_half * 0.5))


def test_pyramid_f0_matches_grating_measurement():
    assert measured_f0() == pytest.approx(PYRAMID_F0, abs=0.01)


def test_pyramid_f0_constant_value():
    assert PYRAMID_F0 == 0.2565


def test_normal_observer_barely_changes_high_contrast_text(text_image):
    out = apply_observer(text_image, NORMAL_OBSERVER, ViewingConditions())
    assert out.dtype == np.uint8
    assert out.shape == text_image.shape
    assert np.abs(out.astype(int) - text_image.astype(int)).mean() < 2


def test_low_cs_removes_faint_grating_but_keeps_strong():
    v = ViewingConditions()
    poor = ObserverCSF(0.0, 0.45)
    # 6 цикл/град ≈ пік CSF цього спостерігача: поріг ≈ 0.16 (log S ≈ 0.79)
    # амплітуда самої ґратки, а не std: вихід без ґратки -- рівне поле біля межі округлення sRGB
    faint = grating_image(contrast=0.02, cpd=6, viewing=v)
    strong = grating_image(contrast=0.5, cpd=6, viewing=v)
    faint_out = grating_amplitude(apply_observer(faint, poor, v), 6, v)
    strong_out = grating_amplitude(apply_observer(strong, poor, v), 6, v)
    assert faint_out < grating_amplitude(faint, 6, v) * 0.2
    assert strong_out > grating_amplitude(strong, 6, v) * 0.7


def test_mean_luminance_is_preserved(text_image):
    out = apply_observer(text_image, ObserverCSF(1.0, 0.9), ViewingConditions())
    np.testing.assert_allclose(
        out.reshape(-1, 3).mean(0), text_image.reshape(-1, 3).mean(0), atol=2
    )


def test_worse_acuity_does_not_add_high_frequencies(text_image):
    v = ViewingConditions()

    def hf(im):
        return np.abs(cv2.Laplacian(im.astype(np.float32), cv2.CV_32F)).mean()

    a = hf(apply_observer(text_image, ObserverCSF(0.3, 1.95), v))
    b = hf(apply_observer(text_image, ObserverCSF(0.7, 1.95), v))
    assert b <= a


def test_tiny_image_keeps_shape():
    img = np.full((5, 7, 3), 128, np.uint8)
    assert apply_observer(img, NORMAL_OBSERVER, ViewingConditions()).shape == (5, 7, 3)


def test_black_image_has_no_nan():
    img = np.zeros((64, 64, 3), np.uint8)
    np.testing.assert_array_equal(apply_observer(img, NORMAL_OBSERVER, ViewingConditions()), img)


def test_far_viewing_keeps_mean(text_image):
    out = apply_observer(text_image, NORMAL_OBSERVER, ViewingConditions(distance_cm=500))
    np.testing.assert_allclose(out.mean(), text_image.mean(), atol=2)


def test_input_not_modified(text_image):
    before = text_image.copy()
    apply_observer(text_image, ObserverCSF(1.0, 0.9), ViewingConditions())
    np.testing.assert_array_equal(text_image, before)


def test_image_without_bands_is_returned_unchanged():
    # менша сторона < 8 px -> смуг немає, лишається точний гамма-цикл sRGB -> лінійний -> sRGB
    img = np.random.default_rng(0).integers(0, 256, (7, 4096, 3), dtype=np.uint8)
    np.testing.assert_array_equal(apply_observer(img, NORMAL_OBSERVER, ViewingConditions()), img)
