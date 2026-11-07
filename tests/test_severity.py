import numpy as np
import pytest

from observer import NORMAL_OBSERVER, apply_observer
from severity import MODELS, preset_name
from viewing import ViewingConditions


@pytest.mark.parametrize("key", list(MODELS))
def test_zero_severity_equals_normal_observer(key, text_image):
    v = ViewingConditions()
    np.testing.assert_array_equal(MODELS[key].simulate(text_image, 0.0, v), apply_observer(text_image, NORMAL_OBSERVER, v))


@pytest.mark.parametrize("key", list(MODELS))
def test_shape_dtype_and_input_untouched(key, text_image):
    before = text_image.copy()
    out = MODELS[key].simulate(text_image, 0.6, ViewingConditions())
    assert out.shape == text_image.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(text_image, before)


@pytest.mark.parametrize("key, s, label", [
    ("acuity", 0.3, "20/40"), ("acuity", 0.7, "20/100"), ("acuity", 1.0, "20/200"),
    ("cataract", 0.3, "20/40"), ("contrast", 0.5, "log CS 1.20"),
    ("scotoma", 0.5, "5°"), ("tunnel", 0.714, "10°"), ("tunnel", 0.0, "30°"),
    ("protanomaly", 0.6, "Machado 0.6"), ("metamorphopsia", 0.25, "0.5°"),
])
def test_clinical_labels(key, s, label):
    m = MODELS[key]
    assert m.format_clinical(m.to_clinical(s)) == label


def test_model_order():
    assert list(MODELS) == ["acuity", "cataract", "contrast", "scotoma", "tunnel",
                            "protanomaly", "deuteranomaly", "tritanomaly", "metamorphopsia"]


def test_preset_name():
    assert preset_name(MODELS["tunnel"], 0.714) == "Тунельний зір (10°)"


@pytest.mark.parametrize("s", [-0.01, 1.01])
def test_severity_out_of_range_raises(s, text_image):
    with pytest.raises(ValueError):
        MODELS["acuity"].simulate(text_image, s, ViewingConditions())


def test_severe_models_change_image(text_image):
    v = ViewingConditions()
    base = apply_observer(text_image, NORMAL_OBSERVER, v)
    for key in ("cataract", "contrast", "scotoma", "protanomaly", "metamorphopsia"):
        assert not np.array_equal(MODELS[key].simulate(text_image, 1.0, v), base), key


@pytest.mark.parametrize("key", ["tunnel", "scotoma"])
def test_zero_severity_exact_at_default_viewport(key):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (1440, 2560, 3), dtype=np.uint8)
    v = ViewingConditions()
    np.testing.assert_array_equal(MODELS[key].simulate(img, 0.0, v), apply_observer(img, NORMAL_OBSERVER, v))


@pytest.mark.parametrize("v, text", [(0.04, "0.04°"), (0.5, "0.5°"), (2.7, "2.7°"), (10, "10°")])
def test_format_degrees(v, text):
    from severity import format_degrees
    assert format_degrees(v) == text
