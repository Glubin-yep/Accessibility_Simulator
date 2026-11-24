import math

import numpy as np
import pytest

import config
import main
from combination import (
    MIN_LOGCS,
    STAGE_ORDER,
    combination_name,
    combine_csf,
    parse_combination,
    simulate_combination,
    validate_components,
)
from observer import LOST_FIELD_OBSERVER, NORMAL_OBSERVER, ObserverCSF, apply_observer
from severity import MODELS
from viewing import ViewingConditions


def test_stage_order_covers_all_models():
    assert sorted(STAGE_ORDER) == sorted(MODELS)


@pytest.mark.parametrize("key", list(MODELS))
@pytest.mark.parametrize("s", [0.0, 0.4, 1.0])
def test_single_component_matches_model(key, s, text_image):
    v = ViewingConditions()
    np.testing.assert_array_equal(
        simulate_combination(text_image, [(key, s)], v), MODELS[key].simulate(text_image, s, v)
    )


def test_all_zero_equals_normal_observer(text_image):
    v = ViewingConditions()
    out = simulate_combination(text_image, [(k, 0.0) for k in MODELS], v)
    np.testing.assert_array_equal(out, apply_observer(text_image, NORMAL_OBSERVER, v))


def test_input_order_does_not_matter(text_image):
    v = ViewingConditions()
    a = simulate_combination(text_image, [("acuity", 0.3), ("protanomaly", 1.0), ("cataract", 0.5)], v)
    b = simulate_combination(text_image, [("cataract", 0.5), ("acuity", 0.3), ("protanomaly", 1.0)], v)
    np.testing.assert_array_equal(a, b)


def test_yellowing_precedes_cones(text_image):
    """Пожовтіння кришталика має йти до колбочок: результат = Machado(катаракта(x))."""
    v = ViewingConditions()
    from simulations import cataract_optics, simulate_anomalous_trichromacy_machado

    expected_stimulus = simulate_anomalous_trichromacy_machado(
        cataract_optics(text_image, 1 - 0.3 * 0.5, 1 - 0.08 * 0.5, 0.3 * 0.5), "deuteranomaly", 1.0
    )
    expected = apply_observer(expected_stimulus, ObserverCSF(0.5, 1.95 - 1.05 * 0.5), v)
    out = simulate_combination(text_image, [("deuteranomaly", 1.0), ("cataract", 0.5)], v)
    np.testing.assert_array_equal(out, expected)


def test_combination_is_worse_than_each_part(text_image):
    v = ViewingConditions()
    base = apply_observer(text_image, NORMAL_OBSERVER, v).astype(float)
    parts = [("acuity", 0.6), ("contrast", 0.6)]
    combo = simulate_combination(text_image, parts, v).astype(float)
    combo_err = np.abs(combo - base).mean()
    for key, s in parts:
        single_err = np.abs(MODELS[key].simulate(text_image, s, v).astype(float) - base).mean()
        assert combo_err > single_err, key


def test_combine_csf_rules():
    assert combine_csf([]) == NORMAL_OBSERVER
    assert combine_csf([NORMAL_OBSERVER]) == NORMAL_OBSERVER
    one = ObserverCSF(0.5, 1.5)
    assert combine_csf([one, NORMAL_OBSERVER]) == one
    # Квадратура MAR: два по 20/100 (MAR ≈ 5) -> MAR = sqrt(1 + 2·(25 − 1)) = 7.
    c = combine_csf([ObserverCSF(math.log10(5), 1.95), ObserverCSF(math.log10(5), 1.95)])
    assert c.logmar == pytest.approx(math.log10(7))
    assert c.logcs == pytest.approx(1.95)
    # Дефіцити log CS додаються: 0.5 + 0.3 -> 1.15.
    c = combine_csf([ObserverCSF(0.0, 1.45), ObserverCSF(0.0, 1.65)])
    assert c.logmar == pytest.approx(0.0)
    assert c.logcs == pytest.approx(1.15)
    # Нижня межа.
    c = combine_csf([ObserverCSF(0.0, 0.45), ObserverCSF(0.0, 0.45)])
    assert c.logcs == MIN_LOGCS
    # Втрачене поле з додатковою вадою не стає кращим.
    c = combine_csf([LOST_FIELD_OBSERVER, ObserverCSF(0.3, 1.95)])
    assert c.logmar > LOST_FIELD_OBSERVER.logmar
    assert c.logcs == pytest.approx(LOST_FIELD_OBSERVER.logcs)


def test_two_field_losses_union():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (1440, 2560, 3), dtype=np.uint8)
    v = ViewingConditions()
    both = simulate_combination(img, [("scotoma", 0.5), ("tunnel", 0.5)], v)
    lost = apply_observer(img, LOST_FIELD_OBSERVER, v)
    h, w = img.shape[:2]
    # Центр втрачено скотомою, кут -- тунельним зором.
    np.testing.assert_array_equal(both[h // 2, w // 2], lost[h // 2, w // 2])
    np.testing.assert_array_equal(both[0, 0], lost[0, 0])


@pytest.mark.parametrize(
    "components",
    [[], [("unknown", 0.5)], [("acuity", 1.5)], [("acuity", -0.1)], [("acuity", float("nan"))],
     [("acuity", 0.2), ("acuity", 0.3)]],
)
def test_validate_rejects(components):
    with pytest.raises(ValueError):
        validate_components(components)


def test_parse_and_name():
    parsed = parse_combination(" acuity=0.3, cataract=0.3 ,protanomaly=1 ")
    assert parsed == [("cataract", 0.3), ("protanomaly", 1.0), ("acuity", 0.3)]
    assert combination_name(parsed) == (
        "Катаракта (20/40) + Протаномалія (Machado 1) + Гострота зору (20/40)"
    )


@pytest.mark.parametrize("text", ["acuity", "acuity=x", "acuity=2", ""])
def test_parse_rejects(text):
    with pytest.raises(ValueError):
        parse_combination(text)


def test_cli_combo_flags():
    args = main.build_parser().parse_args(
        ["--combo", "cataract=0.3,acuity=0.3", "--combo", "tunnel=0.7", "--combo-only"]
    )
    assert args.combo == [[("cataract", 0.3), ("acuity", 0.3)], [("tunnel", 0.7)]]
    assert args.combo_only


def test_cli_bad_combo_exits():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--combo", "cataract=3"])


def test_combo_only_without_combo_exits(monkeypatch):
    monkeypatch.setattr(config, "COMBINATIONS", ())
    with pytest.raises(SystemExit):
        main.main(["--combo-only"])


def test_build_simulations_with_combinations(text_image):
    v = ViewingConditions()
    combo = [("cataract", 0.3), ("acuity", 0.3)]
    sims = config.build_simulations(v, combinations=[combo])
    name = "Катаракта (20/40) + Гострота зору (20/40)"
    assert name in sims and len(sims) == len(config.build_simulations(v)) + 1
    np.testing.assert_array_equal(sims[name](text_image), simulate_combination(text_image, combo, v))
    only = config.build_simulations(v, combinations=[combo], presets=False)
    assert list(only) == [name]
