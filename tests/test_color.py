import numpy as np
import pytest

import color_utils as cu
import simulations as sim


def bgr_pixel(b, g, r):
    return np.array([[[b, g, r]]], dtype=np.uint8)


def test_lms_matrices_are_inverse():
    np.testing.assert_allclose(
        cu.MATRIX_RGB_TO_LMS @ cu.MATRIX_LMS_TO_RGB, np.eye(3), atol=1e-12
    )


def test_gamma_round_trip_is_exact_for_all_levels():
    levels = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
    np.testing.assert_array_equal(cu.gamma_encode(cu.gamma_decode(levels)), levels)


def test_rgb_matrix_to_bgr_matches_channel_swap():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 3))
    rgb = rng.uniform(size=3)
    out_rgb = m @ rgb
    out_bgr = cu.rgb_matrix_to_bgr(m) @ rgb[::-1]
    np.testing.assert_allclose(out_bgr, out_rgb[::-1])


@pytest.mark.parametrize("sim_type", ["protanopia", "deuteranopia", "tritanopia"])
@pytest.mark.parametrize(
    "func", [sim.simulate_dichromacy_brettel, sim.simulate_dichromacy_vienot]
)
def test_dichromacy_preserves_neutral_colors(func, sim_type):
    for v in (0, 64, 128, 200, 255):
        np.testing.assert_array_equal(
            func(bgr_pixel(v, v, v), sim_type=sim_type), bgr_pixel(v, v, v)
        )


@pytest.mark.parametrize(
    "func", [sim.simulate_dichromacy_brettel, sim.simulate_dichromacy_vienot]
)
@pytest.mark.parametrize("sim_type", ["protanopia", "deuteranopia"])
def test_red_green_dichromat_does_not_see_pure_red(func, sim_type):
    """Регресія BGR/RGB: чистий червоний для протанопа/дейтеранопа стає жовто-коричневим."""
    b, g, r = (int(c) for c in func(bgr_pixel(0, 0, 255), sim_type=sim_type)[0, 0])
    assert b < 20  # синя складова майже відсутня
    assert g > 80  # зелена складова з'являється
    assert r - g < 70  # червоний більше не домінує


@pytest.mark.parametrize(
    "func", [sim.simulate_dichromacy_brettel, sim.simulate_dichromacy_vienot]
)
def test_protanopia_keeps_pure_blue_bluish(func):
    b, g, r = (int(c) for c in func(bgr_pixel(255, 0, 0), sim_type="protanopia")[0, 0])
    assert b > 200 and r < 30


def test_brettel_differs_from_vienot_for_tritanopia():
    img = bgr_pixel(0, 0, 255)
    assert not np.array_equal(
        sim.simulate_dichromacy_brettel(img, "tritanopia"),
        sim.simulate_dichromacy_vienot(img, "tritanopia"),
    )


@pytest.mark.parametrize("sim_type", ["protanopia", "deuteranopia", "tritanopia"])
def test_brettel_projection_is_idempotent(sim_type):
    """Колір, який бачить дихромат, після повторної симуляції не змінюється."""
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 1, size=(50, 50, 3)).astype(np.float32)
    once = cu.simulate_brettel_linear_bgr(img, sim_type)
    twice = cu.simulate_brettel_linear_bgr(once.astype(np.float32), sim_type)
    np.testing.assert_allclose(twice, once, atol=1e-4)


def test_unknown_dichromacy_type_raises():
    with pytest.raises(ValueError):
        sim.simulate_dichromacy_brettel(bgr_pixel(0, 0, 0), sim_type="unknown")


def test_machado_severity_zero_is_identity():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    out = sim.simulate_anomalous_trichromacy_machado(img, "protanomaly", severity=0.0)
    np.testing.assert_array_equal(out, img)


def test_machado_matches_published_values():
    np.testing.assert_allclose(
        cu.machado_rgb_matrix("protanomaly", 1.0)[0], [0.152286, 1.052583, -0.204868]
    )
    np.testing.assert_allclose(
        cu.machado_rgb_matrix("deuteranomaly", 0.5)[0], [0.547494, 0.607765, -0.155259]
    )


def test_machado_interpolates_between_table_values():
    m = cu.machado_rgb_matrix("deuteranomaly", 0.55)
    expected = 0.5 * cu.machado_rgb_matrix("deuteranomaly", 0.5) + 0.5 * cu.machado_rgb_matrix(
        "deuteranomaly", 0.6
    )
    np.testing.assert_allclose(m, expected)


def test_machado_changes_red_and_keeps_white():
    out = sim.simulate_anomalous_trichromacy_machado(
        bgr_pixel(0, 0, 255), "protanomaly", severity=0.6
    )
    assert not np.array_equal(out, bgr_pixel(0, 0, 255))
    white = sim.simulate_anomalous_trichromacy_machado(
        bgr_pixel(255, 255, 255), "protanomaly", severity=0.6
    )
    assert np.all(white >= 253)


@pytest.mark.parametrize("severity", [-0.1, 1.1])
def test_machado_rejects_invalid_severity(severity):
    with pytest.raises(ValueError):
        cu.machado_rgb_matrix("protanomaly", severity)
