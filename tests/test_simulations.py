import cv2
import numpy as np
import pytest

import config
import simulations as sim
from viewing import ViewingConditions

SIMS = config.build_simulations(ViewingConditions())


def test_preset_names_match_spec():
    assert list(SIMS) == [
        "Гострота зору (20/40)", "Гострота зору (20/100)", "Катаракта (20/40)", "Катаракта (20/100)",
        "Втрата контрасту (log CS 1.20)", "Центральна скотома (5°)", "Тунельний зір (10°)",
        "Протаномалія (Machado 0.6)", "Метаморфопсія (0.5°)", "Метаморфопсія (1.5°)",
        "Ахроматопсія (Скотопічна)", "Протанопія (Brettel)", "Дейтеранопія (Brettel)",
        "Тританопія (Brettel)", "Флоатери (Діабет. рет.)",
    ]


@pytest.mark.parametrize("name", list(SIMS))
def test_every_configured_simulation_keeps_shape_and_dtype(name, text_image):
    out = SIMS[name](text_image.copy())
    assert out.shape == text_image.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", list(SIMS))
def test_every_configured_simulation_changes_the_image(name, text_image):
    """Жодна симуляція з конфігурації не повинна бути заглушкою."""
    out = SIMS[name](text_image.copy())
    assert not np.array_equal(out, text_image)


def test_simulations_do_not_modify_input(text_image):
    original = text_image.copy()
    for func in SIMS.values():
        func(text_image)
    np.testing.assert_array_equal(text_image, original)


def test_csf_loss_never_increases_contrast():
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    cv2.putText(img, "Low", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (150, 150, 150), 4)
    out = sim.simulate_csf_loss(img, cutoff_frequency_ratio=0.1)
    assert out.min() >= img.min() - 1
    assert out.max() <= img.max() + 1
    assert out.std() < img.std()


def test_csf_loss_preserves_mean_color():
    img = np.full((100, 100, 3), (240, 200, 180), dtype=np.uint8)
    out = sim.simulate_csf_loss(img, cutoff_frequency_ratio=0.2)
    np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), (240, 200, 180), atol=1)


def test_csf_contrast_parameter_reduces_contrast(text_image):
    full = sim.simulate_csf_loss(text_image, 0.5, contrast=1.0)
    half = sim.simulate_csf_loss(text_image, 0.5, contrast=0.5)
    assert half.std() < full.std() * 0.6


def test_floaters_are_reproducible_with_seed(text_image):
    a = sim.simulate_floaters(text_image, seed=7)
    b = sim.simulate_floaters(text_image, seed=7)
    c = sim.simulate_floaters(text_image, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_floaters_only_darken(text_image):
    out = sim.simulate_floaters(text_image, seed=1)
    assert np.all(out.astype(int) <= text_image.astype(int))


def test_central_scotoma_blurs_center_and_keeps_periphery():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    out = sim.simulate_central_scotoma(img, radius_percentage=0.2)
    center = (slice(190, 210), slice(190, 210))
    corner = (slice(0, 20), slice(0, 20))
    assert out[center].std() < img[center].std() * 0.5
    np.testing.assert_allclose(out[corner], img[corner], atol=2)


def test_tunnel_vision_keeps_center_and_blurs_periphery():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    out = sim.simulate_tunnel_vision(img, aperture_percentage=0.3)
    center = (slice(190, 210), slice(190, 210))
    corner = (slice(0, 20), slice(0, 20))
    np.testing.assert_allclose(out[center], img[center], atol=2)
    assert out[corner].std() < img[corner].std() * 0.5


def test_metamorphopsia_has_no_black_borders():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    out = sim.simulate_metamorphopsia(img, amplitude=15, frequency=0.05)
    assert out.min() == 255


def test_cataract_yellows_and_hazes():
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    b, g, r = sim.simulate_cataract(img, blur_ksize=3, glare_weight=0.2)[25, 25]
    assert b < g < r


def test_tile_fixations_centres():
    # Остання плитка неповна (120 px): центр = 2880 + 120 // 2.
    assert sim.tile_fixations(3000, 1000, 1440) == [(500, 720), (500, 2160), (500, 2940)]


def test_single_partial_tile():
    assert sim.tile_fixations(500, 800, 1440) == [(400, 250)]


def test_scotoma_mask_lost_at_fixation_intact_far_away():
    m = sim.field_loss_mask(2880, 1000, radius_px=100, feather_sigma_px=5, tile_height=1440, lose_inside=True)
    assert m[720, 500] > 0.99 and m[2160, 500] > 0.99
    assert m[720, 0] < 0.01


def test_tunnel_mask_is_inverse_of_scotoma():
    kw = dict(height=1440, width=1000, radius_px=200, feather_sigma_px=10, tile_height=1440)
    np.testing.assert_allclose(
        sim.field_loss_mask(lose_inside=True, **kw) + sim.field_loss_mask(lose_inside=False, **kw), 1, atol=1e-6
    )


def test_zero_radius_scotoma_loses_nothing():
    assert sim.field_loss_mask(500, 500, 0, 5, 1440, lose_inside=True).max() < 1e-6


def test_blend_by_mask_zero_mask_returns_intact_exactly(text_image):
    lost = 255 - text_image
    out = sim.blend_by_mask(text_image, lost, np.zeros(text_image.shape[:2], np.float32))
    np.testing.assert_array_equal(out, text_image)
    assert out.dtype == np.uint8


def test_blend_by_mask_full_mask_returns_lost(text_image):
    lost = 255 - text_image
    out = sim.blend_by_mask(text_image, lost, np.ones(text_image.shape[:2], np.float32))
    np.testing.assert_array_equal(out, lost)


def test_cataract_optics_identity_at_zero(text_image):
    np.testing.assert_array_equal(sim.cataract_optics(text_image, 1.0, 1.0, 0.0), text_image)
