import pytest

from viewing import ViewingConditions


def test_default_geometry_px_per_degree():
    v = ViewingConditions()
    assert v.px_per_degree == pytest.approx(44.9, abs=0.2)
    assert v.nyquist_cpd == pytest.approx(22.45, abs=0.1)


def test_degrees_pixels_round_trip():
    v = ViewingConditions(distance_cm=45, ppi=160, dpr=2)
    assert v.px_to_deg(v.deg_to_px(7.5)) == pytest.approx(7.5)
    assert v.cycles_per_px_to_cpd(v.cpd_to_cycles_per_px(12)) == pytest.approx(12)


def test_dpr_makes_css_pixel_larger():
    assert ViewingConditions(dpr=2).px_per_degree == pytest.approx(
        ViewingConditions().px_per_degree / 2
    )


@pytest.mark.parametrize(
    "kwargs", [{"distance_cm": 0}, {"ppi": -1}, {"dpr": 0}, {"viewport_height_px": 0},
              {"distance_cm": float("nan")}, {"ppi": float("nan")}, {"dpr": float("nan")}]
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        ViewingConditions(**kwargs)
