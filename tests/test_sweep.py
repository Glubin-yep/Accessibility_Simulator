import numpy as np
import pytest

from severity import SeverityModel
from sweep import SweepPoint, find_threshold, run_sweep
from viewing import ViewingConditions

P = lambda s, ilr: SweepPoint(s, s, f"{s}", ilr)


def test_threshold_interpolates():
    pts = [P(0, 0), P(0.5, 0.2), P(1.0, 0.6)]
    assert find_threshold(pts, 0.5) == pytest.approx(0.875)


def test_threshold_takes_first_crossing_on_noisy_curve():
    pts = [P(0, 0), P(0.2, 0.06), P(0.4, 0.03), P(0.6, 0.08)]
    assert find_threshold(pts, 0.05) == pytest.approx(0.2 * 0.05 / 0.06)


def test_threshold_skips_failed_points():
    pts = [P(0, 0), SweepPoint(0.5, 0.5, "", None, "boom"), P(1.0, 0.4)]
    assert find_threshold(pts, 0.2) == pytest.approx(0.5)


def test_threshold_not_reached():
    assert find_threshold([P(0, 0), P(1, 0.3)], 0.5) is None


def fake_model(key, fail_at=()):
    def simulate(img, s, v):
        if round(s, 3) in fail_at:
            raise RuntimeError("boom")
        return np.full_like(img, int(round(s * 100)))
    return SeverityModel(key, key, "u", lambda s: s, lambda c: f"{c:.1f}", simulate)


BASE = {f"w{i}" for i in range(100)}
fake_ocr = lambda img: {f"w{i}": [] for i in range(100 - int(img[0, 0, 0]))}  # ILR = s


def test_run_sweep_curve_and_thresholds():
    (curve,) = run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m")], ViewingConditions(), steps=10, workers=2, ocr=fake_ocr)
    assert [p.severity for p in curve.points] == pytest.approx([i / 10 for i in range(11)])
    assert curve.onset == pytest.approx(0.05) and curve.breaking == pytest.approx(0.5)


def test_partial_failures_are_recorded():
    (curve,) = run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m", fail_at=(0.3,))], ViewingConditions(), steps=10, ocr=fake_ocr)
    bad = [p for p in curve.points if p.ilr is None]
    assert len(bad) == 1 and bad[0].error == "boom" and not curve.failed


def test_model_failed_when_most_points_fail():
    fails = tuple(round(i / 10, 3) for i in range(1, 7))
    (curve,) = run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m", fail_at=fails)], ViewingConditions(), steps=10, ocr=fake_ocr)
    assert curve.failed and curve.onset is None and curve.breaking is None


def test_image_sink_called_for_each_point():
    calls = []
    run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m")], ViewingConditions(), steps=4, ocr=fake_ocr, image_sink=lambda k, s, im: calls.append((k, s)))
    assert sorted(s for _, s in calls) == pytest.approx([0.25, 0.5, 0.75, 1.0])
