import json

import pytest

from severity import MODELS
from sweep import SweepCurve, SweepPoint
from sweep_report import (
    format_error_summary,
    format_sweep_table,
    plot_sweep,
    sweep_to_dict,
    write_sweep_json,
)
from viewing import ViewingConditions


def sample_curves():
    pts = [SweepPoint(i / 10, i / 10, "20/20", min(1.0, i / 10)) for i in range(11)]
    return [SweepCurve("acuity", pts, 0.05, 0.5), SweepCurve("cataract", pts, 0.05, None)]


def test_dict_structure():
    d = sweep_to_dict("https://x", ViewingConditions(), 745, 0.05, 0.5, sample_curves(), MODELS, 12.3, "2026-11-07T12:00:00")
    assert d["thresholds"] == {"onset_ilr": 0.05, "breaking_ilr": 0.5}
    assert d["viewing"]["px_per_degree"] == pytest.approx(44.9, abs=0.2)
    acuity = d["models"][0]
    assert acuity["key"] == "acuity" and acuity["unit"] == "logMAR"
    assert acuity["breaking"]["clinical_label"] == "20/65"
    assert d["models"][1]["breaking"] is None
    assert len(acuity["points"]) == 11


def test_dict_key_order():
    d = sweep_to_dict("u", ViewingConditions(), 1, 0.05, 0.5, sample_curves(), MODELS, 1.0, "t")
    assert list(d) == ["url", "created", "viewing", "baseline_word_count", "thresholds", "models", "timing"]
    assert list(d["viewing"]) == ["distance_cm", "ppi", "dpr", "px_per_degree", "nyquist_cpd"]
    m = d["models"][0]
    assert list(m) == ["key", "name", "unit", "failed", "onset", "breaking", "points"]
    assert list(m["onset"]) == ["severity", "clinical", "clinical_label", "upper_bound"]
    assert list(m["points"][0]) == ["severity", "clinical", "clinical_label", "ilr", "error"]
    assert d["timing"] == {"total_s": 1.0}


def test_json_preserves_cyrillic(tmp_path):
    path = tmp_path / "звіт" / "sweep.json"
    path.parent.mkdir()
    d = sweep_to_dict("https://x", ViewingConditions(), 1, 0.05, 0.5, sample_curves(), MODELS, 1.0, "t")
    write_sweep_json(str(path), d)
    assert "Гострота зору" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == d


def test_plot_written(tmp_path):
    p = tmp_path / "sweep_ilr.png"
    plot_sweep(str(p), sample_curves(), MODELS, 0.05, 0.5)
    assert p.stat().st_size > 10_000


def test_plot_survives_failed_curve_and_none_points(tmp_path):
    pts = [SweepPoint(0.0, 0.0, "a", 0.0), SweepPoint(0.5, 0.5, "b", None, "boom"),
           SweepPoint(1.0, 1.0, "c", 0.6)]
    curves = [SweepCurve("acuity", pts, None, 0.9, True), SweepCurve("cataract", pts, 0.5, None)]
    p = tmp_path / "f.png"
    plot_sweep(str(p), curves, MODELS, 0.05, 0.5)
    assert p.stat().st_size > 10_000


def test_table_marks_unreached():
    t = format_sweep_table(sample_curves(), MODELS)
    assert "Катаракта" in t and "> 1.0" in t


def test_table_marks_failed():
    c = SweepCurve("acuity", [], None, None, True)
    assert "ПОМИЛКА" in format_sweep_table([c], MODELS)


def test_upper_bound_key_and_table():
    pts = [SweepPoint(i / 10, i / 10, "20/20", min(1.0, i / 10)) for i in range(11)]
    curves = [SweepCurve("acuity", pts, 0.1, 0.5, False, True, False)]
    d = sweep_to_dict("u", ViewingConditions(), 1, 0.05, 0.5, curves, MODELS, 1.0, "t")
    assert d["models"][0]["onset"]["upper_bound"] is True
    assert d["models"][0]["breaking"]["upper_bound"] is False
    row = format_sweep_table(curves, MODELS).splitlines()[2]
    onset_cell, breaking_cell = row.split(" | ")[1:]
    assert onset_cell.strip().startswith("≤ 0.10 (≤ ")
    assert "≤" not in breaking_cell


def test_plot_with_bounded_threshold(tmp_path):
    pts = [SweepPoint(i / 10, i / 10, "20/20", min(1.0, i / 10)) for i in range(11)]
    plot_sweep(str(tmp_path / "p.png"), [SweepCurve("acuity", pts, 0.1, 0.5, False, True, True)], MODELS, 0.05, 0.5)
    assert (tmp_path / "p.png").exists()


def test_error_summary_first_error_per_model():
    pts = [SweepPoint(0, 0, "", 0.0), SweepPoint(0.1, 0.1, "", None, "boom"), SweepPoint(0.2, 0.2, "", None, "bang")]
    lines = format_error_summary([SweepCurve("acuity", pts, None, None), SweepCurve("cataract", pts[:1], None, None)], MODELS)
    assert lines == ["Гострота зору: 2 помилкових точок, перша: boom"]
