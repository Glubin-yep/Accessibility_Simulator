import pytest

from analysis import compute_metrics, normalize_word
from reporting import simulation_filename


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Привіт!", "привіт"),
        ("«Сторінка»,", "сторінка"),
        ("п'ять", "п'ять"),
        ("WCAG-2.1", "wcag21"),
        ("a!" * 40, "a" * 40),  # регресія: re.sub(..., re.UNICODE) обмежував кількість замін
    ],
)
def test_normalize_word(raw, expected):
    assert normalize_word(raw) == expected


def test_compute_metrics_set_operations():
    base = {"a", "b", "c", "d"}
    sim = {"a", "b", "x"}
    m = compute_metrics(base, sim)
    assert (m.baseline_count, m.sim_count) == (4, 3)
    assert m.common_count == 2  # {a, b}
    assert m.lost_count == 2  # {c, d}
    assert m.artifact_count == 1  # {x}
    assert m.overlap_ratio == pytest.approx(0.5)
    assert m.information_loss_ratio == pytest.approx(0.5)


def test_metrics_with_empty_baseline():
    m = compute_metrics(set(), {"x"})
    assert m.information_loss_ratio == 0.0
    assert m.overlap_ratio == 0.0


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Флоатери (Діабет. рет.)", "screenshot_флоатери_діабет_рет.png"),
        ("Протанопія (Brettel)", "screenshot_протанопія_brettel.png"),
        ("Втрата контрасту (CSF)", "screenshot_втрата_контрасту_csf.png"),
    ],
)
def test_simulation_filename(name, expected):
    assert simulation_filename(name) == expected
