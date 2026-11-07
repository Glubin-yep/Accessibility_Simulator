"""Прогін моделей тяжкості по шкалі s ∈ [0, 1] і пошук порогів деградації."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from analysis import compute_metrics, get_word_data


@dataclass(frozen=True)
class SweepPoint:
    severity: float
    clinical: float
    clinical_label: str
    ilr: float | None
    error: str | None = None


@dataclass(frozen=True)
class SweepCurve:
    model_key: str
    points: list
    onset: float | None
    breaking: float | None
    failed: bool = False


def find_threshold(points, threshold):
    """Перше перетинання порогу з лінійною інтерполяцією; None, якщо не досягнуто.

    Точки з помилками (ilr is None) пропускаються.
    """
    ok = sorted((p for p in points if p.ilr is not None), key=lambda p: p.severity)
    prev = None
    for p in ok:
        if p.ilr >= threshold:
            if prev is None or p.ilr == prev.ilr:
                return p.severity
            return prev.severity + (threshold - prev.ilr) * (
                p.severity - prev.severity
            ) / (p.ilr - prev.ilr)
        prev = p
    return None


def _make_point(model, s, image, viewing, baseline_word_set, ocr, image_sink):
    clinical = model.to_clinical(s)
    label = model.format_clinical(clinical)
    try:
        simulated = model.simulate(image, s, viewing)
        if image_sink is not None:
            image_sink(model.key, s, simulated)
        words = ocr(simulated)
        ilr = compute_metrics(baseline_word_set, words.keys()).information_loss_ratio
        return SweepPoint(s, clinical, label, ilr)
    except Exception as exc:
        return SweepPoint(s, clinical, label, None, str(exc))


def run_sweep(
    image,
    baseline_word_set,
    models,
    viewing,
    steps=10,
    onset_ilr=0.05,
    breaking_ilr=0.5,
    workers=4,
    ocr=get_word_data,
    image_sink=None,
):
    """Прогін кожної моделі по сітці s_i = i / steps; повертає криві в порядку models."""
    grid = [i / steps for i in range(1, steps + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            (m.key, s): pool.submit(
                _make_point, m, s, image, viewing, baseline_word_set, ocr, image_sink
            )
            for m in models
            for s in grid
        }
        results = {k: f.result() for k, f in futures.items()}

    curves = []
    for m in models:
        zero_clinical = m.to_clinical(0.0)
        zero = SweepPoint(0.0, zero_clinical, m.format_clinical(zero_clinical), 0.0)
        points = [zero] + [results[(m.key, s)] for s in grid]
        errors = sum(1 for p in points if p.error is not None)
        failed = errors > steps / 2
        curves.append(
            SweepCurve(
                m.key,
                points,
                None if failed else find_threshold(points, onset_ilr),
                None if failed else find_threshold(points, breaking_ilr),
                failed,
            )
        )
    return curves
