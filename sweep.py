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
    onset_bounded: bool = False
    breaking_bounded: bool = False


def _first_crossing(points, threshold):
    """(інтерпольоване s, s1, чи нижня опора -- s = 0) для першого перетину або None."""
    ok = sorted((p for p in points if p.ilr is not None), key=lambda p: p.severity)
    prev = None
    for p in ok:
        if p.ilr >= threshold:
            if prev is None or p.ilr == prev.ilr:
                return p.severity, p.severity, prev is None or prev.severity <= 0
            interpolated = prev.severity + (threshold - prev.ilr) * (
                p.severity - prev.severity
            ) / (p.ilr - prev.ilr)
            return interpolated, p.severity, prev.severity <= 0
        prev = p
    return None


def find_threshold(points, threshold):
    """Перше перетинання порогу з лінійною інтерполяцією; None, якщо не досягнуто.

    Точки з помилками (ilr is None) пропускаються.
    """
    crossing = _first_crossing(points, threshold)
    return None if crossing is None else crossing[0]


def find_threshold_bounded(points, threshold):
    """Як find_threshold, але повертає (тяжкість, є_верхньою_межею).

    Якщо перетин лежить у (0, s1] (нижня опорна точка інтерполяції -- несимульований
    s = 0), це лише верхня межа: повертається s1 (перша симульована точка >= порогу).
    Не досягнуто -> (None, False).
    """
    crossing = _first_crossing(points, threshold)
    if crossing is None:
        return None, False
    interpolated, first_point, bounded = crossing
    return (first_point if bounded else interpolated), bounded


def _make_point(model, s, image, viewing, baseline_word_set, ocr, image_sink):
    clinical = model.to_clinical(s)
    label = model.format_clinical(clinical)
    try:
        simulated = model.simulate(image, s, viewing)
        if image_sink is not None:
            try:
                image_sink(model.key, s, simulated)
            except Exception as exc:
                print(
                    f"Попередження: не вдалося зберегти зображення ({model.key}, s={s:.2f}): {exc}"
                )
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
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        # По колу між моделями: одночасні точки належать різним моделям,
        # тож піки пам'яті (поле зору) не збігаються.
        futures = {
            (m.key, s): executor.submit(
                _make_point, m, s, image, viewing, baseline_word_set, ocr, image_sink
            )
            for s in grid
            for m in models
        }
        results = {k: f.result() for k, f in futures.items()}
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()

    curves = []
    for m in models:
        zero_clinical = m.to_clinical(0.0)
        zero = SweepPoint(0.0, zero_clinical, m.format_clinical(zero_clinical), 0.0)
        points = [zero] + [results[(m.key, s)] for s in grid]
        errors = sum(1 for p in points if p.error is not None)
        failed = errors > steps / 2
        onset, onset_bounded = (None, False) if failed else find_threshold_bounded(points, onset_ilr)
        breaking, breaking_bounded = (
            (None, False) if failed else find_threshold_bounded(points, breaking_ilr)
        )
        curves.append(
            SweepCurve(
                m.key, points, onset, breaking, failed, onset_bounded, breaking_bounded
            )
        )
    return curves
