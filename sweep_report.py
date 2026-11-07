"""Звіт прогону по шкалі: JSON, графік matplotlib і таблиця для консолі."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _threshold_dict(model, severity):
    """Поріг кривої у вигляді об'єкта звіту; None, якщо не досягнуто."""
    if severity is None:
        return None
    clinical = model.to_clinical(severity)
    return {
        "severity": severity,
        "clinical": clinical,
        "clinical_label": model.format_clinical(clinical),
    }


def sweep_to_dict(
    url,
    viewing,
    baseline_word_count,
    onset_ilr,
    breaking_ilr,
    curves,
    models,
    total_s,
    created,
):
    """Словник звіту (структура — специфікація §7); ILR і пороги — частки 0..1."""
    model_entries = []
    for curve in curves:
        model = models[curve.model_key]
        model_entries.append(
            {
                "key": model.key,
                "name": model.name,
                "unit": model.unit,
                "failed": curve.failed,
                "onset": _threshold_dict(model, curve.onset),
                "breaking": _threshold_dict(model, curve.breaking),
                "points": [
                    {
                        "severity": p.severity,
                        "clinical": p.clinical,
                        "clinical_label": p.clinical_label,
                        "ilr": p.ilr,
                        "error": p.error,
                    }
                    for p in curve.points
                ],
            }
        )
    return {
        "url": url,
        "created": created,
        "viewing": {
            "distance_cm": viewing.distance_cm,
            "ppi": viewing.ppi,
            "dpr": viewing.dpr,
            "px_per_degree": viewing.px_per_degree,
            "nyquist_cpd": viewing.nyquist_cpd,
        },
        "baseline_word_count": baseline_word_count,
        "thresholds": {"onset_ilr": onset_ilr, "breaking_ilr": breaking_ilr},
        "models": model_entries,
        "timing": {"total_s": total_s},
    }


def write_sweep_json(path, data):
    """Записує звіт у JSON (UTF-8, кирилиця без екранування)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_sweep(path, curves, models, onset_ilr, breaking_ilr):
    """Графік ILR(s): панель на модель, сітка 3×3, 300 dpi."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    flat = list(axes.flat)
    try:
        for ax, curve in zip(flat, curves):
            model = models[curve.model_key]
            pts = [p for p in curve.points if p.ilr is not None]
            ax.plot(
                [p.severity for p in pts],
                [p.ilr * 100 for p in pts],
                marker="o",
                markersize=3,
                color="tab:blue",
            )
            ax.axhline(onset_ilr * 100, color="tab:orange", linestyle="--", linewidth=1)
            ax.axhline(breaking_ilr * 100, color="tab:red", linestyle="--", linewidth=1)
            if curve.onset is not None:
                ax.axvline(curve.onset, color="tab:orange", linestyle=":", linewidth=1)
            if curve.breaking is not None:
                ax.axvline(curve.breaking, color="tab:red", linestyle=":", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 100)
            ax.set_title(model.name)
            ax.set_xlabel("Тяжкість s")
            ax.set_ylabel("ILR, %")
            ax.grid(alpha=0.3)
            top = ax.secondary_xaxis("top")
            ticks = [0.0, 0.5, 1.0]
            top.set_xticks(ticks)
            top.set_xticklabels(
                [model.format_clinical(model.to_clinical(s)) for s in ticks],
                fontsize=8,
            )
            if curve.failed:
                ax.text(
                    0.5,
                    0.5,
                    "ПОМИЛКА",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="tab:red",
                    alpha=0.7,
                )
        for ax in flat[len(curves):]:
            ax.set_visible(False)
        fig.savefig(path, dpi=300)
    finally:
        plt.close(fig)


def _fmt_threshold(model, severity):
    if severity is None:
        return "> 1.0"
    return f"{severity:.2f} ({model.format_clinical(model.to_clinical(severity))})"


def format_sweep_table(curves, models):
    """Таблиця для консолі: рядок на модель."""
    rows = [("Модель", "Деградація (s, клінічне)", "Руйнування (s, клінічне)")]
    for curve in curves:
        model = models[curve.model_key]
        if curve.failed:
            rows.append((model.name, "ПОМИЛКА", "ПОМИЛКА"))
        else:
            rows.append(
                (
                    model.name,
                    _fmt_threshold(model, curve.onset),
                    _fmt_threshold(model, curve.breaking),
                )
            )
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)
