"""Клінічні моделі тяжкості: s ∈ [0, 1] -> клінічне значення і симуляція."""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from observer import LOST_FIELD_OBSERVER, NORMAL_OBSERVER, ObserverCSF, apply_observer
from simulations import (
    blend_by_mask,
    cataract_optics,
    field_loss_mask,
    simulate_anomalous_trichromacy_machado,
    simulate_metamorphopsia,
)
from viewing import ViewingConditions


@dataclass(frozen=True)
class SeverityModel:
    key: str
    name: str
    unit: str
    to_clinical: Callable[[float], float]
    format_clinical: Callable[[float], str]
    simulate: Callable[[np.ndarray, float, ViewingConditions], np.ndarray]


def format_snellen(logmar: float) -> str:
    return f"20/{5 * round(20 * 10**logmar / 5)}"


def format_degrees(v: float) -> str:
    return f"{round(v, 1):g}°"


def _format_log_cs(v: float) -> str:
    return f"log CS {v:.2f}"


def _format_machado(v: float) -> str:
    return f"Machado {round(v, 2):g}"


def _checked(simulate):
    def wrapper(image, s, viewing):
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"Тяжкість s має бути в межах [0, 1], отримано {s}")
        return simulate(image, s, viewing)

    return wrapper


def _acuity(image, s, viewing):
    return apply_observer(image, ObserverCSF(1.0 * s, 1.95), viewing)


def _cataract(image, s, viewing):
    optics = cataract_optics(image, 1 - 0.3 * s, 1 - 0.08 * s, 0.3 * s)
    return apply_observer(optics, ObserverCSF(1.0 * s, 1.95 - 1.05 * s), viewing)


def _contrast(image, s, viewing):
    return apply_observer(image, ObserverCSF(0.0, 1.95 - 1.5 * s), viewing)


def _field_loss(image, viewing, radius_deg, lose_inside):
    intact = apply_observer(image, NORMAL_OBSERVER, viewing)
    h, w = image.shape[:2]
    mask = field_loss_mask(
        h, w,
        radius_px=viewing.deg_to_px(radius_deg),
        feather_sigma_px=viewing.deg_to_px(1.0),
        tile_height=viewing.viewport_height_px,
        lose_inside=lose_inside,
    )
    if not mask.any():
        return intact
    lost = apply_observer(image, LOST_FIELD_OBSERVER, viewing)
    return blend_by_mask(intact, lost, mask)


def _scotoma(image, s, viewing):
    return _field_loss(image, viewing, 10.0 * s, True)


def _tunnel(image, s, viewing):
    if s == 0.0:
        # s = 0 означає відсутність втрати поля (радіус 30° не покриває кути фрагмента).
        return apply_observer(image, NORMAL_OBSERVER, viewing)
    return _field_loss(image, viewing, 30.0 - 28.0 * s, False)


def _machado(sim_type):
    def simulate(image, s, viewing):
        optics = simulate_anomalous_trichromacy_machado(image, sim_type, s)
        return apply_observer(optics, NORMAL_OBSERVER, viewing)

    return simulate


def _metamorphopsia(image, s, viewing):
    ppd = viewing.px_per_degree
    optics = simulate_metamorphopsia(
        image, amplitude=2.0 * s * ppd, frequency=2.0 * math.pi / (3.0 * ppd)
    )
    return apply_observer(optics, NORMAL_OBSERVER, viewing)


def _model(key, name, unit, to_clinical, fmt, simulate):
    return SeverityModel(key, name, unit, to_clinical, fmt, _checked(simulate))


def _identity(s):
    return s


MODELS: dict[str, SeverityModel] = {
    m.key: m
    for m in (
        _model("acuity", "Гострота зору", "logMAR", lambda s: 1.0 * s, format_snellen, _acuity),
        _model("cataract", "Катаракта", "logMAR", lambda s: 1.0 * s, format_snellen, _cataract),
        _model("contrast", "Втрата контрасту", "log CS", lambda s: 1.95 - 1.5 * s, _format_log_cs, _contrast),
        _model("scotoma", "Центральна скотома", "°", lambda s: 10.0 * s, format_degrees, _scotoma),
        _model("tunnel", "Тунельний зір", "°", lambda s: 30.0 - 28.0 * s, format_degrees, _tunnel),
        _model("protanomaly", "Протаномалія", "Machado", _identity, _format_machado, _machado("protanomaly")),
        _model("deuteranomaly", "Дейтераномалія", "Machado", _identity, _format_machado, _machado("deuteranomaly")),
        _model("tritanomaly", "Тританомалія", "Machado", _identity, _format_machado, _machado("tritanomaly")),
        _model("metamorphopsia", "Метаморфопсія", "M-CHARTS °", lambda s: 2.0 * s, format_degrees, _metamorphopsia),
    )
}


def preset_name(model: SeverityModel, s: float) -> str:
    return f"{model.name} ({model.format_clinical(model.to_clinical(s))})"
