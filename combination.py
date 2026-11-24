"""Комбінована симуляція кількох порушень зору зі своїм ступенем кожне.

Порушення застосовуються не в порядку, заданому користувачем, а в порядку
проходження світла зоровою системою:

1. Оптичні середовища ока (кришталик): пожовтіння і засвітка катаракти.
2. Сітківка, геометрія: зсув фоторецепторів при метаморфопсії.
3. Сітківка, фоторецептори: аномалії колбочок (Machado), L -> M -> S.
4. Просторовий аналіз (CSF спостерігача) і втрата поля зору.

Кроки 1-3 поелементні або геометричні, тож для них порядок майже не впливає
на результат; принципово, що пожовтіння передує колбочкам (колбочки бачать
уже відфільтроване кришталиком світло) і що CSF застосовується один раз,
наприкінці: подвійне порогування спостерігачем нефізичне.

Втрати CSF від кількох причин зводяться до однієї ObserverCSF:
- гострота: розмиття від незалежних причин додаються в квадратурі
  (як дисперсії гаусових ядер), тому MAR = sqrt(1 + Σ(MAR_i² − 1)),
  де MAR_i = 10**logMAR_i, а одиниця -- внесок нормального ока, врахований один раз;
- контрастна чутливість: послаблення контрасту від незалежних причин
  перемножуються, тобто дефіцити log CS додаються:
  log CS = 1.95 − Σ(1.95 − log CS_i), але не нижче MIN_LOGCS.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from observer import LOST_FIELD_OBSERVER, NORMAL_OBSERVER, ObserverCSF, apply_observer
from severity import MODELS, preset_name
from simulations import (
    blend_by_mask,
    cataract_optics,
    field_loss_mask,
    simulate_anomalous_trichromacy_machado,
    simulate_metamorphopsia,
)

# Нижня межа log CS комбінованого спостерігача: рівень втраченого поля зору.
MIN_LOGCS = LOST_FIELD_OBSERVER.logcs

# Порядок застосування = порядок проходження світла (див. докстрінг модуля).
STAGE_ORDER = (
    "cataract",
    "metamorphopsia",
    "protanomaly",
    "deuteranomaly",
    "tritanomaly",
    "acuity",
    "contrast",
    "scotoma",
    "tunnel",
)


def combine_csf(observers):
    """Зводить кілька ObserverCSF в одну (квадратура MAR, сума дефіцитів log CS)."""
    impaired = [o for o in observers if o != NORMAL_OBSERVER]
    if not impaired:
        return NORMAL_OBSERVER
    if len(impaired) == 1:
        return impaired[0]
    base_mar = 10.0**NORMAL_OBSERVER.logmar
    mar_sq = base_mar**2 + sum((10.0**o.logmar) ** 2 - base_mar**2 for o in impaired)
    logmar = math.log10(math.sqrt(mar_sq))
    deficit = sum(NORMAL_OBSERVER.logcs - o.logcs for o in impaired)
    logcs = max(NORMAL_OBSERVER.logcs - deficit, MIN_LOGCS)
    return ObserverCSF(logmar, logcs)


@dataclass
class _Plan:
    """Внески порушень, розкладені за етапами зорового шляху."""

    pixel_steps: list = field(default_factory=list)  # етапи 1-3: image -> image
    observers: list = field(default_factory=list)  # етап 4: CSF неушкодженого поля
    fields: list = field(default_factory=list)  # етап 4: (радіус °, втрата всередині)


def _add_cataract(plan, s, viewing):
    plan.pixel_steps.append(lambda img: cataract_optics(img, 1 - 0.3 * s, 1 - 0.08 * s, 0.3 * s))
    plan.observers.append(ObserverCSF(1.0 * s, 1.95 - 1.05 * s))


def _add_metamorphopsia(plan, s, viewing):
    ppd = viewing.px_per_degree
    plan.pixel_steps.append(
        lambda img: simulate_metamorphopsia(
            img, amplitude=2.0 * s * ppd, frequency=2.0 * math.pi / (3.0 * ppd)
        )
    )


def _add_machado(sim_type):
    def add(plan, s, viewing):
        plan.pixel_steps.append(
            lambda img: simulate_anomalous_trichromacy_machado(img, sim_type, s)
        )

    return add


def _add_acuity(plan, s, viewing):
    plan.observers.append(ObserverCSF(1.0 * s, 1.95))


def _add_contrast(plan, s, viewing):
    plan.observers.append(ObserverCSF(0.0, 1.95 - 1.5 * s))


def _add_scotoma(plan, s, viewing):
    plan.fields.append((10.0 * s, True))


def _add_tunnel(plan, s, viewing):
    # s = 0 -- поле не втрачене (так само, як у severity._tunnel).
    if s > 0.0:
        plan.fields.append((30.0 - 28.0 * s, False))


_CONTRIBUTIONS = {
    "cataract": _add_cataract,
    "metamorphopsia": _add_metamorphopsia,
    "protanomaly": _add_machado("protanomaly"),
    "deuteranomaly": _add_machado("deuteranomaly"),
    "tritanomaly": _add_machado("tritanomaly"),
    "acuity": _add_acuity,
    "contrast": _add_contrast,
    "scotoma": _add_scotoma,
    "tunnel": _add_tunnel,
}


def validate_components(components):
    """Перевіряє [(ключ моделі, ступінь s), ...]; повертає їх у порядку STAGE_ORDER."""
    components = list(components)
    if not components:
        raise ValueError("Комбінація має містити хоча б одне порушення")
    keys = [key for key, _ in components]
    for key, s in components:
        if key not in _CONTRIBUTIONS:
            raise ValueError(
                f"Невідома модель '{key}'. Доступні: {', '.join(STAGE_ORDER)}"
            )
        if not (isinstance(s, (int, float)) and 0.0 <= s <= 1.0):
            raise ValueError(f"Ступінь порушення '{key}' має бути в межах [0, 1], отримано {s}")
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"Модель вказано більше одного разу: {', '.join(duplicates)}")
    return sorted(components, key=lambda c: STAGE_ORDER.index(c[0]))


def simulate_combination(image, components, viewing):
    """Симулює одночасну дію кількох порушень.

    components -- [(ключ моделі з severity.MODELS, ступінь s ∈ [0, 1]), ...].
    Для одного порушення результат збігається з MODELS[key].simulate.
    """
    plan = _Plan()
    for key, s in validate_components(components):
        _CONTRIBUTIONS[key](plan, float(s), viewing)

    stimulus = image
    for step in plan.pixel_steps:
        stimulus = step(stimulus)

    intact = apply_observer(stimulus, combine_csf(plan.observers), viewing)
    if not plan.fields:
        return intact

    h, w = image.shape[:2]
    masks = [
        field_loss_mask(
            h, w,
            radius_px=viewing.deg_to_px(radius_deg),
            feather_sigma_px=viewing.deg_to_px(1.0),
            tile_height=viewing.viewport_height_px,
            lose_inside=lose_inside,
        )
        for radius_deg, lose_inside in plan.fields
    ]
    if len(masks) == 1:
        mask = masks[0]
    else:
        # Втрачено, якщо втрачено хоча б одним порушенням: 1 − Π(1 − m_i).
        kept = np.ones((h, w), dtype=np.float32)
        for m in masks:
            kept *= 1.0 - m
        mask = (1.0 - kept).astype(np.float32)
    if not mask.any():
        return intact
    lost_csf = combine_csf([LOST_FIELD_OBSERVER, *plan.observers])
    lost = apply_observer(stimulus, lost_csf, viewing)
    return blend_by_mask(intact, lost, mask)


def combination_name(components):
    """'Катаракта (20/40) + Протаномалія (Machado 1)' -- у порядку етапів."""
    return " + ".join(preset_name(MODELS[k], s) for k, s in validate_components(components))


def parse_combination(text):
    """'cataract=0.3,protanomaly=1' -> [('cataract', 0.3), ('protanomaly', 1.0)]."""
    components = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Очікується модель=ступінь, отримано '{part}'")
        try:
            s = float(value)
        except ValueError:
            raise ValueError(f"Ступінь для '{key.strip()}' має бути числом, отримано '{value}'")
        components.append((key.strip(), s))
    return validate_components(components)
