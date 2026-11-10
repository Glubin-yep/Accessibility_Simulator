"""Геометрія перегляду: зв'язок між градусами поля зору, пікселями екрану та cpd."""

import math
from dataclasses import dataclass

_MM_PER_INCH = 25.4
_HALF_DEGREE_RAD = math.radians(0.5)


@dataclass(frozen=True)
class ViewingConditions:
    """Умови перегляду: відстань до екрану, щільність пікселів і масштаб DPR.

    distance_cm -- відстань від ока до екрану, см
    ppi -- фізична щільність пікселів екрану
    dpr -- співвідношення CSS-пікселів до фізичних (device pixel ratio)
    viewport_height_px -- висота області перегляду, CSS-пікселі
    """

    distance_cm: float = 60.0
    ppi: float = 109.0
    dpr: float = 1.0
    viewport_height_px: int = 1440

    def __post_init__(self):
        for name in ("distance_cm", "ppi", "dpr", "viewport_height_px"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} має бути додатним, отримано {getattr(self, name)}")

    @property
    def pixel_mm(self) -> float:
        """Розмір одного CSS-пікселя, мм."""
        return _MM_PER_INCH * self.dpr / self.ppi

    @property
    def px_per_degree(self) -> float:
        """Кількість CSS-пікселів, що відповідає одному градусу поля зору."""
        distance_mm = self.distance_cm * 10.0
        return 2.0 * distance_mm * math.tan(_HALF_DEGREE_RAD) / self.pixel_mm

    @property
    def nyquist_cpd(self) -> float:
        """Гранична просторова частота екрану (Найквіста), цикли на градус."""
        return 0.5 * self.px_per_degree

    def deg_to_px(self, deg: float) -> float:
        """Перетворює градуси поля зору на CSS-пікселі."""
        return deg * self.px_per_degree

    def px_to_deg(self, px: float) -> float:
        """Перетворює CSS-пікселі на градуси поля зору."""
        return px / self.px_per_degree

    def cpd_to_cycles_per_px(self, f: float) -> float:
        """Перетворює цикли на градус на цикли на CSS-піксель."""
        return f / self.px_per_degree

    def cycles_per_px_to_cpd(self, f: float) -> float:
        """Перетворює цикли на CSS-піксель на цикли на градус."""
        return f * self.px_per_degree
