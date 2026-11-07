"""Модель спостерігача: контрастна чутливість (CSF) за лог-параболою Lesmes et al. (2010)."""

import math
from dataclasses import dataclass

import numpy as np

BANDWIDTH_OCTAVES = 3.0
TRUNCATION = 0.5
PELLI_ROBSON_TO_PEAK = 0.35

_KAPPA = math.log10(2.0)
_BETA_LOG = math.log10(2.0 * BANDWIDTH_OCTAVES)
_CUTOFF_CPD_AT_LOGMAR_0 = 30.0
_MIN_LOGMAR = -0.3


@dataclass(frozen=True)
class ObserverCSF:
    """CSF спостерігача, задана двома клінічними числами.

    logmar -- гострота зору, logMAR (0 -- норма)
    logcs -- контрастна чутливість, log CS (норма 1.95)
    """

    logmar: float
    logcs: float

    def __post_init__(self):
        if self.logcs <= 0:
            raise ValueError(f"logcs має бути додатним, отримано {self.logcs}")
        if self.logmar < _MIN_LOGMAR:
            raise ValueError(f"logmar має бути не менше {_MIN_LOGMAR}, отримано {self.logmar}")

    @property
    def log_gamma_max(self) -> float:
        """Логарифм піку чутливості log10 γmax."""
        return self.logcs + PELLI_ROBSON_TO_PEAK

    @property
    def cutoff_cpd(self) -> float:
        """Частота зрізу (S = 1), цикл/град."""
        return _CUTOFF_CPD_AT_LOGMAR_0 * 10 ** -self.logmar

    @property
    def fmax_cpd(self) -> float:
        """Частота піку, цикл/град."""
        exponent = (_BETA_LOG / 2.0) * math.sqrt(self.log_gamma_max / _KAPPA)
        return self.cutoff_cpd / 10 ** exponent

    def log_sensitivity(self, f_cpd):
        """Повертає log10 S(f) для частоти f_cpd (цикл/град).

        Приймає float або np.ndarray; повертає того самого типу.
        Для f < fmax значення усікається знизу на рівні log10 γmax − δ.
        """
        f = np.asarray(f_cpd, dtype=float)
        if np.any(f <= 0):
            raise ValueError("частота має бути додатною")

        log_ratio = np.log10(f / self.fmax_cpd) / (_BETA_LOG / 2.0)
        parabola = self.log_gamma_max - _KAPPA * log_ratio**2
        truncated = np.where(
            f < self.fmax_cpd,
            np.maximum(parabola, self.log_gamma_max - TRUNCATION),
            parabola,
        )
        if np.ndim(f_cpd) == 0:
            return float(truncated)
        return truncated


NORMAL_OBSERVER = ObserverCSF(0.0, 1.95)
LOST_FIELD_OBSERVER = ObserverCSF(1.6, 0.3)
