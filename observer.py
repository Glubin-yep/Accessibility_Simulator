"""Модель спостерігача: контрастна чутливість (CSF) за лог-параболою Lesmes et al. (2010)."""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from color_utils import GAMMA_DECODE_LUT, GAMMA_ENCODE_LUT

BANDWIDTH_OCTAVES = 3.0
TRUNCATION = 0.5
PELLI_ROBSON_TO_PEAK = 0.35

_KAPPA = math.log10(2.0)
_BETA_LOG = math.log10(2.0 * BANDWIDTH_OCTAVES)
_CUTOFF_CPD_AT_LOGMAR_0 = 30.0
_MIN_LOGMAR = -0.3

# Центр рівня 0 піраміди Лапласа, цикл/піксель: геометричне середнє частоти
# половинного підсилення рівня 0 (виміряної синусоїдальними ґратками) і Найквіста.
PYRAMID_F0 = 0.2565
_MIN_BAND_CPD = 0.25
_MIN_LEVEL_SIDE = 8
_MIN_LUMINANCE = 1e-4

# Поелементні кроки виконуються смугами рядків у пулі потоків: OpenCV і
# NumPy звільняють GIL, а смуга вміщується в кеш процесора.
_STRIPE_ROWS = 128
_ENCODE_SCALE = float(GAMMA_ENCODE_LUT.size - 1)
_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)


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


def _band_thresholds(shape, csf, viewing):
    """Повертає множники порогу 10**(−log S(f_k)) для смуг піраміди.

    Рівень k має центр PYRAMID_F0 / 2**k цикл/піксель оригіналу; рівні
    додаються, поки менша сторона рівня ≥ 8 px і центр смуги ≥ 0.25 цикл/град.
    """
    factors = []
    h, w = shape
    k = 0
    while min(h, w) >= _MIN_LEVEL_SIDE:
        f_cpd = viewing.cycles_per_px_to_cpd(PYRAMID_F0 / 2**k)
        if f_cpd < _MIN_BAND_CPD:
            break
        factors.append(10.0 ** -csf.log_sensitivity(f_cpd))
        h, w = (h + 1) // 2, (w + 1) // 2
        k += 1
    return factors


def _for_row_stripes(height, func):
    """Викликає func(start, stop) для смуг рядків висотою _STRIPE_ROWS паралельно."""
    ranges = [(a, min(a + _STRIPE_ROWS, height)) for a in range(0, height, _STRIPE_ROWS)]
    if len(ranges) == 1:
        func(*ranges[0])
        return
    for _ in _POOL.map(lambda r: func(*r), ranges):
        pass


def _threshold_band(gaussian, lowpass, factor):
    """Перетворює gaussian на смугу B = G − L↑ і обнуляє її нижче порогу контрасту.

    Поріг: |B| < max(L↑, 1e-4) · factor. Обидва масиви змінюються на місці:
    gaussian стає смугою, lowpass -- порогом.
    """

    def rows(a, b):
        band = cv2.subtract(gaussian[a:b], lowpass[a:b], dst=gaussian[a:b])
        threshold = cv2.max(lowpass[a:b], _MIN_LUMINANCE, dst=lowpass[a:b])
        cv2.multiply(threshold, factor, dst=threshold)
        invisible = cv2.compare(cv2.absdiff(band, 0.0), threshold, cv2.CMP_LT)
        cv2.subtract(band, band, dst=band, mask=invisible)

    _for_row_stripes(gaussian.shape[0], rows)
    return gaussian


def _observe_channel(linear, factors):
    """Метод Peli для одного каналу (float32, лінійна яскравість; масив змінюється)."""
    gaussian = linear
    bands = []
    for factor in factors:
        size = (gaussian.shape[1], gaussian.shape[0])
        coarser = cv2.pyrDown(gaussian)
        lowpass = cv2.pyrUp(coarser, dstsize=size)
        bands.append(_threshold_band(gaussian, lowpass, factor))
        gaussian = coarser

    result = gaussian
    for band in reversed(bands):
        result = cv2.pyrUp(result, dstsize=(band.shape[1], band.shape[0]))
        cv2.add(result, band, dst=result)
    return result


def _encode_channel(linear):
    """Те саме, що color_utils.gamma_encode (обрізання [0, 1] і LUT), але смугами в потоках.

    Насичувальне перетворення OpenCV у uint16 округлює до парного, як np.rint,
    і обрізає значення поза [0, 1].
    """
    out = np.empty(linear.shape, np.uint8)

    def rows(a, b):
        scaled = cv2.multiply(linear[a:b], _ENCODE_SCALE)
        index = cv2.add(scaled, 0.0, dtype=cv2.CV_16U)
        np.take(GAMMA_ENCODE_LUT, index, out=out[a:b])

    _for_row_stripes(linear.shape[0], rows)
    return out


def apply_observer(image_bgr: np.ndarray, csf: ObserverCSF, viewing) -> np.ndarray:
    """Пропускає зображення крізь спостерігача з CSF csf (метод Peli, 1990).

    Кожен канал у лінійному RGB розкладається на піраміду Лапласа; у смузі k
    обнуляються пікселі, де локальний контраст |B_k| / L_{k+1}↑ нижчий за поріг
    1 / S(f_k), де f_k -- центр смуги в цикл/град за умов перегляду viewing.
    Низькочастотний залишок не змінюється. Вхід uint8 BGR (h, w, 3) не змінюється.
    """
    factors = _band_thresholds(image_bgr.shape[:2], csf, viewing)
    planes = []
    for plane in cv2.split(image_bgr):
        linear = cv2.LUT(plane, GAMMA_DECODE_LUT)
        planes.append(_encode_channel(_observe_channel(linear, factors)))
    return cv2.merge(planes)
