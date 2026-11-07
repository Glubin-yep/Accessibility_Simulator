"""
Колірні перетворення для симуляції порушень кольоросприйняття (CVD).

ВАЖЛИВО про порядок каналів: OpenCV зберігає зображення як B, G, R, а всі
матриці в цьому модулі записані для R, G, B (як у літературі). Функція
apply_linear_rgb_matrix() сама переставляє рядки/стовпці матриці під BGR,
тому викликати її можна безпосередньо із зображенням з cv2.imread().
"""

import numpy as np
import cv2

from machado_matrices import MACHADO_2009

# --- ГАММА sRGB <-> LINEAR RGB ---


def _srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1.0 / 2.4) - 0.055)


# uint8 sRGB -> float32 linear (256 значень)
GAMMA_DECODE_LUT = _srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)

# float linear -> uint8 sRGB. Лінійне значення квантується з кроком 1/65535,
# цього достатньо для точного зворотного перетворення всіх 256 рівнів,
# і це значно швидше, ніж np.power на кожному пікселі.
_ENCODE_STEPS = 65535
GAMMA_ENCODE_LUT = np.round(
    _linear_to_srgb(np.arange(_ENCODE_STEPS + 1) / _ENCODE_STEPS) * 255.0
).astype(np.uint8)


def gamma_decode(img_srgb):
    """Декодує 8-бітне sRGB зображення у лінійний RGB (float32, 0..1)."""
    return GAMMA_DECODE_LUT[img_srgb]


def gamma_encode(img_linear):
    """Кодує лінійний RGB (float, 0..1) назад у 8-бітне sRGB з округленням."""
    idx = np.clip(img_linear, 0.0, 1.0) * _ENCODE_STEPS
    return GAMMA_ENCODE_LUT[np.rint(idx).astype(np.int32)]


def rgb_matrix_to_bgr(matrix):
    """
    Перетворює 3x3 матрицю, записану для (R, G, B), на еквівалентну для (B, G, R).
    P·M·P, де P — перестановка, що змінює порядок каналів на зворотний.
    """
    return np.asarray(matrix, dtype=np.float64)[::-1, ::-1]


def apply_linear_rgb_matrix(image_bgr, matrix_rgb):
    """
    Застосовує 3x3 матрицю (для лінійного RGB) до 8-бітного BGR зображення:
    гамма-декодування -> матриця -> гамма-кодування.
    """
    img_linear = gamma_decode(image_bgr)
    img_out = cv2.transform(img_linear, rgb_matrix_to_bgr(matrix_rgb))
    return gamma_encode(img_out)


# --- ПРОСТІР LMS ---

# Linear sRGB -> LMS (Hunt-Pointer-Estevez, нормалізований до D65).
MATRIX_RGB_TO_LMS = np.array(
    [
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ]
)
MATRIX_LMS_TO_RGB = np.linalg.inv(MATRIX_RGB_TO_LMS)

# Linear sRGB -> CIE XYZ (D65), IEC 61966-2-1.
MATRIX_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
MATRIX_XYZ_TO_LMS = MATRIX_RGB_TO_LMS @ np.linalg.inv(MATRIX_RGB_TO_XYZ)

# Функції узгодження кольорів CIE 1931 (2°) для опорних довжин хвиль Brettel.
CIE1931_XYZ = {
    475: (0.142100, 0.112600, 1.041900),
    485: (0.057950, 0.169300, 0.616200),
    575: (0.978600, 0.915400, 0.001800),
    660: (0.164900, 0.061000, 0.000000),
}

# Тип дихромазії -> (індекс відсутнього колбочкового каналу L/M/S, опорні хвилі)
DICHROMACY_TYPES = {
    "protanopia": (0, (475, 575)),
    "deuteranopia": (1, (475, 575)),
    "tritanopia": (2, (485, 660)),
}


# --- Viénot, Brettel & Mollon (1999): одна площина проєкції ---

VIENOT_LMS_MATRICES = {
    "protanopia": np.array([[0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]]),
    "deuteranopia": np.array([[1, 0, 0], [0.9513092, 0, 0.04866992], [0, 0, 1]]),
    "tritanopia": np.array([[1, 0, 0], [0, 1, 0], [-0.86744736, 1.86727089, 0]]),
}


def vienot_rgb_matrix(sim_type):
    """Повна матриця linear RGB -> linear RGB для методу Viénot (1999)."""
    return MATRIX_LMS_TO_RGB @ VIENOT_LMS_MATRICES[sim_type] @ MATRIX_RGB_TO_LMS


# --- Brettel, Viénot & Mollon (1997): дві напівплощини ---


def _projection_onto_plane(normal, missing):
    """
    Матриця 3x3 (у LMS), що проєктує колір уздовж осі відсутнього каналу
    `missing` на площину з нормаллю `normal`, яка проходить через початок координат.
    """
    p = np.eye(3)
    p[missing, :] = -normal / normal[missing]
    p[missing, missing] = 0.0
    return p


def _build_brettel_params(sim_type):
    missing, (wl1, wl2) = DICHROMACY_TYPES[sim_type]
    lms_white = MATRIX_RGB_TO_LMS @ np.ones(3)
    anchor1 = MATRIX_XYZ_TO_LMS @ np.array(CIE1931_XYZ[wl1])
    anchor2 = MATRIX_XYZ_TO_LMS @ np.array(CIE1931_XYZ[wl2])

    # Напівплощини, що проходять через нейтральну вісь і опорні кольори.
    proj1 = _projection_onto_plane(np.cross(lms_white, anchor1), missing)
    proj2 = _projection_onto_plane(np.cross(lms_white, anchor2), missing)

    # Площина, що розділяє дві напівплощини: містить нейтральну вісь і вісь
    # відсутнього каналу. Знак обираємо так, щоб anchor1 був з боку "+".
    axis = np.zeros(3)
    axis[missing] = 1.0
    sep_normal_lms = np.cross(lms_white, axis)
    if np.dot(anchor1, sep_normal_lms) < 0:
        sep_normal_lms = -sep_normal_lms

    rgb1 = MATRIX_LMS_TO_RGB @ proj1 @ MATRIX_RGB_TO_LMS
    rgb2 = MATRIX_LMS_TO_RGB @ proj2 @ MATRIX_RGB_TO_LMS
    # dot(LMS, n) = dot(T·RGB, n) = dot(RGB, Tᵀ·n)
    sep_normal_rgb = MATRIX_RGB_TO_LMS.T @ sep_normal_lms
    return rgb1, rgb2, sep_normal_rgb


BRETTEL_PARAMS = {name: _build_brettel_params(name) for name in DICHROMACY_TYPES}


def simulate_brettel_linear_bgr(img_linear_bgr, sim_type):
    """Brettel (1997) для лінійного BGR зображення float32."""
    rgb1, rgb2, sep_rgb = BRETTEL_PARAMS[sim_type]
    side = cv2.transform(img_linear_bgr, sep_rgb[::-1].reshape(1, 3))
    out1 = cv2.transform(img_linear_bgr, rgb_matrix_to_bgr(rgb1))
    out2 = cv2.transform(img_linear_bgr, rgb_matrix_to_bgr(rgb2))
    return np.where(side[..., None] >= 0, out1, out2)


# --- Machado, Oliveira & Fernandes (2009) ---


def machado_rgb_matrix(sim_type, severity):
    """
    Матриця Machado (2009) для linear RGB. severity ∈ [0, 1];
    між табличними значеннями (крок 0.1) — лінійна інтерполяція, як у роботі.
    """
    if sim_type not in MACHADO_2009:
        raise ValueError(
            f"Невідомий тип '{sim_type}'. Доступні: {', '.join(MACHADO_2009)}"
        )
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity має бути в діапазоні [0, 1]")
    table = np.asarray(MACHADO_2009[sim_type], dtype=np.float64)
    pos = severity * 10.0
    low = int(np.floor(pos))
    if low >= 10:
        return table[10]
    frac = pos - low
    return (1.0 - frac) * table[low] + frac * table[low + 1]
