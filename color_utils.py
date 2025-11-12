import numpy as np
import cv2

# --- ГЛОБАЛЬНІ ТАБЛИЦІ (LUT) ТА МАТРИЦІ ---


def _build_gamma_decode_lut():
    """
    Створює таблицю пошуку (LUT) для швидкого гамма-декодування sRGB -> Linear RGB.
    """
    lut = np.empty((1, 256), np.float32)
    for i in range(256):
        val = i / 255.0
        if val <= 0.04045:
            lut[0, i] = val / 12.92
        else:
            lut[0, i] = ((val + 0.055) / 1.055) ** 2.4
    return lut


GAMMA_DECODE_LUT = _build_gamma_decode_lut()

# --- Глобальні матриці для симуляції CVD ---
MATRIX_RGB_TO_LMS = np.array(
    [
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ]
)

MATRIX_LMS_TO_RGB = np.array(
    [
        [5.47221206, -4.6419601, 0.16963708],
        [-1.1252419, 2.29317094, -0.1678952],
        [0.02980165, -0.19318073, 1.16364789],
    ]
)

# Матриці симуляції дихромазії
SIM_MATRICES = {
    "protanopia": np.array([[0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]]),  # S_p
    "deuteranopia": np.array([[1, 0, 0], [0.9513092, 0, 0.04866992], [0, 0, 1]]),  # S_d
    "tritanopia": np.array([[1, 0, 0], [0, 1, 0], [-0.86744736, 1.86727089, 0]]),  # S_t
}

# --- ДОПОМІЖНІ ФУНКЦІЇ ---


def gamma_decode(img_srgb):
    """
    Декодує 8-бітне sRGB зображення у лінійний RGB
    """
    return cv2.LUT(img_srgb, GAMMA_DECODE_LUT)


def gamma_encode(img_linear):
    """
    Кодує лінійний RGB назад у 8-бітне sRGB.
    """
    img_linear = np.clip(img_linear, 0.0, 1.0)
    mask = img_linear <= 0.0031308
    img_srgb = np.zeros_like(img_linear)
    img_srgb[mask] = img_linear[mask] * 12.92
    img_srgb[~mask] = 1.055 * np.power(img_linear[~mask], 1.0 / 2.4) - 0.055
    return (img_srgb * 255.0).astype(np.uint8)
