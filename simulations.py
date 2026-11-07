"""
Функції симуляції вад зору.

Кожна функція приймає 8-бітне BGR зображення (як повертає cv2.imread)
і повертає нове 8-бітне BGR зображення того ж розміру.
"""

import cv2
import numpy as np

from color_utils import (
    DICHROMACY_TYPES,
    apply_linear_rgb_matrix,
    gamma_decode,
    gamma_encode,
    machado_rgb_matrix,
    simulate_brettel_linear_bgr,
    vienot_rgb_matrix,
)


def _odd_ksize(ksize, minimum=3):
    ksize = max(int(ksize), minimum)
    return ksize if ksize % 2 == 1 else ksize + 1


# --- ПОМУТНІННЯ КРИШТАЛИКА ---


def simulate_cataract(
    image,
    blur_ksize=15,
    yellow_b_factor=0.8,
    yellow_g_factor=0.95,
    glare_weight=0.2,
):
    """
    Симулює катаракту за 3-компонентною моделлю:
    пожовтіння (поглинання синього/зеленого), розсіювання (розмиття) та засвітка.
    """
    # 1. Пожовтіння
    b, g, r = cv2.split(image)
    b_yellowed = cv2.multiply(b, yellow_b_factor).astype(np.uint8)
    g_yellowed = cv2.multiply(g, yellow_g_factor).astype(np.uint8)
    img_yellowed = cv2.merge([b_yellowed, g_yellowed, r])

    # 2. Розсіювання світла
    blur_ksize = _odd_ksize(blur_ksize, minimum=1)
    img_blurred = cv2.GaussianBlur(img_yellowed, (blur_ksize, blur_ksize), 0)

    # 3. Засвітка (лінійна інтерполяція з білим шаром)
    haze = np.full_like(img_blurred, 255, dtype=np.uint8)
    return cv2.addWeighted(img_blurred, 1.0 - glare_weight, haze, glare_weight, 0)


# --- ПОРУШЕННЯ КОЛЬОРОСПРИЙНЯТТЯ ---


def simulate_achromatopsia_scotopic(image):
    """
    Симулює ахроматопсію (монохромазія паличок): яскравість обчислюється
    за скотопічною чутливістю у лінійному просторі.
    """
    img_linear = gamma_decode(image)
    scotopic_matrix = np.array([[0.2472, 0.5820, 0.1708]])  # B, G, R
    img_gray_linear = cv2.transform(img_linear, scotopic_matrix)
    img_gray_3ch_linear = cv2.merge([img_gray_linear] * 3)
    return gamma_encode(img_gray_3ch_linear)


def _check_dichromacy_type(sim_type):
    if sim_type not in DICHROMACY_TYPES:
        raise ValueError(
            f"Невідомий тип дихромазії '{sim_type}'. "
            f"Доступні: {', '.join(DICHROMACY_TYPES)}"
        )


def simulate_dichromacy_brettel(image, sim_type="protanopia"):
    """
    Симулює дихромазію за Brettel, Viénot & Mollon (1997):
    sRGB -> linear RGB -> LMS, проєкція вздовж осі відсутнього типу колбочок
    на одну з двох напівплощин (нейтральна вісь + опорна довжина хвилі:
    475/575 нм для протанопії і дейтеранопії, 485/660 нм для тританопії),
    LMS -> linear RGB -> sRGB.

    sim_type: "protanopia", "deuteranopia", "tritanopia"
    """
    _check_dichromacy_type(sim_type)
    img_linear = gamma_decode(image)
    return gamma_encode(simulate_brettel_linear_bgr(img_linear, sim_type))


def simulate_dichromacy_vienot(image, sim_type="protanopia"):
    """
    Симулює дихромазію за Viénot, Brettel & Mollon (1999): проєкція на ОДНУ
    площину в LMS. Швидше за Brettel (1997), але менш точне для тританопії.

    sim_type: "protanopia", "deuteranopia", "tritanopia"
    """
    _check_dichromacy_type(sim_type)
    return apply_linear_rgb_matrix(image, vienot_rgb_matrix(sim_type))


def simulate_anomalous_trichromacy_machado(image, sim_type="protanomaly", severity=0.5):
    """
    Симулює аномальну трихромазію за Machado, Oliveira & Fernandes (2009).

    sim_type: "protanomaly", "deuteranomaly", "tritanomaly"
    severity: 0.0 (норма) ... 1.0 (відповідна дихромазія)
    """
    return apply_linear_rgb_matrix(image, machado_rgb_matrix(sim_type, severity))


# --- СПОТВОРЕННЯ ТА ОБМЕЖЕННЯ ПОЛЯ ЗОРУ ---


def simulate_metamorphopsia(image, amplitude=15, frequency=0.05):
    """
    Імітує метаморфопсію: x' = x + A·sin(ω·y), y' = y.
    """
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x += np.float32(amplitude) * np.sin(map_y * np.float32(frequency))
    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def _radial_blend(image, blurred, radius, inside_sharp, feather_ksize=201):
    """
    Змішує чітке та розмите зображення за круглою маскою з м'якими краями.
    inside_sharp=True  -> чітко всередині кола (тунельний зір);
    inside_sharp=False -> розмито всередині кола (скотома).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), int(radius), 1.0, -1)
    k = _odd_ksize(feather_ksize)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    if not inside_sharp:
        mask = 1.0 - mask
    mask = mask[..., None]
    blended = image.astype(np.float32) * mask + blurred.astype(np.float32) * (
        1.0 - mask
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def simulate_central_scotoma(image, radius_percentage=0.15, blur_ksize=99):
    """
    Імітує центральну скотому (вікова макулярна дегенерація):
    розмита зона в центрі поля зору.
    """
    h, w = image.shape[:2]
    k = _odd_ksize(blur_ksize)
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    return _radial_blend(
        image, blurred, min(h, w) * radius_percentage, inside_sharp=False
    )


def simulate_tunnel_vision(image, aperture_percentage=0.5, blur_ksize=99):
    """
    Імітує тунельний зір (глаукома): чітким лишається лише центр поля зору.
    """
    h, w = image.shape[:2]
    k = _odd_ksize(blur_ksize)
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    return _radial_blend(
        image, blurred, min(h, w) * aperture_percentage, inside_sharp=True
    )


def simulate_floaters(
    image,
    num_floaters=20,
    max_size=40,
    min_opacity=0.2,
    max_opacity=0.6,
    seed=None,
):
    """
    Імітує "мушки" (флоатери) при діабетичній ретинопатії.

    seed: зерно генератора випадкових чисел. З однаковим seed результат
    повністю відтворюваний (важливо для порівняння запусків).
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    shadow_layer = np.full((h, w), 255, dtype=np.uint8)
    min_size = max(1, max_size // 4)

    for _ in range(num_floaters):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (
            int(rng.integers(min_size, max_size + 1)),
            int(rng.integers(min_size, max_size + 1)),
        )
        angle = int(rng.integers(0, 180))
        opacity = rng.uniform(min_opacity, max_opacity)
        color_val = int(round(255 * (1.0 - opacity)))  # 0 = чорний, 255 = прозорий
        cv2.ellipse(shadow_layer, center, axes, angle, 0, 360, color_val, -1)

    shadow_layer = cv2.GaussianBlur(shadow_layer, (11, 11), 0)
    shadow = shadow_layer.astype(np.float32)[..., None] / 255.0
    result = image.astype(np.float32) * shadow
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def simulate_csf_loss(image, cutoff_frequency_ratio=0.1, contrast=1.0):
    """
    Імітує втрату контрастної чутливості (CSF) фільтрацією у частотній області.

    cutoff_frequency_ratio: частота зрізу fc як частка частоти Найквіста
        (0.5 цикла/піксель). Передатна функція — гаусів низькочастотний фільтр
        H(f) = exp(-f² / (2·fc²)); на відміну від ідеального фільтра з різким
        зрізом, він не дає кільцевих артефактів (ефект Гіббса).
    contrast: множник контрасту відносно середньої яскравості кожного каналу
        (1.0 — без змін, 0.5 — контраст удвічі менший).

    За теоремою про згортку множення спектра на H(f) еквівалентне згортці
    із гаусовим ядром σ = 1 / (2π·fc) пікселів, тому фільтр застосовується
    у просторовій області: це значно швидше за ДПФ і не "загортає" краї
    зображення (ДПФ виконує циклічну згортку).

    Середня яскравість (нульова частота) зберігається, значення не
    розтягуються до 0..255, тож контраст ніколи не зростає.
    """
    if cutoff_frequency_ratio <= 0:
        raise ValueError("cutoff_frequency_ratio має бути > 0")

    fc = 0.5 * cutoff_frequency_ratio  # цикли на піксель
    sigma = 1.0 / (2.0 * np.pi * fc)
    filtered = cv2.GaussianBlur(
        image.astype(np.float32), (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
    )

    # Зниження контрасту навколо середнього значення
    if contrast != 1.0:
        mean = filtered.mean(axis=(0, 1), keepdims=True)
        filtered = mean + (filtered - mean) * contrast

    return np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
