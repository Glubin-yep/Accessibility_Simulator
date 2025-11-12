import os
import time
import cv2
import re
import numpy as np
import pytesseract
import concurrent.futures
import argparse
import functools
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from collections import defaultdict


# ГЛОБАЛЬНІ ТАБЛИЦІ (LUT) ТА МАТРИЦІ  ---
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


# --- 1. ДОПОМІЖНІ ФУНКЦІЇ ---


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


# --- 2. ЕТАП ЗБОРУ ДАНИХ ---
def take_screenshot(url, filename="screenshot_original.png"):
    """
    Використовує Selenium для створення скріншота веб-сторінки.
    """
    print(f"Збір даних: Роблю скріншот {url}...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=2560,1440")
    try:
        with webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        ) as driver:
            driver.get(url)
            driver.implicitly_wait(2)
            driver.save_screenshot(filename)
        print(f"Скріншот збережено як {filename}")
        return filename
    except Exception as e:
        print(f"Помилка під час зйомки скріншота: {e}")
        return None


# --- 3. ЕТАП СИМУЛЯЦІЇ ---
def simulate_cataract(
    image,
    blur_ksize=15,
    yellow_b_factor=0.8,
    yellow_g_factor=0.95,
    glare_weight=0.2,
):
    """
    Симулює катаракту за 3-компонентною моделлю.
    """
    # 1. Пожовтіння
    b, g, r = cv2.split(image)
    b_yellowed = cv2.multiply(b, yellow_b_factor).astype(np.uint8)
    g_yellowed = cv2.multiply(g, yellow_g_factor).astype(np.uint8)
    img_yellowed = cv2.merge([b_yellowed, g_yellowed, r])

    # 2. Blur
    blur_ksize = int(blur_ksize)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    img_blurred = cv2.GaussianBlur(img_yellowed, (blur_ksize, blur_ksize), 0)

    # 3. Відблиски
    haze = np.full_like(img_blurred, (255, 255, 255), dtype=np.uint8)
    img_glare = cv2.addWeighted(img_blurred, 1.0 - glare_weight, haze, glare_weight, 0)
    return img_glare


def simulate_achromatopsia_scotopic(image):
    """
    Симулює ахроматопсію (монохромазія паличок)
    """
    img_linear = gamma_decode(image)
    scotopic_matrix = np.array([[0.2472, 0.5820, 0.1708]])  # B, G, R
    img_gray_linear = cv2.transform(img_linear, scotopic_matrix)
    img_gray_3ch_linear = cv2.merge([img_gray_linear, img_gray_linear, img_gray_linear])
    return gamma_encode(img_gray_3ch_linear)


def simulate_dichromacy_brettel(image, sim_type="protanopia"):
    """
    Симулює дихромазію

    sim_type: "protanopia", "deuteranopia", "tritanopia"
    """
    if sim_type not in SIM_MATRICES:
        print(f"Помилка: Невідомий тип дихромазії {sim_type}")
        return image

    img_linear = gamma_decode(image)

    img_lms = cv2.transform(img_linear, MATRIX_RGB_TO_LMS)

    sim_matrix = SIM_MATRICES[sim_type]
    img_lms_sim = cv2.transform(img_lms, sim_matrix)

    img_linear_sim = cv2.transform(img_lms_sim, MATRIX_LMS_TO_RGB)

    return gamma_encode(img_linear_sim)


def simulate_metamorphopsia(image, amplitude=15, frequency=0.05):
    """
    Імітує метаморфопсію
    """
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x + amplitude * np.sin(map_y * frequency)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def simulate_central_scotoma(image, radius_percentage=0.15, blur_ksize=99):
    """
    Імітує центральну скотому.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    scotoma_radius = int(min(h, w) * radius_percentage)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize < 3:
        blur_ksize = 3

    scotoma_layer = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    mask = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), scotoma_radius, (0, 0, 0), -1)

    mask_blurred = cv2.GaussianBlur(mask, (201, 201), 0)
    mask_float = mask_blurred.astype(np.float32) / 255.0

    img_blended = image.astype(np.float32) * mask_float + scotoma_layer.astype(
        np.float32
    ) * (1.0 - mask_float)
    return img_blended.astype(np.uint8)


def simulate_tunnel_vision(image, aperture_percentage=0.5, blur_ksize=99):
    """
    Імітує тунельний зір
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    aperture_radius = int(min(h, w) * aperture_percentage)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize < 3:
        blur_ksize = 3

    blurred_layer = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), aperture_radius, (255, 255, 255), -1)

    mask_blurred = cv2.GaussianBlur(mask, (201, 201), 0)
    mask_float = mask_blurred.astype(np.float32) / 255.0

    img_blended = image.astype(np.float32) * mask_float + blurred_layer.astype(
        np.float32
    ) * (1.0 - mask_float)
    return img_blended.astype(np.uint8)


def simulate_floaters(
    image, num_floaters=20, max_size=40, min_opacity=0.2, max_opacity=0.6
):
    """
    Імітує "мушки" (флоатери) при діабетичній ретинопатії.
    """
    h, w = image.shape[:2]

    shadow_layer = np.full_like(image, 255, dtype=np.uint8)

    for _ in range(num_floaters):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size_x = np.random.randint(max_size // 4, max_size)
        size_y = np.random.randint(max_size // 4, max_size)
        angle = np.random.randint(0, 180)
        opacity = np.random.uniform(min_opacity, max_opacity)
        color_val = int(255 * (1.0 - opacity))  # 0=чорний, 255=прозорий
        cv2.ellipse(
            shadow_layer,
            (x, y),
            (size_x, size_y),
            angle,
            0,
            360,
            (color_val, color_val, color_val),
            -1,
        )

    shadow_layer = cv2.GaussianBlur(shadow_layer, (11, 11), 0)

    img_float = image.astype(np.float32) / 255.0
    shadow_float = shadow_layer.astype(np.float32) / 255.0
    result_float = cv2.multiply(img_float, shadow_float)

    return (result_float * 255.0).astype(np.uint8)


def simulate_csf_loss(image, cutoff_frequency_ratio=0.1):
    """
    Імітує втрату контрастної чутливості (CSF) через фільтрацію
    у частотній області (видалення високих частот).
    """
    filtered_channels = []
    for chan in cv2.split(image):
        # 1. Перетворення Фур'є
        dft = cv2.dft(np.float32(chan), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 2. Створення маски фільтра
        rows, cols = chan.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.float32)
        cutoff = int(min(crow, ccol) * cutoff_frequency_ratio)
        cv2.circle(mask, (ccol, crow), cutoff, (1, 1), -1)  # Низькочастотний фільтр

        # 3. Застосування маски
        fshift = dft_shift * mask

        # 4. Зворотне Перетворення Фур'є
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Нормалізація
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        filtered_channels.append(img_back.astype(np.uint8))

    return cv2.merge(filtered_channels)


def simulate_anomalous_trichromacy_machado(image, sim_type, severity=0.5):
    """
    TODO: Реалізація моделі Мачадо (2009) .

    """
    print(
        f"ПОПЕРЕДЖЕННЯ: {sim_type} (Machado, severity={severity}) не реалізовано. "
        "Повертаю оригінальне зображення."
    )
    return image


# --- 4. ЕТАП АНАЛІЗУ ---


def get_word_data(image):
    """
    Аналізує зображення, повертає словник з координатами слів
    """
    try:
        custom_config = r"--psm 11"
        data = pytesseract.image_to_data(
            image,
            lang="ukr+eng",
            config=custom_config,
            output_type=pytesseract.Output.DICT,
        )

        word_map = defaultdict(list)
        n_boxes = len(data["level"])

        for i in range(n_boxes):
            # Беремо лише слова з упевненістю > 30%
            conf = int(data["conf"][i])
            if conf < 30:
                continue

            text = data["text"][i]
            if not text or text.isspace():
                continue

            # Нормалізація (видаляємо все, крім літер, цифр та апострофів)
            normalized_word = re.sub(r"[^\w']+", "", text.lower(), re.UNICODE)

            # Враховуємо лише слова довші за 1 символ
            if len(normalized_word) > 1:
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                word_map[normalized_word].append((x, y, w, h))

        return word_map
    except Exception as e:
        print(f"Помилка Tesseract (image_to_data): {e}")
        return defaultdict(list)


# --- 5. ЕТАП ЗАПУСКУ ТА ЗВІТУВАННЯ ---


def process_simulation(
    simulation_name, simulation_func, original_image, baseline_word_data
):
    """
    Функція обробки з візуалізацією збігів та помилок.
    """
    print(f"[В роботі]: Симуляція '{simulation_name}'...")
    start_time = time.perf_counter()

    simulated_image = simulation_func(original_image)

    sim_word_data = get_word_data(simulated_image)

    sim_word_set = set(sim_word_data.keys())
    baseline_word_set = set(baseline_word_data.keys())

    common_words_set = baseline_word_set.intersection(sim_word_set)
    artifact_words_set = sim_word_set.difference(baseline_word_set)

    image_to_save = simulated_image.copy()

    # Зелені - спільні (правильно розпізнані)
    color_common = (0, 200, 0)
    for word in common_words_set:
        for x, y, w, h in sim_word_data[word]:
            cv2.rectangle(image_to_save, (x, y), (x + w, y + h), color_common, 2)

    # Червоні - артефакти (хибно розпізнані)
    color_artifact = (0, 0, 255)
    for word in artifact_words_set:
        for x, y, w, h in sim_word_data[word]:
            cv2.rectangle(image_to_save, (x, y), (x + w, y + h), color_artifact, 2)

    filename = f"screenshot_{simulation_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}.png"
    try:
        is_success, buffer = cv2.imencode(".png", image_to_save)
        if not is_success:
            raise IOError("Не вдалося закодувати зображення у формат PNG")
        with open(filename, "wb") as f:
            f.write(buffer)
    except Exception as e:
        print(f"ПОМИЛКА: Не вдалося зберегти {filename}. Деталі: {e}")
        filename = None

    end_time = time.perf_counter()
    duration = end_time - start_time

    print(
        f"[Завершено]: Симуляція '{simulation_name}'. Знайдено слів: {len(sim_word_set)}. Час: {duration:.2f} сек."
    )

    return simulation_name, sim_word_set, filename, duration


def main(args):
    URL_TO_ANALYZE = (
        "https://en.wikipedia.org/wiki/International_Organization_for_Standardization"
    )
    ORIGINAL_FILENAME = "screenshot_original.png"

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    SIMULATIONS = {
        # Катаракта (різної сили)
        "Катаракта (Легка)": functools.partial(
            simulate_cataract, blur_ksize=7, glare_weight=0.1
        ),
        "Катаракта (Сильна)": functools.partial(
            simulate_cataract, blur_ksize=21, glare_weight=0.3
        ),
        "Ахроматопсія (Скотопічна)": simulate_achromatopsia_scotopic,
        "Протанопія (Brettel)": functools.partial(
            simulate_dichromacy_brettel, sim_type="protanopia"
        ),
        "Дейтеранопія (Brettel)": functools.partial(
            simulate_dichromacy_brettel, sim_type="deuteranopia"
        ),
        "Метаморфопсія (Легка)": functools.partial(
            simulate_metamorphopsia, amplitude=8, frequency=0.03
        ),
        "Метаморфопсія (Сильна)": functools.partial(
            simulate_metamorphopsia, amplitude=25, frequency=0.07
        ),
        "Центральна Скотома (AMD)": functools.partial(
            simulate_central_scotoma, radius_percentage=0.2
        ),
        "Тунельний Зір (Глаукома)": functools.partial(
            simulate_tunnel_vision, aperture_percentage=0.3
        ),
        "Флоатери (Діабет. рет.)": functools.partial(
            simulate_floaters, num_floaters=30, max_size=50
        ),
        "Втрата контрасту (CSF)": functools.partial(
            simulate_csf_loss, cutoff_frequency_ratio=0.20
        ),
        "Протаномалія (Machado)": functools.partial(
            simulate_anomalous_trichromacy_machado, sim_type="protanomaly", severity=0.6
        ),
    }

    if not take_screenshot(URL_TO_ANALYZE, ORIGINAL_FILENAME):
        return

    original_image = cv2.imread(ORIGINAL_FILENAME)
    if original_image is None:
        print("Не вдалося завантажити оригінальний скріншот.")
        return

    print("Аналізую оригінальне зображення (базова лінія)...")
    baseline_word_data = get_word_data(original_image)
    baseline_word_set = set(baseline_word_data.keys())
    baseline_word_count = len(baseline_word_set)
    print(f"Базова лінія: {baseline_word_count} унікальних слів знайдено.\n")

    report_data = {}
    total_processing_time = 0.0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_simulation,
                name,
                func,
                original_image.copy(),
                baseline_word_data,
            )
            for name, func in SIMULATIONS.items()
        ]

        for future in concurrent.futures.as_completed(futures):

            name, word_set, filename, duration = future.result()
            report_data[name] = (
                word_set,
                filename,
                duration,
            )
            total_processing_time += duration

    print("\n" + "=" * 40)
    print(" ЗВІТ ПРО АНАЛІЗ ДОСТУПНОСТІ")
    print("=" * 40)
    print(f"Веб-сайт: {URL_TO_ANALYZE}")
    print(f"Оригінальний файл: {ORIGINAL_FILENAME}")
    print(f"Базова читабельність: {baseline_word_count} унікальних слів\n")
    print(f"Загальний час обробки: {total_processing_time:.2f} сек.\n")
    print("--- Результати Симуляцій ---")

    for name, (sim_word_set, filename, duration) in sorted(report_data.items()):

        sim_word_count = len(sim_word_set)

        common_words = baseline_word_set.intersection(sim_word_set)
        common_count = len(common_words)

        lost_words = baseline_word_set.difference(sim_word_set)
        lost_count = len(lost_words)

        artifact_words = sim_word_set.difference(baseline_word_set)
        artifact_count = len(artifact_words)

        if baseline_word_count > 0:
            overlap_percentage = (common_count / baseline_word_count) * 100

            loss_percentage = (lost_count / baseline_word_count) * 100
        else:
            overlap_percentage = 0
            loss_percentage = 0

        print(f"\nСимуляція: {name}")
        print(f"  Файл результату: {filename if filename else 'ПОМИЛКА ЗБЕРЕЖЕННЯ'}")
        print(
            f"  Всього розпізнано: {sim_word_count} (в оригіналі {baseline_word_count})"
        )
        print(f"  Збіг з оригіналом: {common_count} слів ({overlap_percentage:.1f}%)")
        print(f"  Втрачено з оригіналу: {lost_count} слів ({loss_percentage:.1f}%)")
        print(f"  Нові артефакти (помилки OCR): {artifact_count} слів")
        print(f"  Час обробки: {duration:.2f} сек.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запуск симулятора доступності вад зору."
    )
    parser.add_argument(
        "--tesseract_cmd",
        type=str,
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Шлях до tesseract.exe (якщо він не в PATH).",
    )
    args = parser.parse_args()
    main(args)
