import os
import cv2
import numpy as np
import pytesseract
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


# --- 1. ДОПОМІЖНІ ФУНКЦІЇ  ---
def gamma_decode(img_srgb):
    """
    Декодує 8-бітне sRGB зображення у лінійний RGB
    """
    img_linear = img_srgb.astype(np.float32) / 255.0
    mask = img_linear <= 0.04045

    img_linear_out = np.zeros_like(img_linear)

    img_linear_out[mask] = img_linear[mask] / 12.92
    img_linear_out[~mask] = np.power((img_linear[~mask] + 0.055) / 1.055, 2.4)

    return img_linear_out


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
    options.add_argument("--window-size=1920,1080")

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
def simulate_cataract(image):
    """
    ЗАМІНА для GaussianBlur.
    Симулює катаракту за допомогою 3-компонентної моделі з дослідження
    """
    # 1. Пожовтіння (маніпуляція каналами)
    b, g, r = cv2.split(image)
    # Зменшуємо синій та зелений  канали
    b_yellowed = cv2.multiply(b, 0.8).astype(np.uint8)
    g_yellowed = cv2.multiply(g, 0.95).astype(np.uint8)
    img_yellowed = cv2.merge([b_yellowed, g_yellowed, r])

    # 2. Blur
    img_blurred = cv2.GaussianBlur(img_yellowed, (15, 15), 0)

    # 3. Відблиски (Glare / Whitening)
    haze = np.full_like(img_blurred, (255, 255, 255), dtype=np.uint8)

    # 80% зображення, 20% відблисків
    img_glare = cv2.addWeighted(img_blurred, 0.8, haze, 0.2, 0)

    return img_glare


def simulate_achromatopsia_scotopic(image):
    """
    ЗАМІНА для cvtColor(BGR2GRAY).
    Симулює ахроматопсію (монохромазія паличок)
    """
    img_linear = gamma_decode(image)

    scotopic_matrix = np.array([[0.2472, 0.5820, 0.1708]])  # B, G, R

    img_gray_linear = cv2.transform(img_linear, scotopic_matrix)

    img_gray_3ch_linear = cv2.merge([img_gray_linear, img_gray_linear, img_gray_linear])

    return gamma_encode(img_gray_3ch_linear)


def simulate_protanopia_brettel(image):
    """
    ЗАМІНА для наївного addWeighted.
    Симулює протанопію (дихромазія) за моделлю Brettel/Viénot
    """
    # 1. sRGB -> Лінійний RGB
    img_linear = gamma_decode(image)

    # 2. Лінійний RGB -> LMS
    T = np.array(
        [
            [0.31399022, 0.63951294, 0.04649755],
            [0.15537241, 0.75789446, 0.08670142],
            [0.01775239, 0.10944209, 0.87256922],
        ]
    )
    img_lms = cv2.transform(img_linear, T)

    # 3. LMS -> LMS' (симуляція протанопії)
    S_p = np.array([[0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]])
    img_lms_sim = cv2.transform(img_lms, S_p)

    # 4. LMS' -> Лінійний RGB'
    T_inv = np.array(
        [
            [5.47221206, -4.6419601, 0.16963708],
            [-1.1252419, 2.29317094, -0.1678952],
            [0.02980165, -0.19318073, 1.16364789],
        ]
    )
    img_linear_sim = cv2.transform(img_lms_sim, T_inv)

    # 5. Лінійний RGB' -> sRGB
    return gamma_encode(img_linear_sim)


def simulate_metamorphopsia(image):
    """
    НОВА СИМУЛЯЦІЯ (Розділ 5).
    Імітує метаморфопсію (спотворення, "хвилясті лінії")
    """
    h, w = image.shape[:2]

    # 1. Створюємо базові карти координат
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    # 2. Вводимо синусоїдальне спотворення
    amplitude = 15
    frequency = 0.05

    # Спотворюємо x-координати на основі y-позиції
    map_x = map_x + amplitude * np.sin(map_y * frequency)

    # 3. Конвертуємо карти у float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 4. Застосовуємо деформацію
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def simulate_central_scotoma(image):
    """
    НОВА СИМУЛЯЦІЯ (Алгоритм 7).
    Імітує центральну скотому (втрата центрального зору, AMD)
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Радіус сліпої плями
    scotoma_radius = int(min(h, w) * 0.15)

    # 1. Створюємо "шар скотоми" (напр., сильно розмите зображення)
    scotoma_layer = cv2.GaussianBlur(image, (99, 99), 0)

    # 2. Створюємо градієнтну маску
    mask = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), scotoma_radius, (0, 0, 0), -1)

    # 3. Розмиваємо маску для м'яких країв
    mask_blurred = cv2.GaussianBlur(mask, (201, 201), 0)
    mask_float = mask_blurred.astype(np.float32) / 255.0

    # 4. Змішуємо зображення
    img_blended = image.astype(np.float32) * mask_float + scotoma_layer.astype(
        np.float32
    ) * (1.0 - mask_float)

    return img_blended.astype(np.uint8)


def simulate_tunnel_vision(image):
    """
    НОВА СИМУЛЯЦІЯ (Алгоритм 6).
    Імітує тунельний зір (втрата периферичного зору, Глаукома)
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Радіус чіткої "апертури"
    aperture_radius = int(min(h, w) * 0.5)

    # 1. Створюємо "шар периферії" (розмите зображення)
    blurred_layer = cv2.GaussianBlur(image, (99, 99), 0)

    # 2. Створюємо градієнтну маску
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.circle(mask, (center_x, center_y), aperture_radius, (255, 255, 255), -1)

    # 3. Розмиваємо маску для м'яких країв
    mask_blurred = cv2.GaussianBlur(mask, (201, 201), 0)
    mask_float = mask_blurred.astype(np.float32) / 255.0

    # 4. Змішуємо зображення [cite: 480]
    # (mask_float = 1.0 -> оригінал, mask_float = 0.0 -> периферія)
    img_blended = image.astype(np.float32) * mask_float + blurred_layer.astype(
        np.float32
    ) * (1.0 - mask_float)

    return img_blended.astype(np.uint8)


# --- 4. ЕТАП АНАЛІЗУ ---
def analyze_readability(image):
    """
    Аналізує, скільки слів може розпізнати Tesseract,
    використовуючи КРАЩУ обробку (адаптивний поріг).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        custom_config = r"--psm 6"
        text = pytesseract.image_to_string(thresh, lang="ukr+eng", config=custom_config)

        word_count = len(text.split())
        return word_count
    except Exception as e:
        print(f"Помилка Tesseract: {e}")
        return 0


# --- 5. ЕТАП ЗАПУСКУ ТА ЗВІТУВАННЯ ---
def process_simulation(simulation_name, simulation_func, original_image):

    print(f"[В роботі]: Симуляція '{simulation_name}'...")

    simulated_image = simulation_func(original_image)

    filename = f"screenshot_{simulation_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"

    try:
        is_success, buffer = cv2.imencode(".png", simulated_image)
        if not is_success:
            raise IOError("Не вдалося закодувати зображення у формат PNG")

        with open(filename, "wb") as f:
            f.write(buffer)

    except Exception as e:
        print(f"ПОМИЛКА: Не вдалося зберегти {filename}. Деталі: {e}")
        filename = None

    word_count = analyze_readability(simulated_image)

    print(f"[Завершено]: Симуляція '{simulation_name}'. Знайдено слів: {word_count}")
    return simulation_name, word_count, filename


def main():
    URL_TO_ANALYZE = "https://uk.wikipedia.org/wiki/%D0%93%D0%BE%D0%BB%D0%BE%D0%B2%D0%BD%D0%B0_%D1%81%D1%82%D0%BE%D1%80%D1%96%D0%BD%D0%BA%D0%B0"
    ORIGINAL_FILENAME = "screenshot_original.png"

    SIMULATIONS = {
        "Катаракта (Композитна модель)": simulate_cataract,
        "Ахроматопсія (Скотопічна)": simulate_achromatopsia_scotopic,
        "Протанопія (Brettel модель)": simulate_protanopia_brettel,
        "Метаморфопсія (Хвилі)": simulate_metamorphopsia,
        "Центральна Скотома (AMD)": simulate_central_scotoma,
        "Тунельний Зір (Глаукома)": simulate_tunnel_vision,
    }

    if not take_screenshot(URL_TO_ANALYZE, ORIGINAL_FILENAME):
        return

    original_image = cv2.imread(ORIGINAL_FILENAME)
    if original_image is None:
        print("Не вдалося завантажити оригінальний скріншот.")
        return

    print("Аналізую оригінальне зображення (базова лінія)...")
    baseline_word_count = analyze_readability(original_image)
    print(f"Базова лінія: {baseline_word_count} слів знайдено.\n")

    report_data = {}

    # ThreadPoolExecutor для паралельної обробки
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_simulation, name, func, original_image.copy())
            for name, func in SIMULATIONS.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            name, word_count, filename = future.result()
            report_data[name] = (word_count, filename)

    print("\n" + "=" * 40)
    print(" ЗВІТ ПРО АНАЛІЗ ДОСТУПНОСТІ (v2 - Науковий) ")
    print("=" * 40)
    print(f"Веб-сайт: {URL_TO_ANALYZE}")
    print(f"Оригінальний файл: {ORIGINAL_FILENAME}")
    print(f"Базова читабельність: {baseline_word_count} слів\n")
    print("--- Результати Симуляцій ---")

    for name, (words, filename) in sorted(report_data.items()):
        if baseline_word_count > 0:
            drop_percentage = (
                (baseline_word_count - words) / baseline_word_count
            ) * 100
        else:
            drop_percentage = 0

        print(f"\nСимуляція: {name}")
        print(f"  Файл результату: {filename}")
        print(f"  Розпізнано слів: {words}")
        print(f"  Втрата читабельності: {drop_percentage:.1f}%")


if __name__ == "__main__":

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    main()
