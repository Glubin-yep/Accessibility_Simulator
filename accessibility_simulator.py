import os
import cv2
import numpy as np
import pytesseract
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


# --- 1. ЕТАП ЗБОРУ ДАНИХ  ---
def take_screenshot(url, filename="screenshot_original.png"):
    """
    Використовує Selenium для створення скріншота веб-сторінки.
    Схоже на "Модуль збору даних" зі статті[cite: 96, 97].
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


# --- 2. ЕТАП СИМУЛЯЦІЇ ---
def simulate_blur(image):
    """
    Симулює загальну розмитість зору або катаракту.
    Використовуємо GaussianBlur з OpenCV[cite: 129], але з великим ядром.
    """
    return cv2.GaussianBlur(image, (21, 21), 0)


def simulate_achromatopsia(image):
    """
    Симулює ахроматопсію (повну колірну сліпоту)
    просто перетворюючи зображення у відтінки сірого.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def simulate_protanopia(image):
    """
    Симулює протанопію (червоно-зелений дальтонізм).
    Це спрощена модель, що змішує червоний та зелений канали,
    щоб показати їх нерозрізненість.
    """
    b, g, r = cv2.split(image)

    rg_combined = cv2.addWeighted(r, 0.5, g, 0.5, 0)

    simulated_image = cv2.merge((b, rg_combined, rg_combined))
    return simulated_image


# --- 3. ЕТАП АНАЛІЗУ (адаптація зі статті) ---
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
            cv2.THRESH_BINARY_INV,  # Інвертуємо (текст стає білим)
            11,  # Розмір "сусідства" для обчислення
            2,  # Константа, яку віднімають від середнього
        )

        custom_config = r"--psm 6"
        text = pytesseract.image_to_string(thresh, lang="ukr+eng", config=custom_config)

        word_count = len(text.split())
        return word_count
    except Exception as e:
        print(f"Помилка Tesseract: {e}")
        return 0


# --- 4. ЕТАП ЗАПУСКУ ТА ЗВІТУВАННЯ ---
def process_simulation(simulation_name, simulation_func, original_image):
    """
    Єдина функція для паралельного виконання[cite: 121, 122]:
    1. Застосовує фільтр
    2. Зберігає результат
    3. Аналізує читабельність
    """
    print(f"[В роботі]: Симуляція '{simulation_name}'...")

    simulated_image = simulation_func(original_image)

    filename = f"screenshot_{simulation_name.lower().replace(' ', '_')}.png"
    cv2.imwrite(filename, simulated_image)

    word_count = analyze_readability(simulated_image)

    print(f"[Завершено]: Симуляція '{simulation_name}'. Знайдено слів: {word_count}")
    return simulation_name, word_count, filename


def main():
    URL_TO_ANALYZE = "https://uk.wikipedia.org/wiki/%D0%93%D0%BE%D0%BB%D0%BE%D0%B2%D0%BD%D0%B0_%D1%81%D1%82%D0%BE%D1%80%D1%96%D0%BD%D0%BA%D0%B0"
    ORIGINAL_FILENAME = "screenshot_original.png"

    SIMULATIONS = {
        "Розмиття (Катаракта)": simulate_blur,
        "Дальтонізм (Протанопія)": simulate_protanopia,
        "Ахроматопсія (Без кольору)": simulate_achromatopsia,
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

    # Використовуємо ThreadPoolExecutor для паралельної обробки
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_simulation, name, func, original_image)
            for name, func in SIMULATIONS.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            name, word_count, filename = future.result()
            report_data[name] = (word_count, filename)

    print("\n" + "=" * 40)
    print(" ЗВІТ ПРО АНАЛІЗ ДОСТУПНОСТІ ")
    print("=" * 40)
    print(f"Веб-сайт: {URL_TO_ANALYZE}")
    print(f"Оригінальний файл: {ORIGINAL_FILENAME}")
    print(f"Базова читабельність: {baseline_word_count} слів\n")
    print("--- Результати Симуляцій ---")

    for name, (words, filename) in report_data.items():
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
