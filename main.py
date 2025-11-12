import argparse
import cv2
import pytesseract
import concurrent.futures
import config
from web_scraper import take_screenshot
from analysis import get_word_data
from reporting import process_simulation, generate_readability_report_table


def main(args):
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    if not take_screenshot(config.URL_TO_ANALYZE, config.ORIGINAL_FILENAME):
        return

    original_image = cv2.imread(config.ORIGINAL_FILENAME)
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
            for name, func in config.SIMULATIONS.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            name, word_set, filename, duration = future.result()
            report_data[name] = (
                word_set,
                filename,
                duration,
            )
            total_processing_time += duration

    # --- БЛОК: ГЕНЕРАЦІЯ ТАБЛИЦЬ-ЗВІТІВ ---
    print("\n" + "=" * 40)
    print(" ГЕНЕРАЦІЯ ЗОБРАЖЕНЬ-ЗВІТІВ")
    print("=" * 40)
    if report_data:
        generate_readability_report_table(
            report_data, baseline_word_set, filename="screenshot_report_readability.png"
        )
    else:
        print("Немає даних для генерації звітів.")
    # --- КІНЕЦЬ БЛОКУ ---

    print("\n" + "=" * 40)
    print(" ЗВІТ ПРО АНАЛІЗ ДОСТУПНОСТІ (КОНСОЛЬ)")
    print("=" * 40)
    print(f"Веб-сайт: {config.URL_TO_ANALYZE}")
    print(f"Оригінальний файл: {config.ORIGINAL_FILENAME}")
    print(f"Базова читабельність: {baseline_word_count} унікальних слів\n")
    print(f"Загальний час обробки: {total_processing_time:.2f} сек.\n")
    print(
        f"Згенеровано таблиці: 'screenshot_report_readability.png' та 'screenshot_report_performance.png'\n"
    )
    print("--- Результати Симуляцій (Детально) ---")

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
            f"  Візуалізація: Зелені рамки = збіг, Червоні рамки = хибно розпізнані (артефакти)"
        )
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
