import argparse
import concurrent.futures
import hashlib
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import pytesseract

import config
from analysis import TESSERACT_LANG, compute_metrics, get_word_data
from observer import NORMAL_OBSERVER, apply_observer
from reporting import generate_readability_report_table, process_simulation, save_image
from severity import MODELS
from sweep import run_sweep
from sweep_report import format_sweep_table, plot_sweep, sweep_to_dict, write_sweep_json
from viewing import ViewingConditions
from web_scraper import take_screenshot

WINDOWS_DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EXIT_OK = 0
EXIT_QUALITY_GATE_FAILED = 1
EXIT_ERROR = 2


def configure_tesseract(tesseract_cmd):
    """
    Шлях до Tesseract: явно заданий аргументом -> з PATH -> стандартний шлях Windows.
    Повертає True, якщо Tesseract і потрібні мовні пакети доступні.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elif shutil.which("tesseract") is None and os.path.exists(WINDOWS_DEFAULT_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = WINDOWS_DEFAULT_TESSERACT

    try:
        version = pytesseract.get_tesseract_version()
        languages = set(pytesseract.get_languages(config=""))
    except (pytesseract.TesseractNotFoundError, OSError) as e:
        print(f"ПОМИЛКА: Tesseract не знайдено ({e}). Вкажіть шлях через --tesseract_cmd.")
        return False

    missing = set(TESSERACT_LANG.split("+")) - languages
    if missing:
        print(f"ПОМИЛКА: Не встановлено мовні пакети Tesseract: {', '.join(sorted(missing))}.")
        return False
    print(f"Tesseract {version}, мови: {TESSERACT_LANG}")
    return True


def url_to_dirname(url):
    """Безпечна назва підпапки для URL."""
    parsed = urlparse(url)
    readable = re.sub(r"[^\w]+", "_", unquote(parsed.netloc + parsed.path)).strip("_")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:60]}_{digest}"


def load_image(path):
    """cv2.imread не відкриває не-ASCII шляхи на Windows, тому читаємо через буфер."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def positive_float(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Очікується число, отримано '{value}'")
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Значення має бути додатним скінченним числом")
    return number


def viewing_from_args(args):
    return ViewingConditions(
        distance_cm=args.distance_cm,
        ppi=args.ppi,
        dpr=args.dpr,
        viewport_height_px=args.viewport[1],
    )


def prepare_baseline(url, output_dir, args, viewing):
    """
    Скріншот -> нормальний спостерігач -> OCR еталона.
    Повертає (оригінальне зображення, word_data еталона) або None у разі помилки.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_path = os.path.join(output_dir, config.ORIGINAL_FILENAME)

    if not take_screenshot(
        url,
        original_path,
        viewport=args.viewport,
        full_page=not args.viewport_only,
        max_height=config.MAX_PAGE_HEIGHT,
    ):
        return None

    original_image = load_image(original_path)
    if original_image is None:
        print("Не вдалося завантажити оригінальний скріншот.")
        return None

    if NORMAL_OBSERVER.cutoff_cpd > viewing.nyquist_cpd:
        print(
            f"Довідка: частота зрізу нормального спостерігача ({NORMAL_OBSERVER.cutoff_cpd:g} цикл/град) "
            f"вища за межу Найквіста екрана ({viewing.nyquist_cpd:.1f} цикл/град), тому легке "
            "зниження гостроти не впливатиме на результат. Це очікувана поведінка, не помилка."
        )

    print("Аналізую оригінальне зображення (базова лінія)...")
    baseline_word_data = get_word_data(apply_observer(original_image, NORMAL_OBSERVER, viewing))
    baseline_word_count = len(baseline_word_data)
    print(f"Базова лінія: {baseline_word_count} унікальних слів знайдено.\n")
    if baseline_word_count == 0:
        print("ПОМИЛКА: На оригінальному зображенні не розпізнано жодного слова.")
        return None
    return original_image, baseline_word_data


def analyze_url(url, output_dir, args):
    """
    Повний цикл аналізу однієї сторінки.
    Повертає список (назва симуляції, ILR) або None у разі помилки.
    """
    viewing = viewing_from_args(args)
    prepared = prepare_baseline(url, output_dir, args, viewing)
    if prepared is None:
        return None
    original_image, baseline_word_data = prepared
    original_path = os.path.join(output_dir, config.ORIGINAL_FILENAME)
    baseline_word_set = set(baseline_word_data.keys())
    baseline_word_count = len(baseline_word_set)
    max_workers = args.workers or config.DEFAULT_WORKERS
    simulations = config.build_simulations(viewing)

    wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_simulation,
                name,
                func,
                original_image.copy(),
                baseline_word_data,
                output_dir,
            )
            for name, func in simulations.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    wall_time = time.perf_counter() - wall_start

    print("\n" + "=" * 40)
    print(" ГЕНЕРАЦІЯ ЗОБРАЖЕННЯ-ЗВІТУ")
    print("=" * 40)
    report_path = os.path.join(output_dir, config.REPORT_FILENAME)
    generate_readability_report_table(results, baseline_word_set, filename=report_path)

    print("\n" + "=" * 40)
    print(" ЗВІТ ПРО АНАЛІЗ ДОСТУПНОСТІ (КОНСОЛЬ)")
    print("=" * 40)
    print(f"Веб-сайт: {url}")
    print(f"Оригінальний файл: {original_path}")
    print(f"Базова читабельність: {baseline_word_count} унікальних слів")
    print(f"Загальний час обробки симуляцій (реальний): {wall_time:.2f} сек.")
    if max_workers != 1:
        print(
            "  Примітка: симуляції виконувались паралельно, тому час окремих симуляцій\n"
            "  нижче завищений через конкуренцію за CPU. Для точних вимірювань: --workers 1"
        )
    print(f"Таблиця-звіт: {report_path}\n")
    print("--- Результати Симуляцій (Детально) ---")
    print("Легенда рамок: зелені = збіг, помаранчеві = втрачені, червоні = артефакти OCR")

    ilr_by_simulation = []
    for result in sorted(results, key=lambda r: r.name):
        m = compute_metrics(baseline_word_set, result.word_set)
        ilr_by_simulation.append((result.name, m.information_loss_ratio))
        print(f"\nСимуляція: {result.name}")
        print(f"  Файл результату: {result.filename or 'ПОМИЛКА ЗБЕРЕЖЕННЯ'}")
        print(f"  Всього розпізнано: {m.sim_count} (в оригіналі {m.baseline_count})")
        print(f"  Збіг з оригіналом: {m.common_count} слів ({m.overlap_ratio * 100:.1f}%)")
        print(
            f"  Втрачено (ILR): {m.lost_count} слів ({m.information_loss_ratio * 100:.1f}%)"
        )
        print(f"  Нові артефакти (помилки OCR): {m.artifact_count} слів")
        print(
            f"  Час: симуляція {result.simulation_time:.2f} с, OCR {result.ocr_time:.2f} с"
        )
    return ilr_by_simulation


def sweep_url(url, output_dir, args, viewing):
    """
    Режим --sweep для однієї сторінки: еталон -> прогін моделей по шкалі -> sweep.json, PNG, таблиця.
    Повертає False, якщо еталон не підготовлено або будь-яка модель завершилась помилкою.
    """
    prepared = prepare_baseline(url, output_dir, args, viewing)
    if prepared is None:
        return False
    original_image, baseline_word_data = prepared
    baseline_word_set = set(baseline_word_data.keys())
    sweep_dir = os.path.join(output_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    image_sink = None
    if args.save_sweep_images:
        images_dir = os.path.join(sweep_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        def image_sink(model_key, s, image):
            save_image(os.path.join(images_dir, f"{model_key}_s{s:.2f}.png"), image)

    onset = args.onset_ilr / 100
    breaking = args.breaking_ilr / 100
    models = list(MODELS.values())
    print(f"Прогін {len(models)} моделей по шкалі ({args.sweep_steps} кроків)...")
    wall_start = time.perf_counter()
    curves = run_sweep(
        original_image,
        baseline_word_set,
        models,
        viewing,
        steps=args.sweep_steps,
        onset_ilr=onset,
        breaking_ilr=breaking,
        workers=args.workers or config.DEFAULT_WORKERS,
        ocr=get_word_data,
        image_sink=image_sink,
    )
    total_s = time.perf_counter() - wall_start

    data = sweep_to_dict(
        url,
        viewing,
        len(baseline_word_set),
        onset,
        breaking,
        curves,
        MODELS,
        total_s,
        datetime.now().isoformat(timespec="seconds"),
    )
    json_path = os.path.join(sweep_dir, "sweep.json")
    png_path = os.path.join(sweep_dir, "sweep_ilr.png")
    write_sweep_json(json_path, data)
    plot_sweep(png_path, curves, MODELS, onset, breaking)

    print("\n" + format_sweep_table(curves, MODELS))
    print(f"\nЧас прогону: {total_s:.2f} сек.")
    print(f"JSON: {json_path}")
    print(f"Графік: {png_path}")

    failed = [c.model_key for c in curves if c.failed]
    if failed:
        print(f"ПОМИЛКА: не вдалося обчислити моделі: {', '.join(failed)}.")
        return False
    return True


def parse_viewport(value):
    match = re.fullmatch(r"(\d+)[xX×](\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("Формат: ШИРИНАxВИСОТА, наприклад 2560x1440")
    return int(match.group(1)), int(match.group(2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Запуск симулятора доступності вад зору."
    )
    parser.add_argument(
        "--url",
        action="append",
        help="URL сторінки для аналізу (можна вказати кілька разів). "
        "За замовчуванням — URL_TO_ANALYZE з config.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help="Папка для результатів (default: %(default)s).",
    )
    parser.add_argument(
        "--tesseract_cmd",
        type=str,
        default=None,
        help="Шлях до виконуваного файлу tesseract (якщо він не в PATH).",
    )
    parser.add_argument(
        "--viewport",
        type=parse_viewport,
        default=config.VIEWPORT,
        help="Розмір області перегляду, напр. 2560x1440 (default: %(default)s).",
    )
    parser.add_argument(
        "--viewport-only",
        action="store_true",
        default=not config.FULL_PAGE,
        help="Знімати лише видиму частину сторінки, а не всю сторінку.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Кількість паралельних потоків (default: %d; 1 — послідовно, для точних замірів часу)."
        % config.DEFAULT_WORKERS,
    )
    parser.add_argument(
        "--distance-cm",
        type=positive_float,
        default=ViewingConditions.distance_cm,
        help="Відстань від очей до екрана, см (default: %(default)s).",
    )
    parser.add_argument(
        "--ppi",
        type=positive_float,
        default=ViewingConditions.ppi,
        help="Щільність пікселів екрана (default: %(default)s).",
    )
    parser.add_argument(
        "--dpr",
        type=positive_float,
        default=ViewingConditions.dpr,
        help="Device pixel ratio (default: %(default)s).",
    )
    parser.add_argument(
        "--max-ilr",
        type=float,
        default=None,
        help="Поріг якості для CI/CD: якщо ILR будь-якої симуляції перевищує "
        "це значення (у відсотках), програма завершується з кодом 1.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Режим прогону моделей вад зору по шкалі тяжкості замість звичайного аналізу.",
    )
    parser.add_argument(
        "--sweep-steps",
        type=int,
        default=10,
        help="Кількість точок сітки шкали, >= 2 (default: %(default)s).",
    )
    parser.add_argument(
        "--onset-ilr",
        type=float,
        default=5.0,
        help="Поріг початку деградації, ILR у відсотках (default: %(default)s).",
    )
    parser.add_argument(
        "--breaking-ilr",
        type=float,
        default=50.0,
        help="Точка руйнування, ILR у відсотках (default: %(default)s).",
    )
    parser.add_argument(
        "--save-sweep-images",
        action="store_true",
        help="Зберігати PNG для кожної точки шкали (лише з --sweep).",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sweep_steps < 2:
        parser.error("--sweep-steps має бути не менше 2")
    if not (0 < args.onset_ilr < args.breaking_ilr <= 100):
        parser.error("потрібно 0 < --onset-ilr < --breaking-ilr <= 100")
    if args.sweep and args.max_ilr is not None:
        print("Попередження: --max-ilr ігнорується в режимі --sweep.")

    if not configure_tesseract(args.tesseract_cmd):
        return EXIT_ERROR

    urls = args.url or [config.URL_TO_ANALYZE]
    exit_code = EXIT_OK
    violations = []

    for url in urls:
        output_dir = (
            args.output_dir
            if len(urls) == 1
            else os.path.join(args.output_dir, url_to_dirname(url))
        )
        if args.sweep:
            if not sweep_url(url, output_dir, args, viewing_from_args(args)):
                exit_code = EXIT_ERROR
            continue
        ilr_by_simulation = analyze_url(url, output_dir, args)
        if ilr_by_simulation is None:
            exit_code = EXIT_ERROR
            continue
        if args.max_ilr is not None:
            violations += [
                (url, name, ilr)
                for name, ilr in ilr_by_simulation
                if ilr * 100 > args.max_ilr
            ]

    if args.max_ilr is not None:
        print("\n" + "=" * 40)
        print(f" ПОРІГ ЯКОСТІ: ILR <= {args.max_ilr:.1f}%")
        print("=" * 40)
        if violations:
            for url, name, ilr in violations:
                print(f"  ПЕРЕВИЩЕНО: {name}: ILR {ilr * 100:.1f}% ({url})")
            if exit_code == EXIT_OK:
                exit_code = EXIT_QUALITY_GATE_FAILED
        else:
            print("  Усі симуляції в межах порогу.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
