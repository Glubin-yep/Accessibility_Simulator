import os
import re
import time
from dataclasses import dataclass

import cv2
from PIL import Image, ImageDraw, ImageFont

from analysis import compute_metrics, get_word_data

# Кольори рамок (BGR)
COLOR_COMMON = (0, 200, 0)  # зелений  — слово збереглося
COLOR_LOST = (0, 165, 255)  # помаранчевий — слово втрачено (позиція з оригіналу)
COLOR_ARTIFACT = (0, 0, 255)  # червоний — хибно розпізнане слово (артефакт)
TABLE_ERROR_COLOR = (200, 0, 0)  # RGB, рядок таблиці для симуляції з помилкою

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


@dataclass
class SimulationResult:
    name: str
    word_set: set
    filename: str
    simulation_time: float  # лише сама симуляція
    ocr_time: float  # розпізнавання тексту на симульованому зображенні
    error: str | None = None  # опис винятку, якщо симуляція або OCR не вдалися

    @property
    def failed(self):
        return self.error is not None

    @property
    def total_time(self):
        return self.simulation_time + self.ocr_time


def _get_font(font_size=16):
    """Завантажує .ttf шрифт із підтримкою кирилиці."""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except OSError as e:
                print(f"ПОПЕРЕДЖЕННЯ: Не вдалося завантажити шрифт {path}: {e}")
    print(
        "ПОПЕРЕДЖЕННЯ: Не знайдено шрифтів Arial або DejaVuSans. "
        "Кирилиця у таблиці може відображатися некоректно."
    )
    return ImageFont.load_default()


def simulation_filename(simulation_name):
    """'Флоатери (Діабет. рет.)' -> 'screenshot_флоатери_діабет_рет.png'"""
    slug = re.sub(r"[^\w]+", "_", simulation_name.lower()).strip("_")
    return f"screenshot_{slug}.png"


def save_image(path, image):
    """Зберігає зображення (підтримує не-ASCII шляхи на Windows)."""
    is_success, buffer = cv2.imencode(".png", image)
    if not is_success:
        raise IOError("Не вдалося закодувати зображення у формат PNG")
    with open(path, "wb") as f:
        f.write(buffer)


def generate_readability_report_table(
    results, baseline_word_set, filename="screenshot_report_readability.png"
):
    """
    Генерує зображення-таблицю зі статистикою читабельності (Pillow).
    results: список SimulationResult.
    """
    print(f"Генерую таблицю звіту читабельності: {filename}...")

    font = _get_font(font_size=15)
    row_height = 30
    padding = 10
    col_widths = [260, 90, 90, 100, 110, 110]
    img_width = sum(col_widths) + 2 * padding

    baseline_count = len(baseline_word_set)
    table_data = [
        ("Симуляція", "Всього", "Збіг %", "ILR %", "Артефакти", "Час, с"),
        ("Оригінал (База)", f"{baseline_count}", "100.0%", "0.0%", "0", "—"),
    ]
    errors = set()
    for result in sorted(results, key=lambda r: r.name):
        if result.failed:
            errors.add(len(table_data))
            table_data.append(
                (result.name, "ПОМИЛКА", "—", "—", "—", f"{result.total_time:.2f}")
            )
            continue
        m = compute_metrics(baseline_word_set, result.word_set)
        table_data.append(
            (
                result.name,
                f"{m.sim_count}",
                f"{m.overlap_ratio * 100:.1f}%",
                f"{m.information_loss_ratio * 100:.1f}%",
                f"{m.artifact_count}",
                f"{result.total_time:.2f}",
            )
        )

    img_height = row_height * len(table_data) + 2 * padding
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    for row_index, row in enumerate(table_data):
        top = padding + row_index * row_height
        if row_index == 0:
            draw.rectangle(
                [(padding // 2, top), (img_width - padding // 2, top + row_height)],
                fill=(235, 235, 235),
            )
        x_offset = padding
        for i, cell in enumerate(row):
            draw.text(
                (x_offset, top + row_height // 2),
                str(cell),
                fill=TABLE_ERROR_COLOR if row_index in errors else "black",
                font=font,
                anchor="lm",
            )
            x_offset += col_widths[i]
        draw.line(
            [(padding // 2, top + row_height), (img_width - padding // 2, top + row_height)],
            fill=(200, 200, 200),
            width=1,
        )

    image.save(filename)
    print(f"[Успішно] Таблицю читабельності збережено як {filename}")


def process_simulation(
    simulation_name, simulation_func, original_image, baseline_word_data, output_dir="."
):
    """
    Застосовує симуляцію, розпізнає текст і зберігає зображення з рамками:
    зелені — збіг з оригіналом, помаранчеві — втрачені слова (позиції з оригіналу),
    червоні — артефакти розпізнавання.

    Виняток у симуляції або OCR не зупиняє прогін: повертається результат
    з error і порожнім набором слів, решта симуляцій продовжується.
    """
    print(f"[В роботі]: Симуляція '{simulation_name}'...")

    simulation_time = ocr_time = 0.0
    stage = "симуляція"
    start = time.perf_counter()
    try:
        simulated_image = simulation_func(original_image)
        simulation_time = time.perf_counter() - start
        stage = "OCR"
        start = time.perf_counter()
        sim_word_data = get_word_data(simulated_image)
        ocr_time = time.perf_counter() - start
    except Exception as e:
        elapsed = time.perf_counter() - start
        if stage == "OCR":
            ocr_time = elapsed
        else:
            simulation_time = elapsed
        error = f"{stage}: {type(e).__name__}: {e}"
        print(f"ПОМИЛКА: Симуляція '{simulation_name}' не виконана ({error}).")
        return SimulationResult(
            simulation_name, set(), None, simulation_time, ocr_time, error
        )

    sim_word_set = set(sim_word_data.keys())
    baseline_word_set = set(baseline_word_data.keys())

    image_to_save = simulated_image.copy()
    groups = (
        (baseline_word_set & sim_word_set, sim_word_data, COLOR_COMMON),
        (baseline_word_set - sim_word_set, baseline_word_data, COLOR_LOST),
        (sim_word_set - baseline_word_set, sim_word_data, COLOR_ARTIFACT),
    )
    for words, boxes_by_word, color in groups:
        for word in words:
            for x, y, w, h in boxes_by_word[word]:
                cv2.rectangle(image_to_save, (x, y), (x + w, y + h), color, 2)

    filename = os.path.join(output_dir, simulation_filename(simulation_name))
    try:
        save_image(filename, image_to_save)
    except Exception as e:
        print(f"ПОМИЛКА: Не вдалося зберегти {filename}. Деталі: {e}")
        filename = None

    print(
        f"[Завершено]: Симуляція '{simulation_name}'. Знайдено слів: {len(sim_word_set)}. "
        f"Симуляція: {simulation_time:.2f} с, OCR: {ocr_time:.2f} с."
    )
    return SimulationResult(
        simulation_name, sim_word_set, filename, simulation_time, ocr_time
    )
