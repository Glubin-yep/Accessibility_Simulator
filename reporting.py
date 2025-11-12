import os
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from analysis import get_word_data


def _get_font(font_size=16):
    """
    Допоміжна функція для пошуку та завантаження .ttf шрифту, що підтримує кирилицю.
    """
    font_paths = [
        "C:/Windows/Fonts/Arial.ttf",
    ]

    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break

    try:
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        else:
            print(
                "ПОПЕРЕДЖЕННЯ: Не знайдено шрифтів Arial або DejaVuSans. Спроба завантажити шрифт за замовчуванням."
            )
            return ImageFont.load_default()
    except IOError as e:
        print(f"ПОМИЛКА: Не вдалося завантажити шрифт {font_path}. {e}")
        return ImageFont.load_default()


def generate_readability_report_table(
    report_data, baseline_word_set, filename="screenshot_report_readability.png"
):
    """
    Генерує зображення-таблицю зі статистикою читабельності (використовує Pillow).
    """
    print(f"Генерую таблицю звіту читабельності: {filename}...")

    font = _get_font(font_size=15)
    header_font = _get_font(font_size=15)
    row_height = 30
    col_widths = [240, 100, 100, 100, 100]
    img_width = sum(col_widths) + 20

    baseline_count = len(baseline_word_set)

    table_data = []
    table_data.append(
        ("[Симуляція]", "[Всього]", "[Збіг %]", "[Втрати %]", "[Артефакти]")
    )
    table_data.append(("Оригінал (База)", f"{baseline_count}", "100.0%", "0.0%", "0"))

    for name, (sim_word_set, _, _) in sorted(report_data.items()):
        sim_count = len(sim_word_set)
        common_count = len(baseline_word_set.intersection(sim_word_set))
        lost_count = len(baseline_word_set.difference(sim_word_set))
        artifact_count = len(sim_word_set.difference(baseline_word_set))

        overlap_perc = (
            (common_count / baseline_count * 100) if baseline_count > 0 else 0
        )
        loss_perc = (lost_count / baseline_count * 100) if baseline_count > 0 else 0

        table_data.append(
            (
                name,
                f"{sim_count}",
                f"{overlap_perc:.1f}%",
                f"{loss_perc:.1f}%",
                f"{artifact_count}",
            )
        )

    img_height = row_height * len(table_data) + 20
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    y_offset = 10
    for row_index, row in enumerate(table_data):
        x_offset = 10
        current_font = header_font if row_index == 0 else font

        for i, cell in enumerate(row):
            draw.text((x_offset, y_offset), str(cell), fill="black", font=current_font)
            x_offset += col_widths[i]

        y_offset += row_height
        draw.line(
            [
                (5, y_offset - int(row_height / 2) + 2),
                (img_width - 5, y_offset - int(row_height / 2) + 2),
            ],
            fill=(200, 200, 200),
            width=1,
        )

    try:
        image.save(filename)
        print(f"[Успішно] Таблицю читабельності збережено як {filename}")
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося зберегти таблицю звіту: {e}")


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
