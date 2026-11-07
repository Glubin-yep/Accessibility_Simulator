import re
from collections import defaultdict
from dataclasses import dataclass

import pytesseract

TESSERACT_LANG = "ukr+eng"
TESSERACT_CONFIG = r"--psm 11"  # Sparse text: пошук тексту в довільному порядку
MIN_CONFIDENCE = 30  # слова з меншою впевненістю відкидаються
MIN_WORD_LENGTH = 2

_NON_WORD_RE = re.compile(r"[^\w']+")


def normalize_word(text):
    """
    Нормалізує токен OCR: нижній регістр, видалення всього, крім літер,
    цифр, '_' та апострофів.
    """
    return _NON_WORD_RE.sub("", text.lower())


def get_word_data(image):
    """
    Розпізнає текст на зображенні та повертає словник
    {нормалізоване_слово: [(x, y, w, h), ...]}.

    Помилки Tesseract (наприклад, не встановлено або не знайдено мовні пакети)
    не приховуються: вони мають зупинити аналіз, а не дати 0 слів у звіті.
    """
    data = pytesseract.image_to_data(
        image,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    word_map = defaultdict(list)
    for i in range(len(data["level"])):
        if float(data["conf"][i]) < MIN_CONFIDENCE:
            continue

        text = data["text"][i]
        if not text or text.isspace():
            continue

        normalized_word = normalize_word(text)
        if len(normalized_word) >= MIN_WORD_LENGTH:
            box = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
            word_map[normalized_word].append(box)

    return word_map


@dataclass(frozen=True)
class ReadabilityMetrics:
    """
    Метрики читабельності для однієї симуляції.

    W_common = W_base ∩ W_sim  — слова, що лишились читабельними;
    W_lost   = W_base \\ W_sim  — втрачені слова;
    W_art    = W_sim \\ W_base  — артефакти розпізнавання;
    ILR (Information Loss Ratio) = |W_lost| / |W_base|.
    """

    baseline_count: int
    sim_count: int
    common_count: int
    lost_count: int
    artifact_count: int

    @property
    def overlap_ratio(self):
        if self.baseline_count == 0:
            return 0.0
        return self.common_count / self.baseline_count

    @property
    def information_loss_ratio(self):
        if self.baseline_count == 0:
            return 0.0
        return self.lost_count / self.baseline_count


def compute_metrics(baseline_words, sim_words):
    baseline_words = set(baseline_words)
    sim_words = set(sim_words)
    return ReadabilityMetrics(
        baseline_count=len(baseline_words),
        sim_count=len(sim_words),
        common_count=len(baseline_words & sim_words),
        lost_count=len(baseline_words - sim_words),
        artifact_count=len(sim_words - baseline_words),
    )
