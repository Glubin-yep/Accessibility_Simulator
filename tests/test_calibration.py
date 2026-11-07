"""
Калібрування моделі гостроти за синтетичною таблицею (штрих літери = 1 MAR).
Потребує Tesseract (ukr+eng); без нього тест пропускається.
"""
import cv2
import numpy as np
import pytest

import main
from analysis import get_word_data
from severity import MODELS
from viewing import ViewingConditions

pytestmark = pytest.mark.tesseract

CHART_VIEWING = ViewingConditions(distance_cm=300)  # ≈ 224 px/град
ROWS = [round(0.1 * i, 1) for i in range(11)]  # logMAR 0.0 … 1.0
WORDS = ["dknrs", "hvzoc", "rcsdk", "nzhvo", "kosrd", "vhcnz", "sdrko", "chnvs", "zokdh", "rvsnc", "hdzko"]
MARGIN = 60  # px білого поля навколо таблиці


@pytest.fixture(scope="module", autouse=True)
def _tesseract():
    if not main.configure_tesseract(None):
        pytest.skip("Tesseract недоступний")


def acuity_chart(rows, words, viewing):
    """Таблиця: висота великої літери 5 MAR, штрих 1 MAR, чорне на білому."""
    ppd = viewing.px_per_degree
    font = cv2.FONT_HERSHEY_SIMPLEX
    specs = []
    for row, word in zip(rows, words):
        h = 5 * 10**row / 60 * ppd
        scale = h / 22
        thickness = max(1, round(h / 5))
        (tw, _), _ = cv2.getTextSize(word.upper(), font, scale, thickness)
        specs.append((word.upper(), h, scale, thickness, tw))

    width = int(max(s[4] for s in specs)) + 2 * MARGIN
    height = int(MARGIN * 2 + sum(1.5 * s[1] for s in specs) + specs[-1][1])
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    bands = []
    y = MARGIN
    for word, h, scale, thickness, _ in specs:
        top = y
        y += h
        cv2.putText(canvas, word, (MARGIN, int(round(y))), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += 0.5 * h
        # смуга рядка: літери + половина міжрядкового відступу з кожного боку
        bands.append((max(0, int(top - 0.25 * h) - 10), min(height, int(y - 0.25 * h) + 10)))
    return canvas, bands


def read_rows(image, bands):
    """
    OCR кожного рядка окремо. Tesseract відкидає текст, що в кілька разів менший
    за решту сторінки, тож на цілій таблиці (діапазон висот 10x) малі рядки
    губляться навіть без зниження гостроти. Симуляція застосовується до всієї
    таблиці, а розпізнавання - до смуг результату.
    """
    seen = set()
    for top, bottom in bands:
        seen |= set(get_word_data(image[top:bottom]))
    return seen


@pytest.mark.parametrize("x", [0.3, 0.6])
def test_acuity_chart_cutoff(x):
    chart, bands = acuity_chart(ROWS, WORDS, CHART_VIEWING)
    seen = read_rows(MODELS["acuity"].simulate(chart, x, CHART_VIEWING), bands)
    for row, word in zip(ROWS, WORDS):
        if row >= x + 0.1 - 1e-9:
            assert word in seen, f"рядок {row} мав читатися при logMAR {x}"
        if row <= x - 0.1 + 1e-9:
            assert word not in seen, f"рядок {row} не мав читатися при logMAR {x}"
