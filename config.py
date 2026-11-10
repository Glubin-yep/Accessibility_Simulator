import functools

import simulations as sim
from observer import NORMAL_OBSERVER, apply_observer
from severity import MODELS, preset_name

# URL за замовчуванням (можна перевизначити аргументом --url)
URL_TO_ANALYZE = "https://uk.wikipedia.org/wiki/%D0%93%D0%BE%D0%BB%D0%BE%D0%B2%D0%BD%D0%B0_%D1%81%D1%82%D0%BE%D1%80%D1%96%D0%BD%D0%BA%D0%B0"

# Параметри скріншота
VIEWPORT = (2560, 1440)  # ширина, висота області перегляду (CSS px)
FULL_PAGE = True  # знімати всю сторінку, а не лише видиму частину
MAX_PAGE_HEIGHT = 16384  # обмеження висоти повного знімка, px

# Куди зберігати результати (для кожного URL створюється підпапка)
OUTPUT_DIR = "results"
ORIGINAL_FILENAME = "screenshot_original.png"
REPORT_FILENAME = "screenshot_report_readability.png"

# Зерно для випадкових симуляцій (флоатери) — для відтворюваності результатів
RANDOM_SEED = 42

# Кількість потоків за замовчуванням
DEFAULT_WORKERS = 4

# (ключ моделі, тяжкість s) — назви пресетів містять клінічне значення
PRESETS = (
    ("acuity", 0.3),
    ("acuity", 0.7),
    ("cataract", 0.3),
    ("cataract", 0.7),
    ("contrast", 0.5),
    ("scotoma", 0.5),
    ("tunnel", 0.714),
    ("protanomaly", 0.6),
    ("metamorphopsia", 0.25),
    ("metamorphopsia", 0.75),
)


def _through_normal_observer(func, viewing):
    def run(image):
        return apply_observer(func(image), NORMAL_OBSERVER, viewing)

    return run


def build_simulations(viewing):
    """Пресети звичайного прогону: назва -> функція(image) для заданих умов перегляду."""
    simulations = {}
    for key, s in PRESETS:
        model = MODELS[key]
        simulations[preset_name(model, s)] = functools.partial(
            model.simulate, s=s, viewing=viewing
        )

    fixed = {
        "Ахроматопсія (Скотопічна)": sim.simulate_achromatopsia_scotopic,
        "Протанопія (Brettel)": functools.partial(
            sim.simulate_dichromacy_brettel, sim_type="protanopia"
        ),
        "Дейтеранопія (Brettel)": functools.partial(
            sim.simulate_dichromacy_brettel, sim_type="deuteranopia"
        ),
        "Тританопія (Brettel)": functools.partial(
            sim.simulate_dichromacy_brettel, sim_type="tritanopia"
        ),
        "Флоатери (Діабет. рет.)": functools.partial(
            sim.simulate_floaters, num_floaters=30, max_size=50, seed=RANDOM_SEED
        ),
    }
    for name, func in fixed.items():
        simulations[name] = _through_normal_observer(func, viewing)
    return simulations
