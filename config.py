import functools

import simulations as sim

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

SIMULATIONS = {
    # Катаракта (різної сили)
    "Катаракта (Легка)": functools.partial(
        sim.simulate_cataract, blur_ksize=7, glare_weight=0.1
    ),
    "Катаракта (Сильна)": functools.partial(
        sim.simulate_cataract, blur_ksize=21, glare_weight=0.3
    ),
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
    "Протаномалія (Machado)": functools.partial(
        sim.simulate_anomalous_trichromacy_machado, sim_type="protanomaly", severity=0.6
    ),
    "Метаморфопсія (Легка)": functools.partial(
        sim.simulate_metamorphopsia, amplitude=8, frequency=0.03
    ),
    "Метаморфопсія (Сильна)": functools.partial(
        sim.simulate_metamorphopsia, amplitude=15, frequency=0.05
    ),
    "Центральна Скотома (AMD)": functools.partial(
        sim.simulate_central_scotoma, radius_percentage=0.2
    ),
    "Тунельний Зір (Глаукома)": functools.partial(
        sim.simulate_tunnel_vision, aperture_percentage=0.5
    ),
    "Флоатери (Діабет. рет.)": functools.partial(
        sim.simulate_floaters, num_floaters=30, max_size=50, seed=RANDOM_SEED
    ),
    "Втрата контрасту (CSF)": functools.partial(
        sim.simulate_csf_loss, cutoff_frequency_ratio=0.28
    ),
}
