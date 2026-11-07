# Клінічна шкала тяжкості та точка руйнування — план реалізації

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Додати модель спостерігача за Peli з клінічними параметрами, дев'ять моделей тяжкості з нормалізованою шкалою s ∈ [0, 1] і режим `--sweep`, що будує криві ILR(s) та знаходить поріг деградації і точку руйнування.

**Architecture:** Нові модулі `viewing.py` (геометрія перегляду) → `observer.py` (CSF + метод Peli) → `severity.py` (моделі «оптика → спостерігач») → `sweep.py` (сітка, пороги) → `sweep_report.py` (JSON, PNG, консоль). `config.build_simulations(viewing)` будує пресети звичайного прогону з тих самих моделей; `main.py` пропускає еталон через нормального спостерігача і додає режим `--sweep`.

**Tech Stack:** Python ≥ 3.10, NumPy, OpenCV, pytesseract, matplotlib (нова), pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-severity-sweep-design.md`

## Global Constraints

- Python 3.10–3.13 (матриця CI); синтаксис `float | None` дозволено.
- Нова залежність лише одна: `matplotlib>=3.7`; scipy не використовувати.
- Контракт зображень: вхід і вихід усіх симуляцій — `np.uint8` BGR форми `(h, w, 3)`, вхід не змінюється.
- Docstring-и та повідомлення для користувача — українською, як у наявному коді.
- Коміти: англійською, **без** рядка `Co-Authored-By` чи будь-якої згадки ШІ.
- Git у цьому репозиторії: `git -c safe.directory=D:/source/repos/Accessibility_Simulator ...`.
- Тести: `venv/Scripts/python -m pytest` з кореня репозиторію; усі тести, крім маркованих `tesseract`, без мережі й Tesseract.
- Константи зі специфікації (копіювати точно): норма `ObserverCSF(0.0, 1.95)`; втрачене поле `ObserverCSF(1.6, 0.3)`; β = 3 октави; δ = 0.5; зсув Pelli-Robson → пік = 0.35; `F0 = 0.2565`; геометрія за замовчуванням 60 см, 109 PPI, DPR 1, viewport 1440 px; пороги 5 % і 50 %; 10 кроків сітки; 4 потоки за замовчуванням.

## Review Focus

1. **Дуже коротка сторінка або крихітне зображення** (висота < viewport, сторона < 8 px): піраміда й нарізка на фрагменти мають повертати зображення тієї ж форми без винятків. Тест — Task 3 (`test_tiny_image_keeps_shape`), Task 4 (`test_single_partial_tile`).
2. **Повністю чорні ділянки** (локальна середня = 0): контраст не повинен давати NaN/inf. Тест — Task 3 (`test_black_image_has_no_nan`).
3. **Нестандартна геометрія** (дуже далеко, `--distance-cm 500`): усі смуги нижче порогу — зображення стає рівномірним, але середня яскравість зберігається і нічого не падає. Тест — Task 3 (`test_far_viewing_keeps_mean`).
4. **Кирилиця в JSON і шляхах**: `sweep.json` пишеться в UTF-8 з `ensure_ascii=False`, назви моделей читаються назад без змін. Тест — Task 8 (`test_json_preserves_cyrillic`).
5. **Пам'ять при паралелізмі**: без `--workers` не можна запускати десятки потоків на кадрі 2560×16384. Тест — Task 6 (`test_default_workers_is_four`).

---

### Task 1: Геометрія перегляду

**Files:**
- Create: `viewing.py`
- Test: `tests/test_viewing.py`

**Interfaces:**
- Produces: `ViewingConditions(distance_cm=60.0, ppi=109.0, dpr=1.0, viewport_height_px=1440)` (frozen dataclass) з властивостями `pixel_mm: float`, `px_per_degree: float`, `nyquist_cpd: float` і методами `deg_to_px(deg: float) -> float`, `px_to_deg(px: float) -> float`, `cpd_to_cycles_per_px(f: float) -> float`, `cycles_per_px_to_cpd(f: float) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_geometry_px_per_degree():
    v = ViewingConditions()
    assert v.px_per_degree == pytest.approx(44.9, abs=0.2)
    assert v.nyquist_cpd == pytest.approx(22.45, abs=0.1)

def test_degrees_pixels_round_trip():
    v = ViewingConditions(distance_cm=45, ppi=160, dpr=2)
    assert v.px_to_deg(v.deg_to_px(7.5)) == pytest.approx(7.5)
    assert v.cycles_per_px_to_cpd(v.cpd_to_cycles_per_px(12)) == pytest.approx(12)

def test_dpr_makes_css_pixel_larger():
    assert ViewingConditions(dpr=2).px_per_degree == pytest.approx(ViewingConditions().px_per_degree / 2)

@pytest.mark.parametrize("kwargs", [{"distance_cm": 0}, {"ppi": -1}, {"dpr": 0}, {"viewport_height_px": 0}])
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        ViewingConditions(**kwargs)
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_viewing.py -v` — Expected: FAIL (`ModuleNotFoundError: viewing`).
- [ ] **Step 3: Implement `viewing.py`** — формули зі специфікації §2: `pixel_mm = 25.4 * dpr / ppi`, `px_per_degree = 2 * distance_mm * tan(0.5°) / pixel_mm`, `nyquist_cpd = 0.5 * px_per_degree`; перевірка в `__post_init__`.
- [ ] **Step 4: Run** the same command — Expected: PASS.
- [ ] **Step 5: Commit** `viewing.py tests/test_viewing.py` — `Add viewing geometry model`.

---

### Task 2: CSF спостерігача

**Files:**
- Create: `observer.py`
- Test: `tests/test_observer.py`

**Interfaces:**
- Produces: `ObserverCSF(logmar: float, logcs: float)` (frozen dataclass) з властивостями `log_gamma_max`, `cutoff_cpd`, `fmax_cpd` і методом `log_sensitivity(f_cpd: np.ndarray | float) -> np.ndarray | float`; константи `BANDWIDTH_OCTAVES = 3.0`, `TRUNCATION = 0.5`, `PELLI_ROBSON_TO_PEAK = 0.35`, `NORMAL_OBSERVER`, `LOST_FIELD_OBSERVER`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("logmar, logcs", [(0, 1.95), (0.3, 1.5), (1.0, 0.9), (1.6, 0.3)])
def test_csf_passes_through_clinical_points(logmar, logcs):
    csf = ObserverCSF(logmar, logcs)
    assert csf.log_gamma_max == pytest.approx(logcs + 0.35)
    assert csf.cutoff_cpd == pytest.approx(30 * 10 ** -logmar)
    assert csf.log_sensitivity(csf.cutoff_cpd) == pytest.approx(0.0, abs=1e-9)

def test_normal_observer_peak():
    assert NORMAL_OBSERVER.fmax_cpd == pytest.approx(2.52, abs=0.01)
    assert NORMAL_OBSERVER.log_sensitivity(NORMAL_OBSERVER.fmax_cpd) == pytest.approx(2.30)

def test_low_frequencies_are_truncated():
    csf = NORMAL_OBSERVER
    assert csf.log_sensitivity(0.05) == pytest.approx(csf.log_gamma_max - 0.5)

def test_sensitivity_drops_above_cutoff():
    assert NORMAL_OBSERVER.log_sensitivity(40) < 0

@pytest.mark.parametrize("logmar, logcs", [(0, 0), (0, -1), (-0.5, 1.95)])
def test_invalid_observer_raises(logmar, logcs):
    with pytest.raises(ValueError):
        ObserverCSF(logmar, logcs)
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_observer.py -v` — Expected: FAIL (`ModuleNotFoundError: observer`).
- [ ] **Step 3: Implement** у `observer.py` за специфікацією §3.1: `κ = log10(2)`, `β' = log10(2·β)`, `fmax = cutoff / 10**((β'/2)·sqrt(log_gamma_max/κ))`; для `f < fmax` — `max(parabola, log_gamma_max − δ)`; перевірки `logcs > 0`, `logmar ≥ −0.3`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `observer.py tests/test_observer.py` — `Add observer contrast sensitivity function`.

---

### Task 3: Метод Peli — `apply_observer`

**Files:**
- Modify: `observer.py`
- Test: `tests/test_observer.py`

**Interfaces:**
- Consumes: `ViewingConditions` (Task 1), `ObserverCSF` (Task 2), `color_utils.gamma_decode/gamma_encode`.
- Produces: `PYRAMID_F0 = 0.2565`; `apply_observer(image_bgr: np.ndarray, csf: ObserverCSF, viewing: ViewingConditions) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** (допоміжні зображення будуються в лінійному просторі й кодуються `gamma_encode`)

```python
def test_pyramid_f0_matches_grating_measurement():
    # рівень 0 = img - pyrUp(pyrDown(img)); амплітуда синусоїди частоти f
    # шукаємо f_half, де підсилення = 0.5; F0 = sqrt(f_half * 0.5)
    assert measured_f0() == pytest.approx(PYRAMID_F0, abs=0.01)

def test_normal_observer_barely_changes_high_contrast_text(text_image):
    out = apply_observer(text_image, NORMAL_OBSERVER, ViewingConditions())
    assert np.abs(out.astype(int) - text_image.astype(int)).mean() < 2

def test_low_cs_removes_faint_grating_but_keeps_strong():
    v = ViewingConditions()
    poor = ObserverCSF(0.0, 0.45)
    # 6 цикл/град ≈ пік CSF цього спостерігача: поріг ≈ 0.16 (log S ≈ 0.79)
    faint = grating_image(contrast=0.02, cpd=6, viewing=v)
    strong = grating_image(contrast=0.5, cpd=6, viewing=v)
    assert apply_observer(faint, poor, v).std() < faint.std() * 0.2
    assert apply_observer(strong, poor, v).std() > strong.std() * 0.7

def test_mean_luminance_is_preserved(text_image):
    out = apply_observer(text_image, ObserverCSF(1.0, 0.9), ViewingConditions())
    np.testing.assert_allclose(out.reshape(-1, 3).mean(0), text_image.reshape(-1, 3).mean(0), atol=2)

def test_worse_acuity_does_not_add_high_frequencies(text_image):
    v = ViewingConditions()
    hf = lambda im: np.abs(cv2.Laplacian(im.astype(np.float32), cv2.CV_32F)).mean()
    a = hf(apply_observer(text_image, ObserverCSF(0.3, 1.95), v))
    b = hf(apply_observer(text_image, ObserverCSF(0.7, 1.95), v))
    assert b <= a

def test_tiny_image_keeps_shape():
    img = np.full((5, 7, 3), 128, np.uint8)
    assert apply_observer(img, NORMAL_OBSERVER, ViewingConditions()).shape == (5, 7, 3)

def test_black_image_has_no_nan():
    img = np.zeros((64, 64, 3), np.uint8)
    np.testing.assert_array_equal(apply_observer(img, NORMAL_OBSERVER, ViewingConditions()), img)

def test_far_viewing_keeps_mean(text_image):
    out = apply_observer(text_image, NORMAL_OBSERVER, ViewingConditions(distance_cm=500))
    np.testing.assert_allclose(out.mean(), text_image.mean(), atol=2)

def test_input_not_modified(text_image):
    before = text_image.copy()
    apply_observer(text_image, ObserverCSF(1.0, 0.9), ViewingConditions())
    np.testing.assert_array_equal(text_image, before)
```

`text_image` — фікстура як у `tests/test_simulations.py` (перенести в `tests/conftest.py`). `grating_image(contrast, cpd, viewing)` і `measured_f0()` — хелпери в тестовому файлі; `grating_image` будує вертикальну синусоїду 256×512 навколо лінійної яскравості 0.5.

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_observer.py -v` — Expected: нові тести FAIL (`ImportError: apply_observer`).
- [ ] **Step 3: Implement `apply_observer`** за специфікацією §3.2. Кожен канал обробляється окремо (обмеження пам'яті). Глибина піраміди: додавати рівні, поки менша сторона ≥ 8 px і центральна частота рівня (у цикл/град) ≥ 0.25. Рівень k у пікселях власної роздільності має центр `PYRAMID_F0`, у пікселях оригіналу — `PYRAMID_F0 / 2**k`. Поріг для смуги: `|B_k| < max(L_up, 1e-4) * 10**(-log_sensitivity(f_k_cpd))` → 0. Залишок нижнього рівня не змінюється. Збірка `pyrUp` з `dstsize`, обрізання [0, 1].
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Measure speed**: `venv/Scripts/python -c "import time,numpy as np;from observer import *;from viewing import *;img=np.random.default_rng(0).integers(0,256,(16384,2560,3),dtype=np.uint8);t=time.perf_counter();apply_observer(img,NORMAL_OBSERVER,ViewingConditions());print(time.perf_counter()-t)"` — Expected: ≤ 3 с. Якщо більше — профілювати й прискорити до переходу далі (не змінюючи результатів тестів).
- [ ] **Step 6: Commit** `observer.py tests/test_observer.py tests/conftest.py tests/test_simulations.py` — `Add Peli band-limited contrast observer model`.

---

### Task 4: Втрата поля зору в градусах

**Files:**
- Modify: `simulations.py`
- Test: `tests/test_simulations.py`

**Interfaces:**
- Produces: `tile_fixations(height: int, width: int, tile_height: int) -> list[tuple[int, int]]`; `field_loss_mask(height: int, width: int, radius_px: float, feather_sigma_px: float, tile_height: int, lose_inside: bool) -> np.ndarray` (float32 `(h, w)`, 1 = втрачено); `blend_by_mask(intact: np.ndarray, lost: np.ndarray, mask: np.ndarray) -> np.ndarray` (uint8); `cataract_optics(image, yellow_b: float, yellow_g: float, glare_weight: float) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
def test_tile_fixations_centres():
    assert tile_fixations(3000, 1000, 1440) == [(500, 720), (500, 2160), (500, 2940)]

def test_single_partial_tile():
    assert tile_fixations(500, 800, 1440) == [(400, 250)]

def test_scotoma_mask_lost_at_fixation_intact_far_away():
    m = field_loss_mask(2880, 1000, radius_px=100, feather_sigma_px=5, tile_height=1440, lose_inside=True)
    assert m[720, 500] > 0.99 and m[2160, 500] > 0.99
    assert m[720, 0] < 0.01

def test_tunnel_mask_is_inverse_of_scotoma():
    kw = dict(height=1440, width=1000, radius_px=200, feather_sigma_px=10, tile_height=1440)
    np.testing.assert_allclose(field_loss_mask(lose_inside=True, **kw) + field_loss_mask(lose_inside=False, **kw), 1, atol=1e-6)

def test_zero_radius_scotoma_loses_nothing():
    assert field_loss_mask(500, 500, 0, 5, 1440, lose_inside=True).max() < 1e-6

def test_cataract_optics_identity_at_zero(text_image):
    np.testing.assert_array_equal(cataract_optics(text_image, 1.0, 1.0, 0.0), text_image)
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_simulations.py -v` — Expected: нові тести FAIL (`ImportError`).
- [ ] **Step 3: Implement** у `simulations.py`: маска — кола радіусом `radius_px` навколо кожної фіксації (`cv2.circle` на маску), розмиття гаусом σ = `feather_sigma_px` (при `radius_px == 0` маска нульова без розмиття); `lose_inside=False` → `1 − маска`. `cataract_optics` — пожовтіння та засвітка з наявної `simulate_cataract`, але без розмиття (її тіло розбити: `simulate_cataract` викликає `cataract_optics` + розмиття між кроками, поведінку `simulate_cataract` не змінювати).
- [ ] **Step 4: Run** — Expected: PASS (усі старі тести теж).
- [ ] **Step 5: Commit** `simulations.py tests/test_simulations.py` — `Add degree-based field loss masks and cataract optics`.

---

### Task 5: Моделі тяжкості

**Files:**
- Create: `severity.py`
- Test: `tests/test_severity.py`

**Interfaces:**
- Consumes: Tasks 1–4; `simulations.simulate_anomalous_trichromacy_machado`, `simulations.simulate_metamorphopsia`.
- Produces: `SeverityModel(key, name, unit, to_clinical, format_clinical, simulate)` (frozen dataclass; `simulate(image, s, viewing) -> np.ndarray` з перевіркою `0 ≤ s ≤ 1`); `MODELS: dict[str, SeverityModel]` у порядку `acuity, cataract, contrast, scotoma, tunnel, protanomaly, deuteranomaly, tritanomaly, metamorphopsia`; `format_snellen(logmar) -> str`, `format_degrees(v) -> str`; `preset_name(model, s) -> str` = `f"{model.name} ({model.format_clinical(model.to_clinical(s))})"`.

Назви моделей: «Гострота зору», «Катаракта», «Втрата контрасту», «Центральна скотома», «Тунельний зір», «Протаномалія», «Дейтераномалія», «Тританомалія», «Метаморфопсія». Відображення і оптика — таблиця специфікації §4. Формати: Snellen `f"20/{5 * round(20 * 10**logmar / 5)}"`; градуси `f"{round(v, 1):g}°"`; log CS `f"log CS {v:.2f}"`; Machado `f"Machado {round(v, 2):g}"`. Метаморфопсія: `amplitude = 2·s·ppd`, `frequency = 2π / (3·ppd)`. Межа поля: `feather_sigma_px = viewing.deg_to_px(1.0)`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("key", list(MODELS))
def test_zero_severity_equals_normal_observer(key, text_image):
    v = ViewingConditions()
    np.testing.assert_array_equal(MODELS[key].simulate(text_image, 0.0, v), apply_observer(text_image, NORMAL_OBSERVER, v))

@pytest.mark.parametrize("key", list(MODELS))
def test_shape_dtype_and_input_untouched(key, text_image):
    before = text_image.copy()
    out = MODELS[key].simulate(text_image, 0.6, ViewingConditions())
    assert out.shape == text_image.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(text_image, before)

@pytest.mark.parametrize("key, s, label", [
    ("acuity", 0.3, "20/40"), ("acuity", 0.7, "20/100"), ("acuity", 1.0, "20/200"),
    ("cataract", 0.3, "20/40"), ("contrast", 0.5, "log CS 1.20"),
    ("scotoma", 0.5, "5°"), ("tunnel", 0.714, "10°"), ("tunnel", 0.0, "30°"),
    ("protanomaly", 0.6, "Machado 0.6"), ("metamorphopsia", 0.25, "0.5°"),
])
def test_clinical_labels(key, s, label):
    m = MODELS[key]
    assert m.format_clinical(m.to_clinical(s)) == label

def test_preset_name():
    assert preset_name(MODELS["tunnel"], 0.714) == "Тунельний зір (10°)"

@pytest.mark.parametrize("s", [-0.01, 1.01])
def test_severity_out_of_range_raises(s, text_image):
    with pytest.raises(ValueError):
        MODELS["acuity"].simulate(text_image, s, ViewingConditions())

def test_severe_models_change_image(text_image):
    v = ViewingConditions()
    base = apply_observer(text_image, NORMAL_OBSERVER, v)
    for key in ("cataract", "contrast", "scotoma", "protanomaly", "metamorphopsia"):
        assert not np.array_equal(MODELS[key].simulate(text_image, 1.0, v), base), key
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_severity.py -v` — Expected: FAIL (`ModuleNotFoundError: severity`).
- [ ] **Step 3: Implement `severity.py`.**
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `severity.py tests/test_severity.py` — `Add clinical severity models`.

---

### Task 6: Пресети та нормальний спостерігач у звичайному прогоні

**Files:**
- Modify: `config.py`, `main.py`, `tests/test_simulations.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `MODELS`, `preset_name` (Task 5), `apply_observer`, `NORMAL_OBSERVER`.
- Produces: `config.build_simulations(viewing: ViewingConditions) -> dict[str, Callable[[np.ndarray], np.ndarray]]`; `config.DEFAULT_WORKERS = 4`; `main.viewing_from_args(args) -> ViewingConditions`; `main.prepare_baseline(url, output_dir, args, viewing) -> tuple[np.ndarray, dict] | None` (повертає оригінальне зображення і `word_data` еталона після нормального спостерігача); нові прапорці `--distance-cm`, `--ppi`, `--dpr` (тип `positive_float`).

Пресети — точно таблиця специфікації §5 у тому ж порядку; фіксовані (ахроматопсія, три Brettel, флоатери з `seed=config.RANDOM_SEED`) обгортаються `apply_observer(..., NORMAL_OBSERVER, viewing)`. Константу `SIMULATIONS` видалити. `prepare_baseline` зберігає наявну поведінку: `None` при невдалому скріншоті, нечитабельному файлі або 0 слів в еталоні (з тим самим повідомленням `ПОМИЛКА`). `analyze_url` використовує `prepare_baseline`, а `max_workers = args.workers or config.DEFAULT_WORKERS`. Якщо `30 > viewing.nyquist_cpd`, вивести довідку зі специфікації §8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulations.py — параметризація замість config.SIMULATIONS
SIMS = config.build_simulations(ViewingConditions())

def test_preset_names_match_spec():
    assert list(SIMS) == [
        "Гострота зору (20/40)", "Гострота зору (20/100)", "Катаракта (20/40)", "Катаракта (20/100)",
        "Втрата контрасту (log CS 1.20)", "Центральна скотома (5°)", "Тунельний зір (10°)",
        "Протаномалія (Machado 0.6)", "Метаморфопсія (0.5°)", "Метаморфопсія (1.5°)",
        "Ахроматопсія (Скотопічна)", "Протанопія (Brettel)", "Дейтеранопія (Brettel)",
        "Тританопія (Brettel)", "Флоатери (Діабет. рет.)",
    ]
```

```python
# tests/test_main.py
def test_viewing_flags_parsed():
    args = main.build_parser().parse_args(["--distance-cm", "45", "--ppi", "160", "--dpr", "2", "--viewport", "1280x800"])
    v = main.viewing_from_args(args)
    assert (v.distance_cm, v.ppi, v.dpr, v.viewport_height_px) == (45, 160, 2, 800)

@pytest.mark.parametrize("flag", ["--distance-cm", "--ppi", "--dpr"])
def test_non_positive_viewing_rejected(flag):
    with pytest.raises(SystemExit):
        main.build_parser().parse_args([flag, "0"])

def test_default_workers_is_four():
    assert config.DEFAULT_WORKERS == 4

def test_baseline_goes_through_normal_observer(monkeypatch, tmp_path, text_image):
    seen = []
    monkeypatch.setattr(main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1])
    monkeypatch.setattr(main, "get_word_data", lambda img: seen.append(img) or {"word": [(0, 0, 1, 1)]})
    args = main.build_parser().parse_args([])
    v = main.viewing_from_args(args)
    main.prepare_baseline("https://x", str(tmp_path), args, v)
    np.testing.assert_array_equal(seen[0], apply_observer(text_image, NORMAL_OBSERVER, v))
```

Старі тести `test_every_configured_simulation_*` і `test_simulations_do_not_modify_input` залишаються, але ітерують `SIMS`.

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_simulations.py tests/test_main.py -v` — Expected: FAIL (`AttributeError: build_simulations` / `viewing_from_args`).
- [ ] **Step 3: Implement** зміни в `config.py` і `main.py`.
- [ ] **Step 4: Run** `venv/Scripts/python -m pytest` — Expected: усі тести PASS.
- [ ] **Step 5: Commit** `config.py main.py tests/` — `Build presets from severity models and filter baseline through normal observer`.

---

### Task 7: Прогін по шкалі та пороги

**Files:**
- Create: `sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `SeverityModel` (Task 5), `analysis.compute_metrics`, `analysis.get_word_data`.
- Produces: `SweepPoint(severity, clinical, clinical_label, ilr: float | None, error: str | None = None)`; `SweepCurve(model_key, points, onset: float | None, breaking: float | None, failed: bool = False)`; `find_threshold(points: list[SweepPoint], threshold: float) -> float | None`; `run_sweep(image, baseline_word_set, models: list[SeverityModel], viewing, steps=10, onset_ilr=0.05, breaking_ilr=0.5, workers=4, ocr=get_word_data, image_sink=None) -> list[SweepCurve]` (порядок кривих = порядок `models`; точки відсортовані за s, першою — s = 0 з ILR 0.0).

- [ ] **Step 1: Write the failing tests**

```python
P = lambda s, ilr: SweepPoint(s, s, f"{s}", ilr)

def test_threshold_interpolates():
    pts = [P(0, 0), P(0.5, 0.2), P(1.0, 0.6)]
    assert find_threshold(pts, 0.5) == pytest.approx(0.875)

def test_threshold_takes_first_crossing_on_noisy_curve():
    pts = [P(0, 0), P(0.2, 0.06), P(0.4, 0.03), P(0.6, 0.08)]
    assert find_threshold(pts, 0.05) == pytest.approx(0.2 * 0.05 / 0.06)

def test_threshold_skips_failed_points():
    pts = [P(0, 0), SweepPoint(0.5, 0.5, "", None, "boom"), P(1.0, 0.4)]
    assert find_threshold(pts, 0.2) == pytest.approx(0.5)

def test_threshold_not_reached():
    assert find_threshold([P(0, 0), P(1, 0.3)], 0.5) is None

def fake_model(key, fail_at=()):
    def simulate(img, s, v):
        if round(s, 3) in fail_at: raise RuntimeError("boom")
        return np.full_like(img, int(round(s * 100)))
    return SeverityModel(key, key, "u", lambda s: s, lambda c: f"{c:.1f}", simulate)

BASE = {f"w{i}" for i in range(100)}
fake_ocr = lambda img: {f"w{i}": [] for i in range(100 - int(img[0, 0, 0]))}  # ILR = s

def test_run_sweep_curve_and_thresholds():
    (curve,) = run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m")], ViewingConditions(), steps=10, workers=2, ocr=fake_ocr)
    assert [p.severity for p in curve.points] == pytest.approx([i / 10 for i in range(11)])
    assert curve.onset == pytest.approx(0.05) and curve.breaking == pytest.approx(0.5)

def test_partial_failures_are_recorded():
    (curve,) = run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m", fail_at=(0.3,))], ViewingConditions(), steps=10, ocr=fake_ocr)
    bad = [p for p in curve.points if p.ilr is None]
    assert len(bad) == 1 and bad[0].error == "boom" and not curve.failed

def test_model_failed_when_most_points_fail():
    fails = tuple(round(i / 10, 3) for i in range(1, 7))
    (curve,) = run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m", fail_at=fails)], ViewingConditions(), steps=10, ocr=fake_ocr)
    assert curve.failed and curve.onset is None and curve.breaking is None

def test_image_sink_called_for_each_point():
    calls = []
    run_sweep(np.zeros((4, 4, 3), np.uint8), BASE, [fake_model("m")], ViewingConditions(), steps=4, ocr=fake_ocr, image_sink=lambda k, s, im: calls.append((k, s)))
    assert sorted(s for _, s in calls) == pytest.approx([0.25, 0.5, 0.75, 1.0])
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_sweep.py -v` — Expected: FAIL (`ModuleNotFoundError: sweep`).
- [ ] **Step 3: Implement `sweep.py`.** Інтерполяція в `find_threshold`: між останньою успішною точкою нижче порогу `(s0, i0)` і першою `≥` порогу `(s1, i1)`: `s0 + (t − i0)·(s1 − s0)/(i1 − i0)`. `failed`, якщо помилкових точок > `steps / 2`. Сітка `s_i = i / steps`, i = 1..steps, точки — `ThreadPoolExecutor(max_workers=workers)` по всіх парах (модель, s).
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `sweep.py tests/test_sweep.py` — `Add severity sweep with degradation and breaking thresholds`.

---

### Task 8: Звіт прогону по шкалі

**Files:**
- Create: `sweep_report.py`
- Modify: `requirements.txt` (додати `matplotlib>=3.7`)
- Test: `tests/test_sweep_report.py`

**Interfaces:**
- Consumes: `SweepCurve`, `SweepPoint` (Task 7), `MODELS` (Task 5), `ViewingConditions`.
- Produces: `sweep_to_dict(url: str, viewing, baseline_word_count: int, onset_ilr: float, breaking_ilr: float, curves: list[SweepCurve], models: dict[str, SeverityModel], total_s: float, created: str) -> dict`; `write_sweep_json(path: str, data: dict) -> None`; `plot_sweep(path: str, curves, models, onset_ilr: float, breaking_ilr: float) -> None`; `format_sweep_table(curves, models) -> str`.

Структура словника — точно приклад зі специфікації §7; поріг кривої: `None` або `{"severity", "clinical", "clinical_label"}`. Графік: `matplotlib.use("Agg")` до імпорту `pyplot`; сітка 3×3; верхня вісь `secondary_xaxis("top")` з підписами `format_clinical` у точках s = 0, 0.5, 1; лінії порогів; `dpi=300`. Таблиця: рядок на модель, не досягнуто — `> 1.0`, `failed` — `ПОМИЛКА`.

- [ ] **Step 1: Write the failing tests**

```python
def sample_curves():
    pts = [SweepPoint(i / 10, i / 10, "20/20", min(1.0, i / 10)) for i in range(11)]
    return [SweepCurve("acuity", pts, 0.05, 0.5), SweepCurve("cataract", pts, 0.05, None)]

def test_dict_structure():
    d = sweep_to_dict("https://x", ViewingConditions(), 745, 0.05, 0.5, sample_curves(), MODELS, 12.3, "2026-11-07T12:00:00")
    assert d["thresholds"] == {"onset_ilr": 0.05, "breaking_ilr": 0.5}
    assert d["viewing"]["px_per_degree"] == pytest.approx(44.9, abs=0.2)
    acuity = d["models"][0]
    assert acuity["key"] == "acuity" and acuity["unit"] == "logMAR"
    assert acuity["breaking"]["clinical_label"] == "20/65"   # 20·10^0.5 = 63.2 → округлення до 5
    assert d["models"][1]["breaking"] is None
    assert len(acuity["points"]) == 11

def test_json_preserves_cyrillic(tmp_path):
    path = tmp_path / "звіт" / "sweep.json"
    path.parent.mkdir()
    d = sweep_to_dict("https://x", ViewingConditions(), 1, 0.05, 0.5, sample_curves(), MODELS, 1.0, "t")
    write_sweep_json(str(path), d)
    assert "Гострота зору" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == d

def test_plot_written(tmp_path):
    p = tmp_path / "sweep_ilr.png"
    plot_sweep(str(p), sample_curves(), MODELS, 0.05, 0.5)
    assert p.stat().st_size > 10_000

def test_table_marks_unreached():
    t = format_sweep_table(sample_curves(), MODELS)
    assert "Катаракта" in t and "> 1.0" in t
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pip install "matplotlib>=3.7"` і `venv/Scripts/python -m pytest tests/test_sweep_report.py -v` — Expected: FAIL (`ModuleNotFoundError: sweep_report`).
- [ ] **Step 3: Implement `sweep_report.py`**; запис JSON — `json.dump(..., ensure_ascii=False, indent=2)` у файл з `encoding="utf-8"`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `sweep_report.py requirements.txt tests/test_sweep_report.py` — `Add sweep JSON, chart and console report`.

---

### Task 9: Режим `--sweep` у CLI

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `prepare_baseline`, `viewing_from_args` (Task 6), `run_sweep` (Task 7), функції Task 8.
- Produces: прапорці `--sweep`, `--sweep-steps` (int ≥ 2, default 10), `--onset-ilr` (default 5), `--breaking-ilr` (default 50), `--save-sweep-images`; `main.sweep_url(url, output_dir, args, viewing) -> bool` (False → код 2).

Валідація після `parse_args`: `0 < onset < breaking ≤ 100`, інакше `parser.error(...)` (код 2, до скріншота). У режимі `--sweep` прапорець `--max-ilr` ігнорується з попередженням. Результати — `<output_dir>/sweep/`: `sweep.json`, `sweep_ilr.png`; з `--save-sweep-images` — `sweep/images/<key>_s<s:.2f>.png` через `reporting.save_image`. Таблиця з `format_sweep_table` друкується в консоль. `created` = `datetime.now().isoformat(timespec="seconds")`, `total_s` — реальний час прогону по шкалі.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("argv", [["--onset-ilr", "50", "--breaking-ilr", "50"], ["--onset-ilr", "0"], ["--breaking-ilr", "101"], ["--sweep-steps", "1"]])
def test_invalid_sweep_args_exit_2(argv, monkeypatch):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: pytest.fail("must validate before Tesseract"))
    with pytest.raises(SystemExit) as e:
        main.main(argv)
    assert e.value.code == 2

def test_sweep_end_to_end_with_fakes(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1])
    monkeypatch.setattr(main, "get_word_data", lambda img: {"a": [], "b": []})
    code = main.main(["--sweep", "--sweep-steps", "2", "--output-dir", str(tmp_path), "--workers", "2"])
    assert code == 0
    data = json.loads((tmp_path / "sweep" / "sweep.json").read_text(encoding="utf-8"))
    assert [m["key"] for m in data["models"]] == list(MODELS)
    assert (tmp_path / "sweep" / "sweep_ilr.png").exists()

def test_sweep_failed_model_exit_2(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1])
    calls = iter(range(10_000))
    def ocr(img):
        if next(calls) == 0: return {"a": []}  # еталон
        raise RuntimeError("ocr down")
    monkeypatch.setattr(main, "get_word_data", ocr)
    assert main.main(["--sweep", "--sweep-steps", "2", "--output-dir", str(tmp_path), "--workers", "1"]) == 2
```

- [ ] **Step 2: Run** `venv/Scripts/python -m pytest tests/test_main.py -v` — Expected: нові тести FAIL.
- [ ] **Step 3: Implement** зміни в `main.py`; `run_sweep` викликається з `ocr=get_word_data` з простору імен `main` (щоб працювали підміни в тестах).
- [ ] **Step 4: Run** `venv/Scripts/python -m pytest` — Expected: усі PASS.
- [ ] **Step 5: Commit** `main.py tests/test_main.py` — `Add --sweep mode to CLI`.

---

### Task 10: Калібрування, документація і справжній прогін

**Files:**
- Create: `tests/test_calibration.py`
- Modify: `pytest.ini`, `README.md`

**Interfaces:**
- Consumes: `MODELS["acuity"]`, `apply_observer`, `analysis.get_word_data`, `main.configure_tesseract`.

- [ ] **Step 1: Register marker** у `pytest.ini`: `markers = tesseract: потребує встановленого Tesseract (ukr+eng)`.
- [ ] **Step 2: Write the calibration test**

```python
pytestmark = pytest.mark.tesseract
CHART_VIEWING = ViewingConditions(distance_cm=300)          # ≈ 224 px/град
ROWS = [round(0.1 * i, 1) for i in range(11)]               # logMAR 0.0 … 1.0
WORDS = ["dknrs", "hvzoc", "rcsdk", "nzhvo", "kosrd", "vhcnz", "sdrko", "chnvs", "zokdh", "rvsnc", "hdzko"]

@pytest.fixture(scope="module", autouse=True)
def _tesseract():
    if not main.configure_tesseract(None):
        pytest.skip("Tesseract недоступний")

@pytest.mark.parametrize("x", [0.3, 0.6])
def test_acuity_chart_cutoff(x):
    chart = acuity_chart(ROWS, WORDS, CHART_VIEWING)   # висота літери 5 MAR, штрих 1 MAR, чорне на білому
    seen = set(get_word_data(MODELS["acuity"].simulate(chart, x, CHART_VIEWING)))
    for row, word in zip(ROWS, WORDS):
        if row >= x + 0.1 - 1e-9: assert word in seen, f"рядок {row} мав читатися при logMAR {x}"
        if row <= x - 0.1 + 1e-9: assert word not in seen, f"рядок {row} не мав читатися при logMAR {x}"
```

`acuity_chart` — хелпер у файлі: висота великої літери `h = 5 · 10**row / 60 · ppd` px, `cv2.putText` (`FONT_HERSHEY_SIMPLEX`, `fontScale = h / 22`, `thickness = max(1, round(h / 5))`), відступ між рядками 1.5·h, слова **великими** літерами.

- [ ] **Step 3: Run** `venv/Scripts/python -m pytest tests/test_calibration.py -v` — Expected: PASS (спершу перевірити, що без зниження гостроти, `x = 0`, OCR читає всі рядки — інакше проблема в хелпері, а не в моделі). Якщо тест модельно не проходить (зрізи зміщені більше ніж на рядок), **не підганяти константи** — зупинитися й повідомити користувачу: це рішення про модель, що змінює специфікацію.
- [ ] **Step 4: Update README** — розділ про `--sweep` (команда, вихідні файли, прапорці геометрії), таблиця моделей з клінічними шкалами та обмеження зі специфікації §12; таблиця пресетів звичайного прогону.
- [ ] **Step 5: Run full suite** `venv/Scripts/python -m pytest` — Expected: усі PASS.
- [ ] **Step 6: Commit** `pytest.ini tests/test_calibration.py README.md` — `Add acuity calibration test and document sweep mode`.
- [ ] **Step 7: Real run** `venv/Scripts/python main.py --sweep --output-dir results/sweep_run` (у фоні, ≤ 10 хв). Перевірити критерії специфікації §11: немає падінь ILR > 5 п.п. між сусідніми точками; деградація кольорових моделей s ≥ 0.5 або не досягнута; руйнування катаракти ≤ руйнування гостроти; час ≤ 10 хв. Потім звичайний прогін `venv/Scripts/python main.py --output-dir results/standard_run --max-ilr 50` — таблиця з 15 пресетами, код виходу узгоджений із ILR. Результати (таблиця порогів, графік) показати користувачу; невиконаний критерій — повідомити, а не приховувати.
