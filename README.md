# Симулятор Доступності Вад Зору (Accessibility Simulator)

Цей Python-інструмент призначений для аналізу веб-доступності. Він автоматично робить скріншот вказаного веб-сайту, симулює різні вади зору (катаракта, дальтонізм, глаукома тощо) та оцінює, наскільки читабельним залишається текст після цих спотворень, використовуючи технологію OCR (Tesseract).

## 🌟 Основні Можливості

* **Автоматизація:** Використовує Selenium для автоматичного захоплення повного скріншота веб-сторінки.
* **Широкий спектр симуляцій**:
    * **Помутніння кришталика:** Катаракта (легка та сильна форми).
    * **Порушення кольоросприйняття:** Протанопія, Дейтеранопія, Протаномалія, Ахроматопсія.
    * **Захворювання сітківки:** Центральна скотома (AMD), Тунельний зір (Глаукома), "Мушки" (Діабетична ретинопатія).
    * **Інші вади:** Метаморфопсія (викривлення), втрата контрастної чутливості (CSF).
* **Аналіз Читабельності (OCR):** Використовує Tesseract OCR для порівняння тексту на оригіналі та на симуляціях.
* **Візуалізація:** Зелені рамки позначають розпізнаний текст, червоні — втрачений текст або помилки.

---

## 📸 Галерея Результатів Симуляцій

Після запуску скрипт генерує зображення для кожного типу вади зору. Нижче наведено повний перелік файлів, які будуть створені.

### 📊 Зведена Статистика
**Файл:** `screenshot_report_readability.png`
Таблиця, що показує відсоток тексту, який залишився читабельним для кожної симуляції.

![Звіт читабельності](https://github.com/user-attachments/assets/27df32b6-afd7-436f-b474-63cdce16668f)

### 🖼️ Оригінал
**Файл:** `screenshot_original.png`
Базове зображення для порівняння (без спотворень).

<img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/c2465f7a-b4a4-4f7b-8bcb-519450cccb16" />


---

### 👁️ Катаракта (Помутніння)
Симуляція розмиття, пожовтіння кришталика та ефекту засліплення.

| Легка стадія | Сильна стадія |
|:---:|:---:|
| **`screenshot_катаракта_легка.png`** | **`screenshot_катаракта_сильна.png`** |
| <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/d92a42b7-e8e7-4dea-a54f-f17112940deb" /> | <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/3cf1c1e8-042d-4779-93bb-ba712aae7e65" /> |

---

### 🎨 Порушення Кольоросприйняття (Дальтонізм)
Симуляція сприйняття кольорів різними типами дихроматів та аномальних трихроматів.

| Протанопія (Сліпота на червоний) | Дейтеранопія (Сліпота на зелений) |
|:---:|:---:|
| **`screenshot_протанопія_brettel.png`** | **`screenshot_дейтеранопія_brettel.png`** |
| <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/8270122e-2d39-4b31-b77a-85e4e42cc611" /> | <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/6ccad122-4ba1-49c2-91df-cd2eef7aeb2d" /> |

| Протаномалія (Слабкість червоного) | Ахроматопсія (Ч/Б зір) |
|:---:|:---:|
| **`screenshot_протаномалія_machado.png`** | **`screenshot_ахроматопсія_скотопічна.png`** |
| ![Протаномалія](screenshot_протаномалія_machado.png) | <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/d210fb32-313f-4a00-ac92-6c6ab771a51a" /> |

---

### 🌑 Обмеження Поля Зору
Симуляція хвороб, що призводять до втрати частин поля зору.

| Тунельний Зір (Глаукома) | Центральна Скотома (AMD) |
|:---:|:---:|
| **`screenshot_тунельний_зір_глаукома.png`** | **`screenshot_центральна_скотома_amd.png`** |
| <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/7aeaa875-53ae-4523-8fb2-1f0cd16d25e9" /> | <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/a1898903-45c9-4e18-a512-343e12990240" /> |
| *Втрата периферійного зору.* | *Втрата центрального зору (читання ускладнене).* |

---

### 〰️ Спотворення та Артефакти
Симуляція викривлень ліній та появи сторонніх об'єктів в оці.

| Метаморфопсія (Легка) | Метаморфопсія (Сильна) |
|:---:|:---:|
| **`screenshot_метаморфопсія_легка.png`** | **`screenshot_метаморфопсія_сильна.png`** |
| <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/d7ba25ea-4438-4e02-b413-ec30a78c1d84" /> | <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/b0c5d11c-b36e-4eac-9792-3fb05fc6825e" /> |
| *Хвилеподібне викривлення тексту.* | *Сильне спотворення геометрії.* |

| Флоатери ("Мушки") | Втрата Контрасту (CSF) |
|:---:|:---:|
| **`screenshot_флоатери_діабет._рет..png`** | **`screenshot_втрата_контрасту_csf.png`** |
| <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/d3eccffb-5b73-4a48-8f77-68d0a363bea4" /> | <img width="2544" height="1293" alt="image" src="https://github.com/user-attachments/assets/2e9d1eb2-f80f-4ddf-bb6a-c619bea62820" /> |
| *Темні плями, що перекривають контент.* | *Зниження чіткості країв та контрасту.* |

---

## 🛠️ Встановлення

### 1. Системні Залежності (Tesseract OCR)

Для роботи OCR необхідно встановити **Tesseract** у вашій ОС:

* **Windows:** Завантажте інсталятор з [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    * ⚠️ **Важливо:** Під час встановлення оберіть мови `Ukrainian` та `English`.
* **macOS:** `brew install tesseract tesseract-lang`
* **Linux:** `sudo apt-get install tesseract-ocr tesseract-ocr-ukr tesseract-ocr-eng`

### 2. Python Залежності

1.  Клонуйте репозиторій:
    ```bash
    git clone https://github.com/Glubin-yep/Accessibility_Simulator.git
    cd accessibility-simulator
    ```

2.  Створіть віртуальне середовище (рекомендовано):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  Встановіть бібліотеки:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Використання

1.  Відкрийте `config.py` та вкажіть URL сайту, який хочете перевірити:
    ```python
    URL_TO_ANALYZE = "https://uk.wikipedia.org"
    ```

2.  Запустіть скрипт:
    ```bash
    python main.py
    ```

3.  **Результат:**
    * Скрипт запустить Chrome у фоновому режимі.
    * У консолі з'явиться прогрес виконання симуляцій.
    * По завершенню в папці з'являться зображення з результатами та звіт.

### Налаштування шляху до Tesseract (Windows)

Якщо Tesseract встановлено не в стандартну папку, вкажіть шлях вручну при запуску:

```bash
python main.py --tesseract_cmd "D:\Apps\Tesseract-OCR\tesseract.exe"
