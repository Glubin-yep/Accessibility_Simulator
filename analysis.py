import re
import pytesseract
from collections import defaultdict


def get_word_data(image):
    """
    Аналізує зображення, повертає словник з координатами слів
    """
    try:
        custom_config = r"--psm 11"
        data = pytesseract.image_to_data(
            image,
            lang="ukr+eng",
            config=custom_config,
            output_type=pytesseract.Output.DICT,
        )

        word_map = defaultdict(list)
        n_boxes = len(data["level"])

        for i in range(n_boxes):
            # Беремо лише слова з упевненістю > 30%
            conf = int(data["conf"][i])
            if conf < 30:
                continue

            text = data["text"][i]
            if not text or text.isspace():
                continue

            # Нормалізація (видаляємо все, крім літер, цифр та апострофів)
            normalized_word = re.sub(r"[^\w']+", "", text.lower(), re.UNICODE)

            # Враховуємо лише слова довші за 1 символ
            if len(normalized_word) > 1:
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                word_map[normalized_word].append((x, y, w, h))

        return word_map
    except Exception as e:
        print(f"Помилка Tesseract (image_to_data): {e}")
        return defaultdict(list)
