import cv2
import numpy as np
import pytest


@pytest.fixture
def text_image():
    img = np.full((240, 320, 3), 230, dtype=np.uint8)
    cv2.putText(img, "Text", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (40, 40, 40), 5)
    cv2.rectangle(img, (220, 30), (300, 90), (30, 60, 200), -1)
    return img
