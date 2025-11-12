import cv2
import numpy as np
from color_utils import (
    gamma_decode,
    gamma_encode,
    MATRIX_RGB_TO_LMS,
    MATRIX_LMS_TO_RGB,
    SIM_MATRICES,
)

# --- ФУНКЦІЇ СИМУЛЯЦІЇ ---


def simulate_cataract(
    image,
    blur_ksize=15,
    yellow_b_factor=0.8,
    yellow_g_factor=0.95,
    glare_weight=0.2,
):
    """
    Симулює катаракту за 3-компонентною моделлю.
    """
    # 1. Пожовтіння
    b, g, r = cv2.split(image)
    b_yellowed = cv2.multiply(b, yellow_b_factor).astype(np.uint8)
    g_yellowed = cv2.multiply(g, yellow_g_factor).astype(np.uint8)
    img_yellowed = cv2.merge([b_yellowed, g_yellowed, r])

    # 2. Blur
    blur_ksize = int(blur_ksize)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    img_blurred = cv2.GaussianBlur(img_yellowed, (blur_ksize, blur_ksize), 0)

    # 3. Відблиски
    haze = np.full_like(img_blurred, (255, 255, 255), dtype=np.uint8)
    img_glare = cv2.addWeighted(img_blurred, 1.0 - glare_weight, haze, glare_weight, 0)
    return img_glare


def simulate_achromatopsia_scotopic(image):
    """
    Симулює ахроматопсію (монохромазія паличок)
    """
    img_linear = gamma_decode(image)
    scotopic_matrix = np.array([[0.2472, 0.5820, 0.1708]])  # B, G, R
    img_gray_linear = cv2.transform(img_linear, scotopic_matrix)
    img_gray_3ch_linear = cv2.merge([img_gray_linear, img_gray_linear, img_gray_linear])
    return gamma_encode(img_gray_3ch_linear)


def simulate_dichromacy_brettel(image, sim_type="protanopia"):
    """
    Симулює дихромазію

    sim_type: "protanopia", "deuteranopia", "tritanopia"
    """
    if sim_type not in SIM_MATRICES:
        print(f"Помилка: Невідомий тип дихромазії {sim_type}")
        return image

    img_linear = gamma_decode(image)

    img_lms = cv2.transform(img_linear, MATRIX_RGB_TO_LMS)

    sim_matrix = SIM_MATRICES[sim_type]
    img_lms_sim = cv2.transform(img_lms, sim_matrix)

    img_linear_sim = cv2.transform(img_lms_sim, MATRIX_LMS_TO_RGB)

    return gamma_encode(img_linear_sim)


def simulate_metamorphopsia(image, amplitude=15, frequency=0.05):
    """
    Імітує метаморфопсію
    """
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x + amplitude * np.sin(map_y * frequency)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def simulate_central_scotoma(image, radius_percentage=0.15, blur_ksize=99):
    """
    Імітує центральну скотому.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    scotoma_radius = int(min(h, w) * radius_percentage)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize < 3:
        blur_ksize = 3

    scotoma_layer = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    mask = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), scotoma_radius, (0, 0, 0), -1)

    mask_blurred = cv2.GaussianBlur(mask, (201, 201), 0)
    mask_float = mask_blurred.astype(np.float32) / 255.0

    img_blended = image.astype(np.float32) * mask_float + scotoma_layer.astype(
        np.float32
    ) * (1.0 - mask_float)
    return img_blended.astype(np.uint8)


def simulate_tunnel_vision(image, aperture_percentage=0.5, blur_ksize=99):
    """
    Імітує тунельний зір
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    aperture_radius = int(min(h, w) * aperture_percentage)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize < 3:
        blur_ksize = 3

    blurred_layer = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), aperture_radius, (255, 255, 255), -1)

    mask_blurred = cv2.GaussianBlur(mask, (201, 201), 0)
    mask_float = mask_blurred.astype(np.float32) / 255.0

    img_blended = image.astype(np.float32) * mask_float + blurred_layer.astype(
        np.float32
    ) * (1.0 - mask_float)
    return img_blended.astype(np.uint8)


def simulate_floaters(
    image, num_floaters=20, max_size=40, min_opacity=0.2, max_opacity=0.6
):
    """
    Імітує "мушки" (флоатери) при діабетичній ретинопатії.
    """
    h, w = image.shape[:2]

    shadow_layer = np.full_like(image, 255, dtype=np.uint8)

    for _ in range(num_floaters):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size_x = np.random.randint(max_size // 4, max_size)
        size_y = np.random.randint(max_size // 4, max_size)
        angle = np.random.randint(0, 180)
        opacity = np.random.uniform(min_opacity, max_opacity)
        color_val = int(255 * (1.0 - opacity))  # 0=чорний, 255=прозорий
        cv2.ellipse(
            shadow_layer,
            (x, y),
            (size_x, size_y),
            angle,
            0,
            360,
            (color_val, color_val, color_val),
            -1,
        )

    shadow_layer = cv2.GaussianBlur(shadow_layer, (11, 11), 0)

    img_float = image.astype(np.float32) / 255.0
    shadow_float = shadow_layer.astype(np.float32) / 255.0
    result_float = cv2.multiply(img_float, shadow_float)

    return (result_float * 255.0).astype(np.uint8)


def simulate_csf_loss(image, cutoff_frequency_ratio=0.1):
    """
    Імітує втрату контрастної чутливості (CSF) через фільтрацію
    у частотній області (видалення високих частот).
    """
    filtered_channels = []
    for chan in cv2.split(image):
        # 1. Перетворення Фур'є
        dft = cv2.dft(np.float32(chan), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 2. Створення маски фільтра
        rows, cols = chan.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.float32)
        cutoff = int(min(crow, ccol) * cutoff_frequency_ratio)
        cv2.circle(mask, (ccol, crow), cutoff, (1, 1), -1)  # Низькочастотний фільтр

        # 3. Застосування маски
        fshift = dft_shift * mask

        # 4. Зворотне Перетворення Фур'є
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Нормалізація
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        filtered_channels.append(img_back.astype(np.uint8))

    return cv2.merge(filtered_channels)


def simulate_anomalous_trichromacy_machado(image, sim_type, severity=0.5):
    """
    TODO: Реалізація моделі Мачадо (2009) .

    """
    print(
        f"ПОПЕРЕДЖЕННЯ: {sim_type} (Machado, severity={severity}) не реалізовано. "
        "Повертаю оригінальне зображення."
    )
    return image
