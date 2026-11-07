import base64

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


def _make_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--hide-scrollbars")
    # Selenium >= 4.11 сам знаходить або завантажує Chrome і chromedriver
    # (Selenium Manager), окремий webdriver-manager не потрібен.
    return webdriver.Chrome(options=options)


def take_screenshot(
    url,
    filename="screenshot_original.png",
    viewport=(2560, 1440),
    full_page=True,
    max_height=16384,
    page_load_timeout=30,
):
    """
    Робить скріншот веб-сторінки через Selenium + Chrome DevTools Protocol.

    viewport:   розмір області перегляду в CSS-пікселях (ширина, висота);
                задається точно через Emulation.setDeviceMetricsOverride
                з deviceScaleFactor = 1.
    full_page:  True — знімок усієї сторінки (не лише видимої частини),
                висота обмежується max_height пікселями.
    """
    width, height = viewport
    print(f"Збір даних: Роблю скріншот {url} ({width}x{height}, повна сторінка: {full_page})...")
    try:
        with _make_driver() as driver:
            driver.set_page_load_timeout(page_load_timeout)
            driver.execute_cdp_cmd(
                "Emulation.setDeviceMetricsOverride",
                {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
            )
            driver.get(url)
            WebDriverWait(driver, page_load_timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            params = {"format": "png"}
            if full_page:
                metrics = driver.execute_cdp_cmd("Page.getLayoutMetrics", {})
                content = metrics.get("cssContentSize") or metrics["contentSize"]
                page_height = min(int(content["height"]), max_height)
                params.update(
                    captureBeyondViewport=True,
                    clip={"x": 0, "y": 0, "width": width, "height": max(page_height, height), "scale": 1},
                )

            shot = driver.execute_cdp_cmd("Page.captureScreenshot", params)
            with open(filename, "wb") as f:
                f.write(base64.b64decode(shot["data"]))

        print(f"Скріншот збережено як {filename}")
        return filename
    except Exception as e:
        print(f"Помилка під час зйомки скріншота: {e}")
        return None
