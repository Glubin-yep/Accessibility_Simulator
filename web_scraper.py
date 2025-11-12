from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


def take_screenshot(url, filename="screenshot_original.png"):
    """
    Використовує Selenium для створення скріншота веб-сторінки.
    """
    print(f"Збір даних: Роблю скріншот {url}...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=2560,1440")
    try:
        with webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        ) as driver:
            driver.get(url)
            driver.implicitly_wait(2)
            driver.save_screenshot(filename)
        print(f"Скріншот збережено як {filename}")
        return filename
    except Exception as e:
        print(f"Помилка під час зйомки скріншота: {e}")
        return None
