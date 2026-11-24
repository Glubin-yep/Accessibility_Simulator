"""Помилка в одній симуляції не зупиняє решту прогону."""

import config
import main
import reporting
from reporting import SimulationResult, generate_readability_report_table, process_simulation, save_image


def _boom(image):
    raise RuntimeError("зламана симуляція")


def _fake_ocr(img):
    return {"a": [(0, 0, 5, 5)], "b": [(10, 0, 5, 5)]}


def test_process_simulation_catches_simulation_error(tmp_path, text_image):
    r = process_simulation("Зламана", _boom, text_image, {"a": [(0, 0, 1, 1)]}, str(tmp_path))
    assert r.failed and "симуляція" in r.error and "RuntimeError" in r.error
    assert r.word_set == set() and r.filename is None
    assert list(tmp_path.iterdir()) == []


def test_process_simulation_catches_ocr_error(monkeypatch, tmp_path, text_image):
    def broken_ocr(img):
        raise OSError("tesseract впав")

    monkeypatch.setattr(reporting, "get_word_data", broken_ocr)
    r = process_simulation("OCR", lambda img: img, text_image, {"a": [(0, 0, 1, 1)]}, str(tmp_path))
    assert r.failed and r.error.startswith("OCR: OSError")


def test_report_table_marks_failed_row(tmp_path):
    results = [
        SimulationResult("Добра", {"a"}, "x.png", 0.1, 0.2),
        SimulationResult("Зламана", set(), None, 0.0, 0.0, "симуляція: RuntimeError: x"),
    ]
    path = tmp_path / "report.png"
    generate_readability_report_table(results, {"a", "b"}, filename=str(path))
    assert path.exists()


def test_one_failing_simulation_does_not_stop_run(monkeypatch, tmp_path, text_image, capsys):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(main, "get_word_data", _fake_ocr)
    monkeypatch.setattr(reporting, "get_word_data", lambda img: {"a": [(0, 0, 5, 5)]})
    monkeypatch.setattr(
        config,
        "build_simulations",
        lambda viewing, **kw: {"Перша": lambda img: img, "Зламана": _boom, "Третя": lambda img: img},
    )
    code = main.main(["--output-dir", str(tmp_path), "--workers", "2", "--max-ilr", "90"])
    out = capsys.readouterr().out
    assert code == main.EXIT_ERROR
    assert (tmp_path / config.REPORT_FILENAME).exists()
    assert (tmp_path / "screenshot_перша.png").exists()
    assert (tmp_path / "screenshot_третя.png").exists()
    assert "НЕ ВИКОНАНО: Зламана" in out
    assert "Усі виконані симуляції в межах порогу." in out


def test_failure_outranks_quality_gate(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(main, "get_word_data", _fake_ocr)
    monkeypatch.setattr(reporting, "get_word_data", lambda img: {})
    monkeypatch.setattr(
        config, "build_simulations", lambda viewing, **kw: {"Сліпа": lambda img: img, "Зламана": _boom}
    )
    # ILR = 100 % > 10 % (код 1), але помилка симуляції має пріоритет (код 2).
    assert main.main(["--output-dir", str(tmp_path), "--max-ilr", "10"]) == main.EXIT_ERROR


def test_all_ok_still_exit_zero(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(main, "get_word_data", _fake_ocr)
    monkeypatch.setattr(reporting, "get_word_data", _fake_ocr)
    monkeypatch.setattr(config, "build_simulations", lambda viewing, **kw: {"Ок": lambda img: img})
    assert main.main(["--output-dir", str(tmp_path)]) == main.EXIT_OK
