import json

import numpy as np
import pytest

import config
import main
from observer import NORMAL_OBSERVER, apply_observer
from reporting import save_image
from severity import MODELS


def test_viewing_flags_parsed():
    args = main.build_parser().parse_args(
        ["--distance-cm", "45", "--ppi", "160", "--dpr", "2", "--viewport", "1280x800"]
    )
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
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(
        main, "get_word_data", lambda img: seen.append(img) or {"word": [(0, 0, 1, 1)]}
    )
    args = main.build_parser().parse_args([])
    v = main.viewing_from_args(args)
    main.prepare_baseline("https://x", str(tmp_path), args, v)
    np.testing.assert_array_equal(seen[0], apply_observer(text_image, NORMAL_OBSERVER, v))


@pytest.mark.parametrize(
    "argv",
    [
        ["--onset-ilr", "50", "--breaking-ilr", "50"],
        ["--onset-ilr", "0"],
        ["--breaking-ilr", "101"],
        ["--sweep-steps", "1"],
    ],
)
def test_invalid_sweep_args_exit_2(argv, monkeypatch):
    monkeypatch.setattr(
        main, "configure_tesseract", lambda cmd: pytest.fail("must validate before Tesseract")
    )
    with pytest.raises(SystemExit) as e:
        main.main(argv)
    assert e.value.code == 2


def test_sweep_end_to_end_with_fakes(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(main, "get_word_data", lambda img: {"a": [], "b": []})
    code = main.main(
        ["--sweep", "--sweep-steps", "2", "--output-dir", str(tmp_path), "--workers", "2"]
    )
    assert code == 0
    data = json.loads((tmp_path / "sweep" / "sweep.json").read_text(encoding="utf-8"))
    assert [m["key"] for m in data["models"]] == list(MODELS)
    assert (tmp_path / "sweep" / "sweep_ilr.png").exists()


def test_sweep_saves_point_images(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(main, "get_word_data", lambda img: {"a": [], "b": []})
    code = main.main(
        [
            "--sweep", "--sweep-steps", "2", "--save-sweep-images",
            "--max-ilr", "1", "--output-dir", str(tmp_path), "--workers", "1",
        ]
    )
    assert code == 0
    key = next(iter(MODELS))
    assert (tmp_path / "sweep" / "images" / f"{key}_s1.00.png").exists()


def test_sweep_failed_model_exit_2(monkeypatch, tmp_path, text_image):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    calls = iter(range(10_000))

    def ocr(img):
        if next(calls) == 0:
            return {"a": []}  # еталон
        raise RuntimeError("ocr down")

    monkeypatch.setattr(main, "get_word_data", ocr)
    code = main.main(
        ["--sweep", "--sweep-steps", "2", "--output-dir", str(tmp_path), "--workers", "1"]
    )
    assert code == 2


@pytest.mark.parametrize("argv", [["--workers", "0"], ["--workers", "-2"], ["--viewport", "0x800"], ["--viewport", "1280x0"]])
def test_non_positive_workers_viewport_rejected(argv):
    with pytest.raises(SystemExit) as e:
        main.build_parser().parse_args(argv)
    assert e.value.code == 2


def test_save_images_without_sweep_warns(monkeypatch, capsys):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: False)
    main.main(["--save-sweep-images"])
    assert "--save-sweep-images ігнорується" in capsys.readouterr().out


def test_sweep_output_oserror_exit_2(monkeypatch, tmp_path, text_image, capsys):
    monkeypatch.setattr(main, "configure_tesseract", lambda cmd: True)
    monkeypatch.setattr(
        main, "take_screenshot", lambda url, path, **kw: (save_image(path, text_image), path)[1]
    )
    monkeypatch.setattr(main, "get_word_data", lambda img: {"a": [], "b": []})

    def boom(path, data):
        raise OSError("read-only")

    monkeypatch.setattr(main, "write_sweep_json", boom)
    code = main.main(["--sweep", "--sweep-steps", "2", "--output-dir", str(tmp_path), "--workers", "1"])
    assert code == 2
    assert "ПОМИЛКА" in capsys.readouterr().out
