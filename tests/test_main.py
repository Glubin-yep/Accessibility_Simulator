import numpy as np
import pytest

import config
import main
from observer import NORMAL_OBSERVER, apply_observer
from reporting import save_image


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
