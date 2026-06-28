import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
import censor_engine as ce


def test_defaults():
    d = ce.AUTO_CENSOR_DEFAULTS
    assert d["style"] == "mosaic" and d["mosaicBlocks"] == 10 and d["padding"] == 0.08
    assert d["barCount"] == 9 and d["glitchSeed"] == 7 and d["mergeGap"] == 30


def test_lcg_deterministic():
    r1, r2 = ce.lcg(7), ce.lcg(7)
    seq1 = [r1() for _ in range(5)]
    seq2 = [r2() for _ in range(5)]
    assert seq1 == seq2
    assert all(0.0 <= v < 1.0 for v in seq1)


def test_merge_rects_merges_within_gap():
    a = {"x": 0, "y": 0, "w": 10, "h": 10, "ellipse": False}
    b = {"x": 15, "y": 0, "w": 10, "h": 10, "ellipse": False}
    merged = ce.merge_rects([a, b], gap=10)
    assert len(merged) == 1 and merged[0]["w"] == 25


def test_merge_rects_keeps_far_apart():
    a = {"x": 0, "y": 0, "w": 10, "h": 10, "ellipse": False}
    b = {"x": 100, "y": 0, "w": 10, "h": 10, "ellipse": False}
    assert len(ce.merge_rects([a, b], gap=10)) == 2


def test_blur_radius_formula():
    assert ce.blur_radius(1000, 500, 70) == max(2, round((1000/100)*(70/100)*28))


def test_bar_style_changes_only_region():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    canvas = img.copy()
    rect = {"x": 20, "y": 20, "w": 40, "h": 40, "ellipse": False}
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "bar", "barColor": "#000000"}
    ce.style_region(img, canvas, rect, opts, ce.lcg(7))
    assert img[40, 40].tolist() == [0, 0, 0]
    assert img[5, 5].tolist() == [200, 200, 200]


def test_mosaic_style_changes_region():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    canvas = img.copy()
    rect = {"x": 10, "y": 10, "w": 40, "h": 40, "ellipse": False}
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "mosaic", "mosaicBlocks": 4}
    ce.style_region(img, canvas, rect, opts, ce.lcg(7))
    assert not np.array_equal(img[10:50, 10:50], canvas[10:50, 10:50])
    assert np.array_equal(img[60:, 60:], canvas[60:, 60:])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
