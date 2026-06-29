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
    assert ce.blur_radius(1000, 500, 70) == max(1, ce.jround((1000 / 100) * (70 / 100) * 5))


def test_blur_strength_is_responsive():
    # Regression: the blur slider must visibly change output across its WHOLE range.
    # The old 14x multiplier saturated — sigma >= ~9 flattens a ~220px region, so
    # strengths 30..100 all yielded an identical fully-blurred block (slider looked
    # dead). We assert remaining detail decreases monotonically AND that the top
    # half of the slider still differs (the bug had std(50) ~ std(100) ~ 0).
    W = H = 220
    yy, xx = np.mgrid[0:H, 0:W]
    region = np.stack([(((xx // 12 + yy // 12) % 2) * 200 + 30).astype(np.uint8)] * 3, -1)
    rect = {"x": 0, "y": 0, "w": W, "h": H, "ellipse": False}

    def remaining_std(strength):
        img = region.copy()
        ce.style_region(img, region.copy(), rect,
                        {**ce.AUTO_CENSOR_DEFAULTS, "style": "blur", "blurStrength": strength},
                        ce.lcg(7))
        return float(img[:, :, 0].std())

    stds = [remaining_std(s) for s in (10, 30, 50, 70, 100)]
    assert all(a >= b for a, b in zip(stds, stds[1:])), f"not monotonic: {stds}"
    assert stds[2] - stds[4] > 2.0, f"upper slider saturated (50~100 identical): {stds}"
    assert stds[0] - stds[4] > 20.0, f"slider barely moves: {stds}"


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


def test_all_styles_region_locality():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    rect = {"x": 30, "y": 30, "w": 50, "h": 50, "ellipse": False}
    mask = ce.shape_mask(rect, 120, 120)
    for style in ("bar", "mosaic", "blur", "barsV", "barsH", "manga", "glitch"):
        img = base.copy()
        ce.style_region(img, base.copy(), rect, {**ce.AUTO_CENSOR_DEFAULTS, "style": style}, ce.lcg(7))
        assert np.array_equal(img[mask == 0], base[mask == 0]), f"{style} touched outside region"
        assert not np.array_equal(img[mask == 255], base[mask == 255]), f"{style} did not change region"


def test_glitch_additive_ghosts_symmetric():
    # Bright vertical line at the region's center. The additive +-sx ghosts span the
    # FULL region height, so col (center-sx) is brightened by the -sx ghost (the bug
    # dropped this) and (center+sx) by the +sx ghost. Averaging over rows washes out
    # the random slices. If the -sx draw is dropped, left_bright collapses toward 0.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 49:51] = 255                    # vertical line at col ~50, region height
    canvas = img.copy()
    rect = {"x": 20, "y": 30, "w": 60, "h": 40, "ellipse": False}  # sx = jround(60*0.06)+1 = 5
    out = img.copy()
    ce.style_region(out, canvas, rect,
                    {**ce.AUTO_CENSOR_DEFAULTS, "style": "glitch", "glitchIntensity": 100}, ce.lcg(7))
    rows = slice(30, 70)
    left_bright = float((out[rows, 45].max(axis=-1) > 50).mean())   # center-sx, only the -sx ghost
    right_bright = float((out[rows, 55].max(axis=-1) > 50).mean())  # center+sx, the +sx ghost
    # Both ghosts must render (not dropped) and be roughly symmetric. If the -sx
    # draw is dropped, left_bright collapses toward 0 while right stays high.
    assert left_bright > 0.2, f"left (-sx) ghost missing: {left_bright}"
    assert right_bright > 0.2, f"right (+sx) ghost missing: {right_bright}"
    assert abs(left_bright - right_bright) < 0.25, f"asymmetric ghosts (a side dropped): {left_bright} vs {right_bright}"


def test_lcg_golden_values():
    r = ce.lcg(7)
    seq = [r() for _ in range(4)]
    expected = [0.55898, 0.23697, 0.31676, 0.55051]
    for a, b in zip(seq, expected):
        assert abs(a - b) < 1e-4, (seq, expected)


from PIL import Image


def test_apply_censor_obscures_box_region():
    img = Image.new("RGB", (200, 200), (180, 180, 180))
    box = {"x1": 0.25, "y1": 0.25, "x2": 0.5, "y2": 0.5, "score": 0.9,
           "class_id": 3, "label": "FEMALE_BREAST_EXPOSED", "sensitive": True,
           "exposed": True, "ellipse": False, "bodyPart": None, "derived": False}
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "bar", "barColor": "#000000", "shape": "rect", "padding": 0.0}
    out = np.asarray(ce.apply_auto_censor(img, [box], opts))
    assert out[80, 70].tolist() == [0, 0, 0]
    assert out[10, 10].tolist() == [180, 180, 180]


def test_apply_censor_brush_mask():
    img = Image.new("RGB", (100, 100), (200, 200, 200))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "bar", "barColor": "#000000"}
    out = np.asarray(ce.apply_auto_censor(img, [], opts, manual_mask=Image.fromarray(mask, "L")))
    assert out[50, 50].tolist() == [0, 0, 0]
    assert out[5, 5].tolist() == [200, 200, 200]


def test_apply_no_targets_returns_unchanged():
    img = Image.new("RGB", (64, 64), (123, 45, 67))
    out = np.asarray(ce.apply_auto_censor(img, [], ce.AUTO_CENSOR_DEFAULTS))
    assert np.array_equal(out, np.asarray(img))


def test_export_preset_returns_named_outputs():
    img = Image.new("RGB", (256, 256), (180, 180, 180))
    box = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6, "score": 0.9, "class_id": 3,
           "label": "FEMALE_BREAST_EXPOSED", "sensitive": True, "exposed": True,
           "ellipse": False, "bodyPart": None, "derived": False}
    outs = ce.export_preset(img, [box], "Both")
    names = [o[0] for o in outs]
    assert any("master" in n.lower() for n in names) and any("censor" in n.lower() for n in names)


def test_export_preset_mosaic_censors_small_region():
    # Regression: JP mosaic presets must use an ABSOLUTE tile (longSide/divisor),
    # so even a SMALL region is properly mosaiced. The old code converted it to a
    # region-relative mosaicBlocks, so for a region much smaller than the image the
    # tile collapsed to ~1px — i.e. no censoring at all (presets "didn't work").
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8))
    box = {"x1": 0.46, "y1": 0.46, "x2": 0.54, "y2": 0.54, "score": 0.9, "class_id": 3,
           "label": "X", "sensitive": True, "exposed": True, "ellipse": False,
           "bodyPart": None, "derived": False}
    y0, y1, x0, x1 = 478, 545, 478, 545  # well inside the ~82px region
    orig_std = float(np.asarray(img)[y0:y1, x0:x1, 0].std())
    for preset in ("DLsite", "FANZA", "Pixiv"):
        outs = ce.export_preset(img, [box], preset)
        censored = next(im for name, im, *_ in outs if "censor" in name.lower())
        std = float(np.asarray(censored.convert("RGB"))[y0:y1, x0:x1, 0].std())
        assert std < orig_std * 0.4, f"{preset}: small region barely censored ({std:.1f} vs {orig_std:.1f})"


def test_frosted_is_blurred_and_lighter():
    # Frosted = blur (lower variance) + white veil (lighter average), inside the region only.
    rng = np.random.default_rng(2)
    base = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    rect = {"x": 30, "y": 30, "w": 50, "h": 50, "ellipse": False}
    mask = ce.shape_mask(rect, 120, 120)
    img = base.copy()
    ce.style_region(img, base.copy(), rect,
                    {**ce.AUTO_CENSOR_DEFAULTS, "style": "frosted", "frostAmount": 60}, ce.lcg(7))
    # outside untouched
    assert np.array_equal(img[mask == 0], base[mask == 0]), "frosted touched outside region"
    inside_before = base[mask == 255].astype(np.float32)
    inside_after = img[mask == 255].astype(np.float32)
    assert inside_after.std() < inside_before.std(), "frosted should reduce variance (blur)"
    assert inside_after.mean() > inside_before.mean(), "frosted should lighten (white veil)"


def test_static_changes_region_and_is_deterministic():
    base = np.full((120, 120, 3), 128, dtype=np.uint8)
    rect = {"x": 30, "y": 30, "w": 50, "h": 50, "ellipse": False}
    mask = ce.shape_mask(rect, 120, 120)

    def run(seed):
        img = base.copy()
        ce.style_region(img, base.copy(), rect,
                        {**ce.AUTO_CENSOR_DEFAULTS, "style": "static",
                         "staticIntensity": 100, "glitchSeed": seed}, ce.lcg(7))
        return img

    a, a2, b = run(7), run(7), run(9)
    assert np.array_equal(a[mask == 0], base[mask == 0]), "static touched outside region"
    assert not np.array_equal(a[mask == 255], base[mask == 255]), "static did not change region"
    assert np.array_equal(a, a2), "same seed must be identical"
    assert not np.array_equal(a, b), "different seed must differ"


# ---------------------------------------------------------------------------
# Sticker tests (Task 3)
# ---------------------------------------------------------------------------

def _solid_sticker(h, w, rgb, alpha):
    s = np.zeros((h, w, 4), dtype=np.uint8)
    s[:, :, 0], s[:, :, 1], s[:, :, 2] = rgb
    s[:, :, 3] = alpha
    return s


def test_composite_sticker_cover_fills_bbox():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    sticker = _solid_sticker(10, 20, (0, 0, 255), 255)   # opaque blue, non-square
    rect = {"x": 30, "y": 30, "w": 40, "h": 40, "ellipse": False}
    ce.composite_sticker(img, rect, sticker, "cover", 100.0, 100.0, 0.0)
    # cover must fully paint the bbox; the bbox center is blue, outside untouched
    assert img[50, 50].tolist() == [0, 0, 255], "cover did not paint bbox center"
    assert img[5, 5].tolist() == [200, 200, 200], "cover painted outside bbox"


def test_composite_sticker_contain_stays_within_bbox():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    sticker = _solid_sticker(10, 20, (255, 0, 0), 255)   # wide → contain leaves vertical margin
    rect = {"x": 30, "y": 30, "w": 40, "h": 40, "ellipse": False}
    ce.composite_sticker(img, rect, sticker, "contain", 100.0, 100.0, 0.0)
    changed = np.any(img != 200, axis=2)
    ys, xs = np.where(changed)
    assert xs.min() >= 30 and xs.max() <= 69, "contain overflowed bbox horizontally"
    assert ys.min() >= 30 and ys.max() <= 69, "contain overflowed bbox vertically"
    assert changed.any(), "contain painted nothing"


def test_composite_sticker_alpha_blends_half():
    img = np.zeros((40, 40, 3), dtype=np.uint8)             # black bg
    sticker = _solid_sticker(40, 40, (255, 255, 255), 128)  # ~50% white
    rect = {"x": 0, "y": 0, "w": 40, "h": 40, "ellipse": False}
    ce.composite_sticker(img, rect, sticker, "stretch", 100.0, 100.0, 0.0)
    # 255 * (128/255) ≈ 128 over black
    assert 118 <= int(img[20, 20, 0]) <= 138, f"alpha blend wrong: {img[20,20,0]}"


def test_style_region_sticker_uses_path():
    import tempfile, os
    from PIL import Image
    d = tempfile.mkdtemp()
    p = os.path.join(d, "dot.png")
    Image.fromarray(_solid_sticker(20, 20, (0, 255, 0), 255), "RGBA").save(p)
    ce._sticker_cache.clear()
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    rect = {"x": 25, "y": 25, "w": 50, "h": 50, "ellipse": False}
    ce.style_region(img, img.copy(), rect,
                    {**ce.AUTO_CENSOR_DEFAULTS, "style": "sticker", "stickerPath": str(p),
                     "stickerFit": "cover"}, ce.lcg(7))
    assert img[50, 50].tolist() == [0, 255, 0], "sticker not composited onto region"
    assert img[5, 5].tolist() == [100, 100, 100], "sticker leaked outside region bbox"


def test_apply_censor_sticker_brush_mask():
    import tempfile, os
    from PIL import Image
    d = tempfile.mkdtemp()
    p = os.path.join(d, "blue.png")
    Image.fromarray(_solid_sticker(20, 20, (0, 0, 255), 255), "RGBA").save(p)
    ce._sticker_cache.clear()
    img = Image.new("RGB", (100, 100), (200, 200, 200))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "sticker", "stickerPath": p,
            "stickerFit": "cover"}
    out = np.asarray(ce.apply_auto_censor(img, [], opts, manual_mask=Image.fromarray(mask, "L")))
    assert out[50, 50].tolist() == [0, 0, 255], "sticker not applied at brush bbox"
    assert out[5, 5].tolist() == [200, 200, 200], "sticker leaked outside brush bbox"


def test_list_stickers_builtin_and_custom():
    import tempfile, os
    from PIL import Image
    bd = tempfile.mkdtemp(); cd = tempfile.mkdtemp()
    Image.new("RGBA", (8, 8)).save(os.path.join(bd, "heart.png"))
    Image.new("RGBA", (8, 8)).save(os.path.join(bd, "star.png"))
    items = ce.list_stickers(builtin=bd, custom=cd)
    names = [n for n, _ in items]
    assert "heart" in names and "star" in names, names
    # a file dropped into custom shows up on the next call
    Image.new("RGBA", (8, 8)).save(os.path.join(cd, "mine.png"))
    items2 = ce.list_stickers(builtin=bd, custom=cd)
    assert "mine" in [n for n, _ in items2], items2
    assert os.path.isabs(items2[0][1]), "paths must be absolute"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
