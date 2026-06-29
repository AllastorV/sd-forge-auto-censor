"""Censor engine helpers — pure numpy/cv2, no gradio/webui.

Faithful port of the TypeScript autoCensor.ts functions:
  lcg, mergeRects, boxToRect, clipShape (→ shape_mask), drawBars,
  blurRadius, styleRegion  (7 styles: bar, mosaic, blur, barsV, barsH,
  manga, glitch).

Canvas-2D → numpy translation notes:
  ctx.drawImage(src, sx,sy,sw,sh, dx,dy,dw,dh) scaling
      → cv2.resize with INTER_AREA (down) / INTER_NEAREST (up, no smooth)
        or INTER_LINEAR (up, smooth)
  imageSmoothingEnabled=false  → INTER_NEAREST
  CSS filter:blur(Npx)         → cv2.GaussianBlur(img,(0,0),sigmaX=N)
  ctx.clip() + shape          → build 0/255 mask; composite where mask==255
  globalCompositeOperation='lighter' + globalAlpha=0.85
      → float accumulate: dst = clip(dst + src*0.85, 0, 255)
"""
from __future__ import annotations

import math
import numpy as np
import cv2


def jround(x: float) -> int:
    """JS Math.round equivalent: floor(x + 0.5) — ties round up, not banker's."""
    return math.floor(x + 0.5)

# ---------------------------------------------------------------------------
# Defaults (mirror of AUTO_CENSOR_DEFAULTS in autoCensor.ts)
# ---------------------------------------------------------------------------

AUTO_CENSOR_DEFAULTS: dict = {
    "mode": "censor",
    "style": "mosaic",
    "shape": "auto",
    "barColor": "#0a0a0a",
    "mosaicBlocks": 10,
    "blurStrength": 70,
    "glitchIntensity": 70,
    "glitchSeed": 7,
    "barCount": 9,
    "barThickness": 0.55,
    "padding": 0.08,
    "mergeGap": 30,
    "bgEffect": "glitch",
    "bgIntensity": 70,
    "boxFrames": False,
    "frameLabels": True,
}

# ---------------------------------------------------------------------------
# LCG — matches TS 32-bit signed arithmetic exactly
# ---------------------------------------------------------------------------

def lcg(seed: int):
    """Return a deterministic () -> float[0,1) RNG matching the TS lcg().

    TS: let s = (seed | 0) || 1
        each call: s = (s*1664525+1013904223)|0; return (s>>>0)%100000/100000
    """
    def _to_int32(v: int) -> int:
        v = int(v) & 0xFFFF_FFFF
        return v - 0x1_0000_0000 if v >= 0x8000_0000 else v

    s = _to_int32(int(seed)) or 1

    def _rand() -> float:
        nonlocal s
        s = _to_int32(s * 1664525 + 1013904223)
        return (s & 0xFFFF_FFFF) % 100000 / 100000

    return _rand

# ---------------------------------------------------------------------------
# merge_rects — iterative near-merge (TS mergeRects L56-74)
# ---------------------------------------------------------------------------

def merge_rects(rects: list[dict], gap: int) -> list[dict]:
    """Merge rectangles whose bounding boxes are within *gap* pixels.

    Rects: list of {x, y, w, h, ellipse}  (all ints).
    Returns list of the same schema; merged rects get ellipse=False,
    a single-origin rect keeps its original ellipse flag.
    """
    lst = [dict(r, n=1) for r in rects]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(lst) and not changed:
            j = i + 1
            while j < len(lst):
                a, b = lst[i], lst[j]
                near = (
                    a["x"] < b["x"] + b["w"] + gap
                    and b["x"] < a["x"] + a["w"] + gap
                    and a["y"] < b["y"] + b["h"] + gap
                    and b["y"] < a["y"] + a["h"] + gap
                )
                if near:
                    x = min(a["x"], b["x"])
                    y = min(a["y"], b["y"])
                    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
                    y2 = max(a["y"] + a["h"], b["y"] + b["h"])
                    lst[i] = {
                        "x": x, "y": y, "w": x2 - x, "h": y2 - y,
                        "ellipse": False, "n": a["n"] + b["n"],
                    }
                    lst.pop(j)
                    changed = True
                    break
                j += 1
            i += 1

    return [
        {
            "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
            "ellipse": r["ellipse"] if r["n"] == 1 else False,
        }
        for r in lst
    ]

# ---------------------------------------------------------------------------
# box_to_rect — TS boxToRect L76-85
# ---------------------------------------------------------------------------

def box_to_rect(box: dict, W: int, H: int, padding: float, shape: str) -> dict:
    """Convert a NudeNet detection box (x1,y1,x2,y2 normalised) to a pixel Rect.

    Applies padding, clamps to image, sets ellipse flag from shape/box.
    """
    bx = box["x1"] * W
    by = box["y1"] * H
    bw = (box["x2"] - box["x1"]) * W
    bh = (box["y2"] - box["y1"]) * H
    px = bw * padding
    py = bh * padding
    x = max(0, jround(bx - px))
    y = max(0, jround(by - py))
    w = min(W - x, jround(bw + 2 * px))
    h = min(H - y, jround(bh + 2 * py))
    ellipse = shape == "ellipse" or (shape == "auto" and bool(box.get("ellipse", False)))
    return {"x": x, "y": y, "w": w, "h": h, "ellipse": ellipse}

# ---------------------------------------------------------------------------
# blur_radius — TS blurRadius L151-153
# ---------------------------------------------------------------------------

def blur_radius(w: int, h: int, strength: float) -> int:
    """Blur sigma in pixels: max(1, jround(longest/100 * strength/100 * 5)).

    Fed to cv2.GaussianBlur as sigmaX. The earlier 14× multiplier *saturated*:
    on a typical ~220px region, sigma ≥ ~9 already flattens it completely, so
    strengths 30–100 all produced an identical fully-blurred block and the slider
    looked dead. Measuring remaining-detail std across strengths showed 5× spreads
    a perceptible blur ramp over the WHOLE slider (sharp at 10 → fully hidden at
    100) for both the region and the full-image (stylize / brush-mask) paths.
    """
    return max(1, jround((max(w, h) / 100) * (strength / 100) * 5))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_color(hex_str: str) -> tuple[int, int, int]:
    """Parse '#rrggbb' → (r, g, b)."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def shape_mask(rect: dict, W: int, H: int) -> np.ndarray:
    """Build a uint8 [H,W] mask: 255 inside the shape, 0 outside.

    For ellipse: semi-axes × 1.16 (matching TS clipShape L89).
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    if rect.get("ellipse", False):
        cx, cy = jround(x + w / 2), jround(y + h / 2)
        ax = jround(w / 2 * 1.16)
        ay = jround(h / 2 * 1.16)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    else:
        mask[y : y + h, x : x + w] = 255
    return mask


def _composite(img: np.ndarray, styled: np.ndarray, mask: np.ndarray) -> None:
    """Copy pixels from *styled* into *img* where mask == 255, in-place."""
    img[mask == 255] = styled[mask == 255]

# ---------------------------------------------------------------------------
# style_region — TS styleRegion L156-193 (7 styles)
# ---------------------------------------------------------------------------

def style_region(
    img: np.ndarray,
    canvas: np.ndarray,
    rect: dict,
    opts: dict,
    rand,
) -> None:
    """Apply a censor style to *rect* in *img* (RGB uint8, HWC), in-place.

    *canvas* is the unmodified original image used for source sampling.
    *rand* is a callable () -> float produced by lcg().
    Pixels outside the rect (or its ellipse mask) are never touched.
    """
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    if w < 2 or h < 2:
        return

    H_img, W_img = img.shape[:2]
    mask = shape_mask(rect, W_img, H_img)
    style = opts.get("style", "mosaic")

    # ------------------------------------------------------------------
    # bar — solid fill with barColor
    # ------------------------------------------------------------------
    if style == "bar":
        color = parse_color(opts.get("barColor", "#0a0a0a"))
        img[mask == 255] = color

    # ------------------------------------------------------------------
    # mosaic — pixelate: downsample INTER_AREA → upsample INTER_NEAREST
    # TS L162-170
    # ------------------------------------------------------------------
    elif style == "mosaic":
        longest = max(w, h)
        tile = opts.get("mosaicTile")
        if tile:
            # Absolute pixel tile size. JP export presets mandate a fixed tile
            # (longSide/100 etc.) that must NOT shrink for smaller regions.
            tile = max(2, jround(tile))
            sw = max(2, jround(w / tile))
            sh = max(2, jround(h / tile))
        else:
            blocks = max(3, jround(opts.get("mosaicBlocks", 10)))
            sw = max(2, jround((w / longest) * blocks))
            sh = max(2, jround((h / longest) * blocks))
        region = canvas[y : y + h, x : x + w]
        small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        temp = img.copy()
        temp[y : y + h, x : x + w] = big
        _composite(img, temp, mask)

    # ------------------------------------------------------------------
    # blur — Gaussian blur, sigma = blur_radius()
    # TS L171-175
    # ------------------------------------------------------------------
    elif style == "blur":
        sigma = blur_radius(w, h, opts.get("blurStrength", 70))
        region = canvas[y : y + h, x : x + w].copy()
        blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=sigma)
        temp = img.copy()
        temp[y : y + h, x : x + w] = blurred
        _composite(img, temp, mask)

    # ------------------------------------------------------------------
    # barsV / barsH / manga — bar patterns (TS drawBars L102-124)
    # ------------------------------------------------------------------
    elif style in ("barsV", "barsH", "manga"):
        color = np.array(parse_color(opts.get("barColor", "#0a0a0a")), dtype=np.uint8)
        count = max(2, jround(opts.get("barCount", 9)))
        # Bar size along the period axis (width for barsV, height for barsH/manga).
        thickness = min(1.0, max(0.05, float(opts.get("barThickness", 0.55))))
        # Start from the canvas so image peeks through gaps
        temp = canvas.copy()

        if style == "barsV":
            # Vertical bars: period along x, bar=55% of period, full height
            period = w / count
            bw_bar = period * thickness
            for i in range(count):
                x0 = jround(x + i * period + (period - bw_bar) / 2)
                x1 = jround(x0 + bw_bar)
                x0 = max(x, min(x + w, x0))
                x1 = max(x, min(x + w, x1))
                if x0 < x1:
                    temp[y : y + h, x0:x1] = color

        elif style == "barsH":
            # Horizontal bars: period along y, bar=55% of period, full width
            period = h / count
            bh_bar = period * thickness
            for i in range(count):
                y0 = jround(y + i * period + (period - bh_bar) / 2)
                y1 = jround(y0 + bh_bar)
                y0 = max(y, min(y + h, y0))
                y1 = max(y, min(y + h, y1))
                if y0 < y1:
                    temp[y0:y1, x : x + w] = color

        else:  # manga
            # Horizontal bars + gap slivers displaced ±12% sideways
            period = h / count
            bh_bar = period * thickness
            for i in range(count):
                y0 = y + i * period
                gap_y = y0 + bh_bar
                gap_h = period - bh_bar
                if gap_h > 0.5:
                    off = jround((1 if i % 2 == 0 else -1) * w * 0.12)
                    src_y1 = max(0, jround(gap_y))
                    src_y2 = min(H_img, jround(gap_y + gap_h))
                    if src_y1 < src_y2:
                        sliver = canvas[src_y1:src_y2, x : x + w].copy()
                        dst_x = x + off
                        sc = max(0, -dst_x)          # = max(0, -(x + off)); clamp of dst start
                        dc = max(0, dst_x)           # dst col start (clamped)
                        de = min(W_img, dst_x + w)   # dst col end (clamped)
                        cw = de - dc
                        if cw > 0 and sc + cw <= w:
                            temp[src_y1:src_y2, dc:de] = sliver[:, sc : sc + cw]
                # Draw bar over the gap
                bar_y1 = max(0, jround(y0))
                bar_y2 = min(H_img, jround(y0 + bh_bar))
                if bar_y1 < bar_y2:
                    temp[bar_y1:bar_y2, x : x + w] = color

        _composite(img, temp, mask)

    # ------------------------------------------------------------------
    # frosted — blur + white veil + fine grain (frosted-glass look)
    # ------------------------------------------------------------------
    elif style == "frosted":
        sigma = blur_radius(w, h, opts.get("frostAmount", 60))
        region = canvas[y : y + h, x : x + w].astype(np.float32)
        blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=sigma)
        veil = 0.35  # white veil opacity
        out = blurred * (1.0 - veil) + 255.0 * veil
        rng = np.random.default_rng(int(opts.get("glitchSeed", 7)))
        out += rng.normal(0.0, 6.0, out.shape)  # subtle deterministic grain
        temp = img.copy()
        temp[y : y + h, x : x + w] = np.clip(out, 0, 255).astype(np.uint8)
        _composite(img, temp, mask)

    # ------------------------------------------------------------------
    # static — TV static noise fill (mono/colour, optional scanlines)
    # ------------------------------------------------------------------
    elif style == "static":
        intensity = min(1.0, max(0.1, opts.get("staticIntensity", 100) / 100.0))
        rng = np.random.default_rng(int(opts.get("glitchSeed", 7)))
        region = canvas[y : y + h, x : x + w].astype(np.float32)
        if opts.get("staticMono", True):
            n = rng.integers(0, 256, (h, w, 1)).astype(np.float32)
            noise = np.repeat(n, 3, axis=2)
        else:
            noise = rng.integers(0, 256, (h, w, 3)).astype(np.float32)
        if opts.get("staticScanlines", False):
            noise[::2, :, :] *= 0.5      # darken every other row → CRT scanlines
        out = noise * intensity + region * (1.0 - intensity)
        temp = img.copy()
        temp[y : y + h, x : x + w] = np.clip(out, 0, 255).astype(np.uint8)
        _composite(img, temp, mask)

    # ------------------------------------------------------------------
    # glitch — chromatic-shift + random slice displacement  TS L176-192
    # ------------------------------------------------------------------
    else:
        k = opts.get("glitchIntensity", 70) / 100
        sx = jround(w * 0.06 * k) + 1
        region = canvas[y : y + h, x : x + w].astype(np.float32)

        # Start from a float copy of the current img for additive blending
        temp = img.astype(np.float32)

        def _add_region_shifted(shift_x: int) -> None:
            """Additively blend region into temp shifted by shift_x cols, alpha=0.85."""
            dst_x = x + shift_x
            sc = max(0, -dst_x)          # = max(0, -(x + shift_x)); clamp of dst start
            dc = max(0, dst_x)           # destination column start (clamped)
            de = min(W_img, dst_x + w)   # destination column end (clamped)
            cw = de - dc
            if cw <= 0 or sc + cw > w:
                return
            reg_y1, reg_y2 = 0, h
            dst_y1, dst_y2 = max(0, y), min(H_img, y + h)
            if dst_y1 >= dst_y2:
                return
            ry1 = dst_y1 - y
            ry2 = dst_y2 - y
            temp[dst_y1:dst_y2, dc:de] = np.clip(
                temp[dst_y1:dst_y2, dc:de] + region[ry1:ry2, sc : sc + cw] * 0.85,
                0, 255,
            )

        _add_region_shifted(-sx)
        _add_region_shifted(sx)

        # Random horizontal slice displacements from canvas (source-over)
        slices = jround((4 + rand() * 10) * (0.4 + k))
        for _ in range(int(slices)):
            sy = y + int(rand() * h)
            sh_s = 2 + int(rand() * (h * 0.12))
            off = jround((rand() - 0.5) * w * 0.5 * k)
            src_y1 = max(0, sy)
            src_y2 = min(H_img, sy + sh_s)
            if src_y1 >= src_y2:
                continue
            dst_x = x + off
            sc = max(0, -dst_x)          # = max(0, -(x + off)); clamp of dst start
            dc = max(0, dst_x)
            de = min(W_img, dst_x + w)
            cw = de - dc
            if cw > 0 and sc + cw <= w:
                temp[src_y1:src_y2, dc:de] = (
                    canvas[src_y1:src_y2, x + sc : x + sc + cw].astype(np.float32)
                )

        temp_u8 = np.clip(temp, 0, 255).astype(np.uint8)
        _composite(img, temp_u8, mask)

# ---------------------------------------------------------------------------
# _glitch_whole — TS glitchCanvas L196-212
# ---------------------------------------------------------------------------

def _glitch_whole(img_rgb: np.ndarray, intensity: float, rand, gray: bool = False) -> np.ndarray:
    """Whole-image grayscale + horizontal band tear glitch.

    Port of glitchCanvas(ctx, src, W, H, intensity, rand, gray=false).
    Each band is a horizontal slice shifted by `shift` pixels (wrap-around),
    implemented as np.roll which matches the TS ((x-shift)%W+W)%W formula.
    """
    H, W = img_rgb.shape[:2]
    out = img_rgb.copy()

    if gray:
        luma = (
            out[:, :, 0].astype(np.float32) * 0.299 +
            out[:, :, 1].astype(np.float32) * 0.587 +
            out[:, :, 2].astype(np.float32) * 0.114
        ).astype(np.uint8)
        out[:, :, 0] = luma
        out[:, :, 1] = luma
        out[:, :, 2] = luma

    src = out.copy()
    k = max(0.1, intensity / 100)
    bands = jround((10 + rand() * 26) * k)

    for _ in range(int(bands)):
        y0 = int(rand() * H)
        bh = 1 + int(rand() * 26 * k)
        shift = jround((rand() - 0.5) * W * 0.4 * k)
        y1 = min(H, y0 + bh)
        if y0 >= y1:
            continue
        # np.roll(arr, shift, axis=1): output[y,x] = input[y, (x-shift)%W]
        # matches TS: sx = ((x - shift) % W + W) % W; out[y,x] = src[y, sx]
        out[y0:y1] = np.roll(src[y0:y1], shift, axis=1)

    return out


# ---------------------------------------------------------------------------
# style_whole — TS styleWhole L127-149
# ---------------------------------------------------------------------------

def style_whole(img_rgb: np.ndarray, opts: dict, rand) -> np.ndarray:
    """Apply style across the whole image. Returns a new numpy array.

    Mosaic block size: max(3, jround(max(W,H) / (mosaicBlocks * 3.2))).
    Used for the brush-mask compositing path.
    """
    H, W = img_rgb.shape[:2]
    style = opts.get("style", "mosaic")

    if style == "bar":
        out = np.empty_like(img_rgb)
        out[:] = parse_color(opts.get("barColor", "#0a0a0a"))
        return out

    elif style in ("barsV", "barsH", "manga"):
        out = img_rgb.copy()
        rect = {"x": 0, "y": 0, "w": W, "h": H, "ellipse": False}
        style_region(out, img_rgb, rect, opts, rand)
        return out

    elif style == "mosaic":
        tile = opts.get("mosaicTile")
        if tile:
            block = max(2, jround(tile))   # absolute tile (JP export presets)
        else:
            block = max(3, jround(max(W, H) / (opts.get("mosaicBlocks", 10) * 3.2)))
        sw = max(2, jround(W / block))
        sh = max(2, jround(H / block))
        small = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        return big

    elif style == "blur":
        sigma = blur_radius(W, H, opts.get("blurStrength", 70))
        return cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma)

    elif style == "frosted":
        sigma = blur_radius(W, H, opts.get("frostAmount", 60))
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma).astype(np.float32)
        out = blurred * 0.65 + 255.0 * 0.35
        rng = np.random.default_rng(int(opts.get("glitchSeed", 7)))
        out += rng.normal(0.0, 6.0, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    elif style == "static":
        intensity = min(1.0, max(0.1, opts.get("staticIntensity", 100) / 100.0))
        rng = np.random.default_rng(int(opts.get("glitchSeed", 7)))
        if opts.get("staticMono", True):
            n = rng.integers(0, 256, (H, W, 1)).astype(np.float32)
            noise = np.repeat(n, 3, axis=2)
        else:
            noise = rng.integers(0, 256, (H, W, 3)).astype(np.float32)
        if opts.get("staticScanlines", False):
            noise[::2, :, :] *= 0.5
        out = noise * intensity + img_rgb.astype(np.float32) * (1.0 - intensity)
        return np.clip(out, 0, 255).astype(np.uint8)

    else:  # glitch
        return _glitch_whole(img_rgb, opts.get("glitchIntensity", 70), rand)


# ---------------------------------------------------------------------------
# stagger_label_positions — greedy label declutter (shared by both renderers)
# ---------------------------------------------------------------------------

def stagger_label_positions(label_boxes: list, W: int, H: int, pad: int = 2) -> list:
    """Collision-avoided top-left (lx, ly) for each detection label's background.

    Two boxes that overlap (e.g. two FEMALE_BREAST_COVERED side by side) would
    otherwise stamp their labels on top of each other. This settles labels
    top-to-bottom and pushes any that collide down below the one they hit, so
    every label stays readable.

    label_boxes : list of {box_x, box_y, lw, lh} in the SAME order as the boxes
        being drawn. box_x/box_y = detection rect top-left; lw/lh = label
        background size (text + its own padding).
    Returns a list of (lx, ly) in input order.
    """
    placed = []          # already-claimed label rects: (x1, y1, x2, y2)
    pos = {}
    # Settle higher boxes first so labels cascade downward predictably.
    order = sorted(range(len(label_boxes)),
                   key=lambda i: (label_boxes[i]["box_y"], label_boxes[i]["box_x"]))
    for i in order:
        lb = label_boxes[i]
        lw, lh = lb["lw"], lb["lh"]
        lx = max(0, min(W - lw, lb["box_x"]))
        ly = lb["box_y"] - lh - pad           # preferred: just above the box
        if ly < 0:
            ly = lb["box_y"] + pad            # no room above → drop inside top edge
        for _ in range(64):                   # push below whatever it overlaps
            r = (lx, ly, lx + lw, ly + lh)
            hit = next((p for p in placed
                        if not (r[2] <= p[0] or r[0] >= p[2]
                                or r[3] <= p[1] or r[1] >= p[3])), None)
            if hit is None:
                break
            ly = hit[3] + 1
        ly = max(0, min(ly, H - lh))           # keep on-canvas
        placed.append((lx, ly, lx + lw, ly + lh))
        pos[i] = (lx, ly)
    return [pos[i] for i in range(len(label_boxes))]


# ---------------------------------------------------------------------------
# draw_frames — TS drawFrames L218-233 (+ label declutter)
# ---------------------------------------------------------------------------

def draw_frames(img_rgb: np.ndarray, boxes: list, opts: dict) -> None:
    """Draw detection box outlines and optional labels onto img_rgb in-place.

    Sensitive boxes: #ff3b3b (255, 59, 59). Other: #39d353 (57, 211, 83).
    Label format: "LABEL  NN%" (double space before percent). Labels are
    decluttered via stagger_label_positions so overlapping boxes don't collide.
    """
    H, W = img_rgb.shape[:2]
    lw = max(2, jround(min(W, H) / 320))
    fs = max(11, jround(min(W, H) / 55))

    # 1) outlines first (so later label fills sit on top of every box edge)
    for b in boxes:
        x = int(b["x1"] * W)
        y = int(b["y1"] * H)
        bw = int((b["x2"] - b["x1"]) * W)
        bh = int((b["y2"] - b["y1"]) * H)
        col_rgb = (255, 59, 59) if b.get("sensitive", False) else (57, 211, 83)
        cv2.rectangle(img_rgb, (x, y), (x + bw, y + bh), col_rgb, lw)

    if not opts.get("frameLabels", True):
        return

    # 2) measure every label, then declutter their positions
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = max(0.5, fs / 18.0)
    thickness = max(1, lw // 2)
    labels = [f"{b['label']}  {jround(b['score'] * 100)}%" for b in boxes]
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in labels]
    lboxes = [{"box_x": int(b["x1"] * W), "box_y": int(b["y1"] * H),
               "lw": tw + 8, "lh": th + 6}
              for b, (tw, th) in zip(boxes, sizes)]
    positions = stagger_label_positions(lboxes, W, H)

    # 3) label background + text at the decluttered positions
    for b, label, (tw, th), (lx, ly) in zip(boxes, labels, sizes, positions):
        col_rgb = (255, 59, 59) if b.get("sensitive", False) else (57, 211, 83)
        cv2.rectangle(img_rgb, (lx, ly), (lx + tw + 8, ly + th + 6), col_rgb, -1)
        cv2.putText(img_rgb, label, (lx + 4, ly + th + 2), font, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# apply_auto_censor — TS applyAutoCensor L243-287
# ---------------------------------------------------------------------------

def apply_auto_censor(pil, boxes: list, opts: dict, manual_mask=None):
    """Orchestrate censoring on a PIL image. Returns PIL.Image.

    Modes
    -----
    stylize  — apply bg effect to whole image, reveal original inside each box.
    censor   — (default) obscure each box region; also apply brush mask if given.

    manual_mask : PIL.Image "L" mode mask — pixels > 127 are styled via
                  style_whole() (whole-image style composited where painted).
    Empty boxes AND no manual_mask → returns original unchanged.
    """
    from PIL import Image as _PILImage

    img_rgb = np.asarray(pil.convert("RGB")).copy()
    H, W = img_rgb.shape[:2]

    # Fast path: nothing to do
    if not boxes and manual_mask is None:
        return _PILImage.fromarray(img_rgb)

    rand = lcg(opts.get("glitchSeed", 7))
    mode = opts.get("mode", "censor")

    if mode == "stylize":
        orig = img_rgb.copy()
        bg_effect = opts.get("bgEffect", "glitch")
        bg_intensity = opts.get("bgIntensity", 70)
        # "reverse <effect>" inverts which side gets the effect: the REGIONS are
        # effected and the background stays original (instead of vice-versa).
        reverse = bg_effect.startswith("reverse")
        effect = bg_effect[len("reverse "):] if reverse else bg_effect

        # Build the effected version of the whole image once.
        styled = orig.copy()
        if effect == "grayscale":
            luma = (
                orig[:, :, 0].astype(np.float32) * 0.299 +
                orig[:, :, 1].astype(np.float32) * 0.587 +
                orig[:, :, 2].astype(np.float32) * 0.114
            ).astype(np.uint8)
            styled[:, :, 0] = luma
            styled[:, :, 1] = luma
            styled[:, :, 2] = luma
        elif effect == "glitch":
            styled = _glitch_whole(orig, bg_intensity, rand, gray=True)
        elif effect == "blur":
            sigma = blur_radius(W, H, bg_intensity)
            styled = cv2.GaussianBlur(orig, (0, 0), sigmaX=sigma)
        # effect == "none": styled stays == orig

        if reverse:
            img_rgb[:] = orig      # background original; regions get the effect
            fill = styled
        else:
            img_rgb[:] = styled    # background effected; regions revealed original
            fill = orig

        for b in boxes:
            bx = max(0, jround(b["x1"] * W))
            by = max(0, jround(b["y1"] * H))
            bw = min(W - bx, jround((b["x2"] - b["x1"]) * W))
            bh = min(H - by, jround((b["y2"] - b["y1"]) * H))
            if bw < 2 or bh < 2:
                continue
            img_rgb[by:by + bh, bx:bx + bw] = fill[by:by + bh, bx:bx + bw]

    else:
        # Censor mode (default)
        canvas = img_rgb.copy()  # pristine source for all sampling

        # --- region censor for detected boxes ---
        if boxes:
            gap = jround(opts.get("mergeGap", 30) * max(1, min(W, H) / 1000))
            shape = opts.get("shape", "auto")
            rects = [
                box_to_rect(b, W, H, opts.get("padding", 0.08), shape)
                for b in boxes
            ]
            if shape != "ellipse":
                rects = merge_rects(rects, gap)
            for r in rects:
                style_region(img_rgb, canvas, r, opts, rand)

        # --- brush-mask compositing ---
        if manual_mask is not None:
            mask_pil = manual_mask.convert("L").resize((W, H), _PILImage.NEAREST)
            mask_arr = np.asarray(mask_pil)
            styled = style_whole(canvas, opts, rand)
            m = mask_arr > 127
            img_rgb[m] = styled[m]

    if opts.get("boxFrames", False):
        draw_frames(img_rgb, boxes, opts)

    return _PILImage.fromarray(img_rgb)


# ---------------------------------------------------------------------------
# export_preset — CensorExportModal.tsx L33-40
# ---------------------------------------------------------------------------

_EXPORT_PRESETS = {
    # key → (style, divisor, minTile, fmt, dpi, master, censored)
    "dlsite":  ("mosaic", 100, 4, "jpg", 300, False, True),
    "fanza":   ("mosaic",  58, 4, "jpg", 300, False, True),
    "pixiv":   ("mosaic", 100, 4, "png",  72, False, True),
    "bar":     ("bar",    100, 4, "png",  72, False, True),
    "master":  ("mosaic", 100, 4, "png",  72, True,  False),
    "both":    ("mosaic", 100, 4, "jpg", 300, True,  True),
}


def export_preset(pil, boxes: list, preset: str, manual_mask=None) -> list:
    """Return list of (name, PIL.Image, fmt, dpi) for the given preset key.

    Preset keys (case-insensitive): DLsite, FANZA, Pixiv, Bar, Master, Both.
    Mosaic tile rule: tileSize = max(minTile, jround(longSide / divisor)).
    Names contain 'master' for uncensored output, 'censored' for censored.
    """
    from PIL import Image as _PILImage

    W, H = pil.size
    long_side = max(W, H)

    key = preset.lower()
    if key not in _EXPORT_PRESETS:
        return []

    style, divisor, min_tile, fmt, dpi, do_master, do_censored = _EXPORT_PRESETS[key]

    def _make_censored(censor_style: str, div: int, mt: int) -> "_PILImage.Image":
        # JP presets mandate an ABSOLUTE mosaic tile (longSide/divisor, min mt px).
        # Pass it as mosaicTile so the tile stays fixed regardless of region size.
        # The old code converted it to relative mosaicBlocks, which shrank the tile
        # for any region smaller than the whole image — so it barely censored.
        tile_size = max(mt, jround(long_side / div))
        censor_opts = {
            **AUTO_CENSOR_DEFAULTS,
            "style": censor_style,
            "mosaicTile": tile_size,
        }
        return apply_auto_censor(pil, boxes, censor_opts, manual_mask)

    # "Both" → master PNG/72 + censored JPG/300 (DLsite tile rule: divisor 100, minTile 4)
    if key == "both":
        master_img = pil.convert("RGB")
        censored_img = _make_censored("mosaic", 100, 4)
        return [
            ("master", master_img, "png", 72),
            ("censored", censored_img, "jpg", 300),
        ]

    results = []
    if do_master:
        results.append(("master", pil.convert("RGB"), fmt, dpi))
    if do_censored:
        results.append(("censored", _make_censored(style, divisor, min_tile), fmt, dpi))
    return results
