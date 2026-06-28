# Auto-Censor Extension (Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Forge "🔞 Censor" tab that detects NSFW/body regions with NudeNet and auto-censors them (7 styles, shapes, padding, stylize, export presets) plus a brush for custom mask regions; with "Send to Censor" buttons in txt2img/img2img.

**Architecture:** Two pure, unit-tested modules — `nudenet_detect.py` (ONNX detector) and `censor_engine.py` (numpy/cv2/PIL censor ports of the photo-editor `autoCensor.ts`) — plus `censor_tab.py` (Gradio tab) and `censor_send.js` (cross-tab transfer). Ports the photo-editor source 1:1 (no feature loss).

**Tech Stack:** Python 3.13, onnxruntime-gpu, numpy, opencv (cv2), Pillow, gradio 4.40 (Forge).

## Global Constraints

- EXT dir: `C:\Users\cavas\Desktop\Programlar\sd-webui-forge-classic-neo\sd-webui-forge-classic-neo\extensions\sd-forge-auto-censor`
- VENV_PY: `"C:/Users/cavas/Desktop/Programlar/sd-webui-forge-classic-neo/sd-webui-forge-classic-neo/venv/Scripts/python.exe"` (run git/tests from EXT).
- SOURCE (read for exact algorithms): `C:\Users\cavas\Desktop\Programlar\photo-editor\src` — `main/vision/nudenet.ts`, `renderer/utils/autoCensor.ts`. Port 1:1.
- Model: bundle `nudenet.onnx` (copy from `C:\Users\cavas\Desktop\Programlar\photo-editor\models\nudenet.onnx`, ~12 MB) into `EXT/models/`.
- Detector: input `[1,3,320,320]` float32 RGB, letterbox (uniform scale, black pad), `/255`, CHW. Output `output0 [1,22,2100]`, indexed `o[c*2100 + a]` (c: 0–3 = cx,cy,w,h in 320-space; 4–21 = 18 class scores).
- Thresholds: sensitive `score_thr=0.22`, bodypart `bodypart_thr=0.1`. NMS IoU 0.45 keyed by `bodyPart or class_id`.
- 18 classes (verbatim), defaults (`AUTO_CENSOR_DEFAULTS`), and style formulas are given in the tasks below — copy exact numbers.
- onnxruntime providers `["CUDAExecutionProvider","CPUExecutionProvider"]`. Heavy imports lazy.
- Pure modules import NO gradio/`modules`. Tests run with VENV_PY (no pytest), print PASS + summary.
- Commits go to the git repo in EXT (identity already set: cavas / cavaskutay@gmail.com).

---

### Task 1: `nudenet_detect.py` + model bundle

**Files:**
- Create: `EXT/scripts/nudenet_detect.py`
- Copy: `EXT/models/nudenet.onnx`
- Test: `EXT/tests/test_nudenet_detect.py`

**Interfaces (produces):**
- `NUDENET_CLASSES: list[str]` (18), `SENSITIVE: dict[str,{exposed,ellipse}]`, `BODY_PARTS: dict[str,str]`.
- `letterbox_params(sw, sh, n=320) -> (scale, padX, padY, rw, rh)`
- `decode(output0_flat, sw, sh, scale, padX, padY, score_thr, bodypart_thr) -> list[Box]` (pure, no model)
- `detect(pil, score_thr=0.22, bodypart_thr=0.1) -> list[Box]`
- `Box = dict(x1,y1,x2,y2,score,class_id,label,sensitive,exposed,ellipse,bodyPart,derived)` (coords 0–1).

- [ ] **Step 1: Scaffold + copy model + git already-init**

Run (from EXT): copy the model and confirm size.
```bash
cp "C:/Users/cavas/Desktop/Programlar/photo-editor/models/nudenet.onnx" models/nudenet.onnx
ls -la models/nudenet.onnx
```
Add `*.onnx` is NOT ignored (we bundle it) — verify `.gitignore` does not exclude it.

- [ ] **Step 2: Write the failing decode unit test**

Create `EXT/tests/test_nudenet_detect.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
import nudenet_detect as nd


def test_classes_count_and_order():
    assert len(nd.NUDENET_CLASSES) == 18
    assert nd.NUDENET_CLASSES[3] == "FEMALE_BREAST_EXPOSED"
    assert nd.NUDENET_CLASSES[14] == "MALE_GENITALIA_EXPOSED"


def test_letterbox_params_square_pad():
    scale, padX, padY, rw, rh = nd.letterbox_params(1000, 500, 320)
    assert abs(scale - 0.32) < 1e-6
    assert rw == 320 and rh == 160
    assert padX == 0 and padY == 80  # centered vertical pad


def test_decode_one_box_normalized():
    # 320x320 model space, one anchor with a centered breast-exposed box.
    A, C = 2100, 18
    out = np.zeros((4 + C) * A, dtype=np.float32)
    a = 0
    out[0 * A + a] = 160.0   # cx (320-space)
    out[1 * A + a] = 160.0   # cy
    out[2 * A + a] = 64.0    # w
    out[3 * A + a] = 64.0    # h
    out[(4 + 3) * A + a] = 0.9  # class 3 score (FEMALE_BREAST_EXPOSED)
    boxes = nd.decode(out, sw=320, sh=320, scale=1.0, padX=0, padY=0,
                      score_thr=0.22, bodypart_thr=0.1)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["label"] == "FEMALE_BREAST_EXPOSED" and b["sensitive"] and b["exposed"]
    assert abs(b["x1"] - (128/320)) < 1e-4 and abs(b["x2"] - (192/320)) < 1e-4


def test_decode_threshold_filters():
    A, C = 2100, 18
    out = np.zeros((4 + C) * A, dtype=np.float32)
    out[(4 + 3) * A + 0] = 0.1   # below 0.22
    assert nd.decode(out, 320, 320, 1.0, 0, 0, 0.22, 0.1) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
```
Run: `<VENV_PY> tests/test_nudenet_detect.py` → FAIL (module missing).

- [ ] **Step 3: Implement `nudenet_detect.py`**

Port `src/main/vision/nudenet.ts` (classes L15–21; decode L130–183). Create `EXT/scripts/nudenet_detect.py`:
```python
"""NudeNet ONNX detector — pure (no gradio/webui). Ports photo-editor nudenet.ts."""
from __future__ import annotations
from pathlib import Path
import numpy as np

NUDENET_CLASSES = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
    "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
# exposed flag + ellipse (oval) flag per sensitive class
SENSITIVE = {
    "FEMALE_GENITALIA_COVERED": {"exposed": False, "ellipse": True},
    "FEMALE_GENITALIA_EXPOSED": {"exposed": True, "ellipse": True},
    "MALE_GENITALIA_EXPOSED": {"exposed": True, "ellipse": True},
    "ANUS_EXPOSED": {"exposed": True, "ellipse": True},
    "ANUS_COVERED": {"exposed": False, "ellipse": True},
    "FEMALE_BREAST_EXPOSED": {"exposed": True, "ellipse": True},
    "FEMALE_BREAST_COVERED": {"exposed": False, "ellipse": True},
    "BUTTOCKS_EXPOSED": {"exposed": True, "ellipse": False},
    "BUTTOCKS_COVERED": {"exposed": False, "ellipse": False},
}
BODY_PARTS = {
    "ARMPITS_EXPOSED": "armpit", "ARMPITS_COVERED": "armpit",
    "FEET_EXPOSED": "feet", "FEET_COVERED": "feet",
}
N = 320
_MODEL = Path(__file__).resolve().parent.parent / "models" / "nudenet.onnx"
_session = None


def letterbox_params(sw, sh, n=N):
    scale = min(n / sw, n / sh)
    rw, rh = round(sw * scale), round(sh * scale)
    padX, padY = (n - rw) // 2, (n - rh) // 2
    return scale, padX, padY, rw, rh


def _iou(a, b):
    xx1, yy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    xx2, yy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    ua = (a["x2"]-a["x1"])*(a["y2"]-a["y1"]) + (b["x2"]-b["x1"])*(b["y2"]-b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def decode(out, sw, sh, scale, padX, padY, score_thr=0.22, bodypart_thr=0.1):
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    A, C = 2100, len(NUDENET_CLASSES)
    dets = []
    for a in range(A):
        scores = out[(4 + np.arange(C)) * A + a]
        k = int(np.argmax(scores)); s = float(scores[k])
        label = NUDENET_CLASSES[k]
        bp = BODY_PARTS.get(label)
        thr = bodypart_thr if bp else score_thr
        if s < thr:
            continue
        cx, cy, w, h = float(out[a]), float(out[A+a]), float(out[2*A+a]), float(out[3*A+a])
        x1 = (((cx - w/2) - padX) / scale) / sw
        y1 = (((cy - h/2) - padY) / scale) / sh
        x2 = (((cx + w/2) - padX) / scale) / sw
        y2 = (((cy + h/2) - padY) / scale) / sh
        sens = SENSITIVE.get(label)
        dets.append({
            "x1": max(0.0, x1), "y1": max(0.0, y1), "x2": min(1.0, x2), "y2": min(1.0, y2),
            "score": s, "class_id": k, "label": label,
            "sensitive": sens is not None,
            "exposed": sens["exposed"] if sens else False,
            "ellipse": sens["ellipse"] if sens else False,
            "bodyPart": bp, "derived": False,
        })
    dets.sort(key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        key = d["bodyPart"] or str(d["class_id"])
        if any((k["bodyPart"] or str(k["class_id"])) == key and _iou(k, d) > 0.45 for k in keep):
            continue
        keep.append(d)
    for f in [b for b in keep if b["bodyPart"] == "feet"]:
        keep.append({**f, "class_id": -1, "label": "SOLES_DERIVED",
                     "sensitive": False, "exposed": False, "ellipse": False,
                     "bodyPart": "soles", "derived": True})
    return keep


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(_MODEL),
                                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _session


def detect(pil, score_thr=0.22, bodypart_thr=0.1):
    import cv2
    rgb = np.asarray(pil.convert("RGB"))
    sh, sw = rgb.shape[:2]
    scale, padX, padY, rw, rh = letterbox_params(sw, sh, N)
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((N, N, 3), dtype=np.uint8)
    canvas[padY:padY+rh, padX:padX+rw] = resized
    chw = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # [1,3,320,320]
    sess = _get_session()
    out = sess.run(["output0"], {"images": chw})[0]
    return decode(out, sw, sh, scale, padX, padY, score_thr, bodypart_thr)
```

- [ ] **Step 4: Run unit test, expect pass**

`<VENV_PY> tests/test_nudenet_detect.py` → `All 4 tests passed.`

- [ ] **Step 5: Smoke (real model) — write + run**

Create `EXT/tests/test_nudenet_smoke.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from PIL import Image
import nudenet_detect as nd

boxes = nd.detect(Image.new("RGB", (512, 768), (127, 110, 100)))
print("providers:", nd._get_session().get_providers())
print("boxes:", len(boxes))
for b in boxes[:5]:
    assert 0.0 <= b["x1"] <= 1.0 and 0.0 <= b["score"] <= 1.0
print("NUDENET SMOKE OK")
```
Run (timeout 600000): `<VENV_PY> tests/test_nudenet_smoke.py`
Expected: prints providers, a box count (may be 0 on a flat image), `NUDENET SMOKE OK`. Network/model errors → report BLOCKED with the exact error.

- [ ] **Step 6: Commit**

```bash
git add scripts/nudenet_detect.py tests/test_nudenet_detect.py tests/test_nudenet_smoke.py models/nudenet.onnx
git commit -q -m "feat: NudeNet ONNX detector + model + tests"
```

---

### Task 2: `censor_engine.py` — helpers + 7 region styles (pure)

**Files:**
- Create: `EXT/scripts/censor_engine.py`
- Test: `EXT/tests/test_censor_engine.py`

**Interfaces (produces):**
- `AUTO_CENSOR_DEFAULTS: dict` (verbatim below).
- `lcg(seed) -> callable() -> float` (deterministic PRNG, exact port).
- `merge_rects(rects, gap) -> list[Rect]` where `Rect = {x,y,w,h,ellipse}` (ints).
- `box_to_rect(box, W, H, padding, shape) -> Rect`.
- `blur_radius(w, h, strength) -> int`.
- `style_region(img, canvas, rect, opts, rand) -> None` (mutates `img` numpy RGB uint8 in place; `canvas` = original copy for sampling). Implements bar/mosaic/blur/barsV/barsH/manga/glitch.
- `shape_mask(rect, W, H) -> np.ndarray[H,W] uint8` (rect or ellipse×1.16).

Port `autoCensor.ts`: `lcg` L49–52, `mergeRects` L56–74, `boxToRect` L76–85, `clipShape` L87–92 (ellipse radius ×1.16), `drawBars` L102–124, `blurRadius` L151–153, `styleRegion` L156–193. Use cv2/numpy: canvas `drawImage(scale)` → `cv2.resize`; `imageSmoothingEnabled=false` → `cv2.INTER_NEAREST`; CSS `blur(px)` → `cv2.GaussianBlur(sigma≈px)`; clip → composite styled pixels only where `shape_mask==255`.

- [ ] **Step 1: Write failing tests**

Create `EXT/tests/test_censor_engine.py`:
```python
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
    assert img[40, 40].tolist() == [0, 0, 0]          # inside region filled
    assert img[5, 5].tolist() == [200, 200, 200]      # outside untouched


def test_mosaic_style_changes_region():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    canvas = img.copy()
    rect = {"x": 10, "y": 10, "w": 40, "h": 40, "ellipse": False}
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "mosaic", "mosaicBlocks": 4}
    ce.style_region(img, canvas, rect, opts, ce.lcg(7))
    assert not np.array_equal(img[10:50, 10:50], canvas[10:50, 10:50])  # region changed
    assert np.array_equal(img[60:, 60:], canvas[60:, 60:])              # outside same


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
```
Run: `<VENV_PY> tests/test_censor_engine.py` → FAIL (module missing).

- [ ] **Step 2: Implement helpers + styles**

Create `EXT/scripts/censor_engine.py` with `AUTO_CENSOR_DEFAULTS` (verbatim), `lcg`, `merge_rects`, `box_to_rect`, `blur_radius`, `shape_mask`, `style_region`. Port the exact algorithms from `autoCensor.ts` (lines cited above). Requirements the tests pin:
- `AUTO_CENSOR_DEFAULTS` = `{mode:"censor", style:"mosaic", shape:"auto", barColor:"#0a0a0a", mosaicBlocks:10, blurStrength:70, glitchIntensity:70, glitchSeed:7, barCount:9, padding:0.08, mergeGap:30, bgEffect:"glitch", bgIntensity:70, boxFrames:False, frameLabels:True}`.
- `lcg`: `s=(seed|0) or 1; each call s=(s*1664525+1013904223) & 0xffffffff; return (s % 100000)/100000`. (Match TS `| 0` 32-bit wrap and `>>> 0`.)
- `merge_rects`: iterative near-merge with `gap` (L56–74); merged rect `ellipse=False`; single rects keep their ellipse flag.
- `box_to_rect`: pad by `padding` fraction, clamp to image, `ellipse = shape=="ellipse" or (shape=="auto" and box.ellipse)`.
- `blur_radius(w,h,strength) = max(2, round((max(w,h)/100)*(strength/100)*28))`.
- `shape_mask(rect,W,H)`: ellipse uses semi-axes `(w/2)*1.16,(h/2)*1.16`; else filled rect. uint8 0/255.
- `style_region` per style (port L156–193): `bar` fill barColor; `mosaic` blocks=round(mosaicBlocks), sw=round((w/longest)*blocks), sh=round((h/longest)*blocks), `cv2.resize` down (INTER_AREA) then up (INTER_NEAREST); `blur` `cv2.GaussianBlur` with sigma from `blur_radius`; `barsV/barsH` period=dim/count, bar=55%; `manga` thick bars + slivers displaced ±12% width; `glitch` double draw ±sx (sx=round(w*0.06*k)+1) blended + random slices using `rand`. All clipped via `shape_mask` composite.

(The implementer translates canvas→numpy guided by the failing tests; the tests above pin region-locality and determinism. Colors parse `#rrggbb` → (r,g,b).)

- [ ] **Step 3: Run tests, expect pass**

`<VENV_PY> tests/test_censor_engine.py` → `All 7 tests passed.`

- [ ] **Step 4: Commit**

```bash
git add scripts/censor_engine.py tests/test_censor_engine.py
git commit -q -m "feat: censor engine helpers + 7 region styles"
```

---

### Task 3: `censor_engine.py` — orchestration (apply, stylize, brush mask, frames, export)

**Files:**
- Modify: `EXT/scripts/censor_engine.py`
- Test: `EXT/tests/test_censor_engine.py` (add cases)

**Interfaces (produces):**
- `apply_auto_censor(pil, boxes, opts, manual_mask=None) -> PIL.Image`
- `export_preset(pil, boxes, preset, manual_mask=None) -> list[(name, PIL.Image, fmt, dpi)]`

Port `applyAutoCensor` L243–287: censor branch (mergeGap scaled `round(mergeGap*max(1,min(W,H)/1000))`, `box_to_rect`, ellipse shapes not merged, `style_region` each); **brush mask** = the source `shape=="mask"` path L270–277 (`style_whole` then keep only where mask white — reuse for `manual_mask`); stylize L251–269 (bg grayscale/glitch/blur via `bgIntensity`, then redraw original box regions); `draw_frames` L218–233. `style_whole` = L127–149 (whole-image variant; mosaic block `max(3, round(max(W,H)/(mosaicBlocks*3.2)))`). Export presets from `CensorExportModal.tsx` L33–40.

- [ ] **Step 1: Add failing tests** (append before `__main__`):
```python
from PIL import Image


def test_apply_censor_obscures_box_region():
    img = Image.new("RGB", (200, 200), (180, 180, 180))
    box = {"x1": 0.25, "y1": 0.25, "x2": 0.5, "y2": 0.5, "score": 0.9,
           "class_id": 3, "label": "FEMALE_BREAST_EXPOSED", "sensitive": True,
           "exposed": True, "ellipse": False, "bodyPart": None, "derived": False}
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "bar", "barColor": "#000000", "shape": "rect", "padding": 0.0}
    out = np.asarray(ce.apply_auto_censor(img, [box], opts))
    assert out[80, 70].tolist() == [0, 0, 0]            # inside box censored
    assert out[10, 10].tolist() == [180, 180, 180]      # outside untouched


def test_apply_censor_brush_mask():
    img = Image.new("RGB", (100, 100), (200, 200, 200))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    opts = {**ce.AUTO_CENSOR_DEFAULTS, "style": "bar", "barColor": "#000000"}
    out = np.asarray(ce.apply_auto_censor(img, [], opts, manual_mask=Image.fromarray(mask, "L")))
    assert out[50, 50].tolist() == [0, 0, 0]            # painted area censored
    assert out[5, 5].tolist() == [200, 200, 200]        # unpainted untouched


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
```

- [ ] **Step 2: Run to verify fail** → AttributeError (apply_auto_censor missing).

- [ ] **Step 3: Implement** `apply_auto_censor`, `style_whole`, `draw_frames`, `export_preset` per the source references above. Pin: empty targets + no mask → return a copy unchanged; brush mask path applies `style_whole` then keeps only masked pixels; stylize reveals original box regions.

- [ ] **Step 4: Run tests** → `All 11 tests passed.`

- [ ] **Step 5: Commit**
```bash
git add scripts/censor_engine.py tests/test_censor_engine.py
git commit -q -m "feat: auto-censor orchestration (apply, stylize, brush mask, frames, export)"
```

---

### Task 4: `censor_tab.py` — Gradio tab

**Files:**
- Create: `EXT/scripts/censor_tab.py`

**Interfaces (consumes):** `nudenet_detect.detect`, `censor_engine.apply_auto_censor`, `censor_engine.export_preset`, `censor_engine.AUTO_CENSOR_DEFAULTS`.

Build the tab via `script_callbacks.on_ui_tabs`. Layout = the spec's two-column full layout. Components, defaults, and flow exactly as `DESIGN.md` §Component 3. Defensive sibling imports (None on failure, like prior extensions). Verified statically with `py_compile`; runtime verification deferred to the controller.

- [ ] **Step 1: Write `censor_tab.py`** — full Gradio tab:
  - sibling-path import of `nudenet_detect`, `censor_engine` (guarded).
  - `on_ui_tabs` → `with gr.Blocks() as tab: ...` returns `[(tab, "🔞 Censor", "auto_censor_tab")]`.
  - Components per DESIGN §Component 3 (ImageEditor input w/ Brush+Eraser; detect button + conf slider; preview Image; quick Radio; classes CheckboxGroup; mode/style/shape; style sliders + ColorPicker; padding/merge; bg effect/intensity; frames/labels; export dropdown; censor button; output Image; download File; `gr.State` boxes).
  - `_detect(editor, conf)`: read `editor["background"]` → `detect` → return boxes state, preview (boxes drawn via PIL), `gr.update(choices=[f"{b['label']}  {b['score']:.2f}" for b in boxes], value=[exposed ones])`.
  - `_quick(boxes, mode)` → set checkbox value by filter.
  - `_censor(editor, boxes, checked, mode, style, shape, ...all opts..., export)`: build opts dict; filter boxes by checked labels; build manual mask from `editor["layers"]` alpha-union; if export!="None" → `export_preset` (return first/zip); else `apply_auto_censor` → output + download file path (write to a temp dir).
  - Wire `.click`/`.change`. Style-specific slider visibility via `style.change` (optional; keep all in accordion if simpler).
- [ ] **Step 2: `<VENV_PY> -m py_compile scripts/censor_tab.py`** → exit 0.
- [ ] **Step 3: Commit** `git add scripts/censor_tab.py && git commit -q -m "feat: Censor Gradio tab (detect, styles, brush, export)"`

---

### Task 5: Cross-tab "Send to Censor" (`censor_send.js` + registration)

**Files:**
- Create: `EXT/javascript/censor_send.js`
- Modify: `EXT/scripts/censor_tab.py` (register paste target + expose input component id)

**Interfaces:** uses `modules.generation_parameters_copypaste` and/or `on_after_component`. The input ImageEditor must have a stable `elem_id="auto_censor_input"`.

Resolve the exact mechanism by reading `modules/generation_parameters_copypaste.py` (look for `register_paste_params_button`, `ParamBinding`, `create_buttons`) and how img2img's "Send to img2img/extras" buttons are built in `modules/ui.py`. Implement: a "🔞 Censor" send button in txt2img and img2img output toolbars that copies the selected gallery image into `#auto_censor_input` and switches to the Censor tab (JS `switch_to_tab`).

- [ ] **Step 1: Investigate** — read the copypaste module + ui send-button creation; write findings into the report.
- [ ] **Step 2: Implement** the registration in `censor_tab.py` + `censor_send.js` (button injection via `on_after_component` near the existing send buttons; click → set input image + `gradioApp()` tab switch).
- [ ] **Step 3: `py_compile` + `node --check`** both files.
- [ ] **Step 4: Commit** `git commit -q -m "feat: Send to Censor buttons in txt2img/img2img"`

(Runtime verification deferred to controller. If the copypaste API path proves infeasible, fall back to a pure-JS approach: read the live gallery image and POST to the tab input — document in the report.)

---

### Task 6: README + acceptance + final review

**Files:**
- Create: `EXT/README.md`

- [ ] **Step 1: Write README** — features, install, the tab usage, model note (`nudenet.onnx` bundled), Phase-1 scope + deferred phases, credits (NudeNet), license.
- [ ] **Step 2: Run all pure tests** — `<VENV_PY> tests/test_nudenet_detect.py` and `tests/test_censor_engine.py` → all pass.
- [ ] **Step 3: Commit** `git commit -q -m "docs: README"`.
- [ ] **Step 4: Manual Forge acceptance (controller-run)** — restart Forge → 🔞 Censor tab loads; upload → Detect draws boxes + class toggles; each style censors; brush mask; stylize; export preset; Send-to-Censor from txt2img/img2img switches tab with the image.

---

## Self-Review

**Spec coverage:** NudeNet detect+decode+NMS+soles (T1) ✓; 18 classes + metadata (T1) ✓; 7 styles + shapes + padding/merge + blur/mosaic/glitch/manga formulas (T2) ✓; apply + stylize + brush-mask + frames + export presets (T3) ✓; Gradio tab full layout + brush + class toggles + quick filter (T4) ✓; Send-to-Censor cross-tab (T5) ✓; model bundle (T1) ✓; tests (T1–T3) ✓; README (T6) ✓; deferred items absent ✓.

**Placeholder scan:** detect module fully coded; engine tasks give exact formulas + source line refs + behavior-pinning tests (TDD contract); tab/send are integration tasks with explicit component/flow specs + static gates. No "TODO/implement later".

**Type consistency:** `Box` dict schema identical across T1→T3→T4; `Rect` dict `{x,y,w,h,ellipse}` consistent T2↔T3; `apply_auto_censor(pil, boxes, opts, manual_mask=None)` signature used identically in T3 and T4; `AUTO_CENSOR_DEFAULTS` keys match the opts consumed by `style_region`/`apply_auto_censor`.
