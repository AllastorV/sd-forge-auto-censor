# Auto-Censor — Forge Neo extension + tab (Phase 1) — Design

**Date:** 2026-06-28
**Target:** sd-webui-forge-classic-neo (Forge Classic Neo, Python 3.13, torch 2.11+cu130, gradio 4.40, onnxruntime-gpu)
**Status:** Approved design, ready for implementation plan
**Source to port:** `C:\Users\cavas\Desktop\Programlar\photo-editor` (Electron/TS) — DETECT + AUTO-CENSOR features, ported to Python/Gradio with **no feature loss**.

## Goal

Add a new **🔞 Censor** tab to Forge that detects NSFW/body regions (NudeNet) and applies
auto-censoring (mosaic/blur/bars/manga/glitch/bar, with shapes, padding, stylize, export presets),
plus a **brush** to paint custom mask regions. Generated images from txt2img/img2img can be sent to
this tab with one button. User-friendly two-column layout; full feature parity with the source's
auto-censor (manual decorative editor, DWPose, SAM are deferred to later phases — sequenced, not cut).

## Phase 1 scope (this spec)

- **Detection:** NudeNet (`nudenet.onnx`, 18 classes) — full decode, NMS, normalized boxes + class metadata.
- **Auto-censor engine:** modes `censor` / `stylize`; styles `mosaic, blur, bar, barsV, barsH, manga, glitch`; shapes `auto, rect, ellipse`; padding, mergeGap, box frames (+labels); export presets (DLsite/FANZA/Pixiv/Bar/Master/Both).
- **Manual brush mask:** paint freeform regions → censored with the chosen style (union with detected regions).
- **UI:** Gradio tab (two-column full layout, below), class-level toggles + quick filters, brush via `gr.ImageEditor`.
- **Cross-tab:** "Send to Censor" buttons in txt2img + img2img → load generated image into the tab + switch.
- **Model:** bundle `nudenet.onnx` (~12 MB) in the extension (copied from the photo-editor `models/`).

**Deferred (later phases, no feature loss — just ordering):** DWPose detection (eyes/armpits/feet/soles), MobileSAM `mask` shape, 9 decorative shapes + manual per-box drag editor, batch-folder processing.

## Source → port mapping (read these when implementing)

| Source (photo-editor) | Port target | Notes |
|---|---|---|
| `src/main/vision/nudenet.ts` | `scripts/nudenet_detect.py` | classes L15–21; thresholds L99–108; preprocess (letterbox 320, /255, CHW RGB); decode `[1,22,2100]` |
| `src/renderer/utils/autoCensor.ts` | `scripts/censor_engine.py` | defaults L28–44; `styleRegion` L156–193; `mergeRects`; `lcg`; stylize |
| `src/renderer/utils/mosaic.ts` `gaussianBlur.ts` | `censor_engine.py` | exact algorithms |
| `src/renderer/components/CensorExportModal.tsx` | `censor_engine.py` (`export_preset`) | presets L33–40 |
| `models/nudenet.onnx` | `models/nudenet.onnx` (bundled) | 12 MB |

## Locked decisions

| Decision | Value |
|---|---|
| Detector | NudeNet `nudenet.onnx`, input `[1,3,320,320]` float32 RGB, letterbox, /255, CHW |
| Detect thresholds | scoreThreshold 0.22 (sensitive), bodyPartThreshold 0.1 (armpit/feet); NMS IoU 0.45 by category |
| Styles | mosaic(blocks 10), blur(strength 70), bar(#0a0a0a), barsV/barsH(count 9), manga(count 9), glitch(intensity 70, seed 7) |
| Shapes | auto (ellipse if box.ellipse else rect), rect, ellipse |
| Region | padding 0.08, mergeGap 30 |
| Modes | censor (default), stylize (bg glitch/grayscale/blur + reveal regions, bgIntensity 70) |
| Frames | boxFrames off, frameLabels on; sensitive `#ff3b3b`, other `#39d353` |
| Selection | class-level CheckboxGroup + quick filter (Exposed only/All sensitive/All/None); default = exposed checked |
| Manual mask | `gr.ImageEditor` brush (#ff2d2d, size 40) + eraser; painted area = extra region, same global style |
| Input | `gr.ImageEditor` (upload/clipboard) + "Send to Censor" from txt2img/img2img |
| Provider | onnxruntime CUDAExecutionProvider → CPU fallback |

## Component 1 — `scripts/nudenet_detect.py` (pure, no Gradio)

- `NUDENET_CLASSES` (18, verbatim from source L15–21).
- Class metadata sets: `SENSITIVE` (breast/genitalia/buttocks/anus exposed+covered), `EXPOSED` flag (label endswith `_EXPOSED`), `ELLIPSE` (genitalia/anus/breast → oval), `BODYPART` map (armpits→"armpit", feet→"feet").
- `detect(pil, score_thr=0.22, bodypart_thr=0.1) -> list[Box]`:
  - Lazy `onnxruntime.InferenceSession(["CUDAExecutionProvider","CPUExecutionProvider"])`, model at `<ext>/models/nudenet.onnx`.
  - Preprocess: letterbox to 320 (uniform scale, black pad, record scale+padX+padY), RGB, `/255`, CHW, `[1,3,320,320]`.
  - Run → `output0 [1,22,2100]`. Decode: transpose to `[2100,22]`; cols 0–3 = cx,cy,w,h (320-space); cols 4–21 = 18 class scores; per anchor `cls=argmax(scores)`, `score=scores[cls]`; threshold (bodypart classes use `bodypart_thr`, else `score_thr`); box → xyxy in 320-space → undo letterbox (subtract pad, /scale) → normalize by original W/H → clamp 0–1.
  - NMS IoU 0.45 grouped by **semantic category** (merge feet_exposed+feet_covered etc.), keep highest score per overlap.
  - `Box = {x1,y1,x2,y2 (0-1), score, class_id, label, sensitive, exposed, ellipse, bodyPart}`.

## Component 2 — `scripts/censor_engine.py` (pure, numpy/cv2/PIL)

`AUTO_CENSOR_DEFAULTS` mirrors source L28–44. Helpers `lcg(seed)`, `merge_rects(rects, gap)` (iterative union of rects within gap; ellipse rects not merged).

- `apply_auto_censor(pil, boxes, options, manual_mask=None) -> PIL.Image`:
  1. Selected boxes → pixel rects; expand by `padding` (fraction of box size); `merge_rects` (gap scaled to image, ref 1000 px) for non-ellipse.
  2. `manual_mask` (PIL "L", image-size, white=paint): connected components → each a region with its own mask slice (shape = freeform mask).
  3. For each region: build a region mask (rect / ellipse per `shape` & `box.ellipse`, or the freeform component mask); render the chosen **style** into the region bbox; composite styled pixels only where region mask = white.
  4. If `boxFrames`: draw rectangles (+labels if `frameLabels`) with sensitive/other colors.
- **Styles** (port exact from `autoCensor.ts styleRegion` L156–193, `mosaic.ts`, `gaussianBlur.ts`):
  - `mosaic` — blocks scaled to region; downscale→nearest upscale. param `mosaicBlocks` (10).
  - `blur` — Gaussian; radius = longest_side/100 × strength/100 × 28. param `blurStrength` (70).
  - `bar` — solid fill `barColor` (#0a0a0a).
  - `barsV`/`barsH` — alternating stripes; period = dim/count; bar = 55% of period. `barCount` (9), `barColor`.
  - `manga` — thick bars + slivers displaced ±12% region width. `barCount` (9).
  - `glitch` — seeded LCG scanline H-shift (±width×0.3×intensity) + R↔B channel scramble. `glitchIntensity` (70), `glitchSeed` (7).
- **Stylize mode:** apply `bgEffect` (grayscale/glitch/blur, `bgIntensity` 70) to whole image, then reveal ORIGINAL pixels inside the regions (inverse of censor).
- `export_preset(pil, boxes, preset) -> list[(name, PIL, format, dpi)]`:
  - DLsite mosaic tile `max(longSide/100, 4)`, JPG/300; FANZA `max(longSide/58,4)` JPG/300; Pixiv `/100` PNG/72; Bar solid-black PNG/72; Master = original PNG/72; Both = master + censored.

## Component 3 — `scripts/censor_tab.py` (Gradio tab — "old" full two-column layout)

`script_callbacks.on_ui_tabs` → returns one tab `🔞 Censor`. Layout:

```
┌─ 🔞 Censor ───────────────────────────────────────────────────────────────┐
│  ┌─ INPUT (gr.ImageEditor, brush+eraser) ─┐  ┌─ OUTPUT (gr.Image) ───────┐  │
│  │   upload / clipboard / paint mask      │  │   result                  │  │
│  └────────────────────────────────────────┘  └───────────────────────────┘  │
│  [🔍 Detect]  Confidence ●0.22        [✨ CENSOR]   [⬇ Download (gr.File)] │
│  ┌─ DETECTED ─────────────────────────┐  ┌─ CENSOR STYLE ───────────────┐  │
│  │ [preview gr.Image, boxes drawn]    │  │ Mode  (•Censor)(Stylize)     │  │
│  │ Quick (gr.Radio): Exposed/Sensitive│  │ Style [mosaic ▽]  Shape[auto▽]│  │
│  │       /All/None                    │  │ ▾ Style options (Accordion)  │  │
│  │ Classes (gr.CheckboxGroup,         │  │   mosaicBlocks/blur/glitch+   │  │
│  │   populated after detect:          │  │   seed/barCount/ColorPicker  │  │
│  │   "LABEL  score")                  │  │ ▾ Region: padding, mergeGap  │  │
│  │                                    │  │ ▾ Stylize: bgEffect, intensity│  │
│  │                                    │  │ ☐ Box frames  ☑ Labels       │  │
│  │                                    │  │ Export preset [None ▽]       │  │
│  └────────────────────────────────────┘  └──────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

- Components: `input = gr.ImageEditor(type="pil", sources=["upload","clipboard"], brush=gr.Brush(colors=["#ff2d2d"], default_size=40), eraser=gr.Eraser())`; `detect_btn`, `conf` slider; `preview = gr.Image(interactive=False)`; `quick = gr.Radio([...], value="Exposed only")`; `classes = gr.CheckboxGroup([])`; `mode`, `style`, `shape` dropdowns; style sliders + `gr.ColorPicker`; `padding`, `merge`; `bg_effect`, `bg_intensity`; `frames`, `labels` checkboxes; `export` dropdown; `censor_btn`; `output = gr.Image`; `download = gr.File`. A `gr.State` holds the detected boxes list.
- Flow: **Detect** → run `nudenet_detect.detect(editor["background"])` → store boxes in State, draw preview, populate `classes.choices` (exposed pre-checked). **Quick radio** `.change` → bulk-set `classes` value. **CENSOR** → filter boxes by checked classes; build manual mask from `editor["layers"]` (alpha union); `censor_engine.apply_auto_censor(...)` (or `export_preset` if preset≠None) → output + download. Painting or pressing CENSOR without a prior detect auto-runs detect first.
- Style-specific param visibility via `style.change` (show only the active style's extra control); keep simple — others stay in the accordion.

## Component 4 — Cross-tab "Send to Censor"

- Register the tab input as a paste destination with `modules.generation_parameters_copypaste` (`register_paste_params_button` + `ParamBinding(tabname="censor", source_image_component=input_editor)`).
- Add a **Send to Censor** button to the txt2img and img2img output toolbars via `on_after_component` injection (same family as the WD14 toprow-button technique): clicking copies the currently-selected gallery image into the Censor input and switches to the Censor tab (JS `switch_to_censor`).

## Data flow

```
image (upload / paint / Send-to-Censor)
  → Detect (NudeNet, GPU) → boxes in State + preview + class toggles
  → select classes (+ optional brush mask)
  → CENSOR → censor_engine (style+shape+padding/merge, or export preset)
  → output image + download
```

## Error handling

- Model missing / load failure → tab shows status error; no crash.
- No image → "load an image" status; no-op.
- No detections + no manual mask → output = input unchanged + status "nothing to censor".
- Inference / engine exception → caught, status message, original image preserved.
- CUDA unavailable → CPU fallback (slower).

## Testing

- **`nudenet_detect.py`** unit: decode a synthetic `[1,22,2100]` tensor with known anchors → assert boxes/labels/normalization/NMS (no model). Smoke: run the real model on a test image → boxes returned, scores in [0,1].
- **`censor_engine.py`** unit (no model, synthetic image + boxes): each style changes pixels only inside the region and leaves outside untouched; shape masks (rect vs ellipse corners); padding expands; merge_rects merges within gap; manual mask region censored; export preset mosaic tile sizes correct.
- **Manual integration** in Forge: tab loads; detect draws boxes; class toggles; each style; brush mask; stylize; export; Send-to-Censor from txt2img/img2img.

## Out of scope (Phase 1)

DWPose, MobileSAM `mask` shape, 9 decorative shapes + manual drag editor, per-region individual methods, batch folder. (All planned for later phases.)

## Integration risks

1. **Send-to-Censor cross-tab transfer + tab switch** — exact `parameters_copypaste` wiring / `on_after_component` injection point; resolve by reading `modules/generation_parameters_copypaste.py` and `modules/ui.py` during planning. #1 risk.
2. **NudeNet YOLO decode** (`[1,22,2100]` layout, letterbox inverse) must match source exactly or boxes drift — covered by the synthetic-tensor unit test + real-image smoke.
3. **`gr.ImageEditor` return shape** (`background`/`layers`/`composite`) across gradio 4.40 — read the painted mask from `layers` alpha; verify in a quick gradio probe during planning.
4. Style ports (glitch LCG, manga slivers, mosaic block scaling) must match source numerics — unit tests assert region-locality; visual parity checked manually.
