# Auto-Censor — Stable Diffusion WebUI Forge extension

Adds a **🔞 Censor** tab to Forge that detects NSFW/body regions with **NudeNet** and
applies auto-censoring (mosaic / blur / bars / manga / glitch / bar), with shapes, a
brush for custom mask regions, a stylize mode, and Japanese export presets. Generated
images in txt2img / img2img can be sent to the tab with one button.

Ported from the photo-editor desktop app's detect + auto-censor engine, 1:1.

## Usage

1. Open the **🔞 Censor** tab. Upload an image (or click **🔞 Send to Censor** under a
   txt2img / img2img result).
2. **🔍 Detect** → detected regions are drawn and listed as class checkboxes (exposed
   pre-selected). Use the **Quick select** radio (Exposed only / All sensitive / All / None).
3. Pick **Style** + **Shape** (and tune the accordions). Optionally **paint** over the
   image — the brush stroke becomes a custom censor mask (union with the detected regions).
4. **✨ CENSOR** → the result appears on the right with a download. Or pick an **Export
   preset** (DLsite / FANZA / Pixiv mosaic-tile rules, Bar, Master, Both).

## Styles & options

- **Styles:** mosaic (blocks), blur (strength), bar (color), barsV / barsH (count),
  manga (sliced bars), glitch (intensity + seed).
- **Shapes:** auto (oval for private regions, rect otherwise), rect, ellipse.
- **Mode:** Censor (obscure regions) or Stylize (effect the background, reveal regions).
- **Region:** padding, merge gap. **Frames:** draw detection boxes + labels.

## Detection

`models/nudenet.onnx` (~12 MB, bundled) — NudeNet 320n, 18 classes. Runs on
onnxruntime (CUDA, CPU fallback). First-class metadata drives sensitive / exposed /
ellipse / body-part handling and category-grouped NMS.

## Phase 1 scope

This is Phase 1: NudeNet detection + the full auto-censor engine + brush mask + export
presets + the tab + Send-to-Censor. **Deferred to later phases** (no feature loss, just
sequencing): DWPose detection (eyes / armpits / feet / soles), MobileSAM AI-segmentation
mask shape, the 9 decorative shapes + manual drag editor, batch-folder processing.

## Components

- `scripts/nudenet_detect.py` — pure ONNX detector (letterbox, decode, NMS).
- `scripts/censor_engine.py` — pure censor engine (7 styles, shapes, stylize, brush mask, export).
- `scripts/censor_tab.py` — Gradio tab + cross-tab Send-to-Censor wiring.
- `javascript/censor_send.js` — tab-switch glue.

## Credits

- NudeNet (notAI-tech) — NSFW detector.

## Tests

- `venv\Scripts\python.exe tests\test_nudenet_detect.py` — detector decode (no model).
- `venv\Scripts\python.exe tests\test_censor_engine.py` — engine + 7 styles + orchestration.
- `*_smoke.py` — run the real model.
