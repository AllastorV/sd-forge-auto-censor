"""Auto-Censor Gradio tab for Forge: detect (NudeNet) + censor (engine) + brush mask."""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

from modules import script_callbacks

_dir = str(Path(__file__).resolve().parent)
if _dir not in sys.path:
    sys.path.insert(0, _dir)
try:
    import nudenet_detect as nd
    import censor_engine as ce
except Exception as e:  # noqa: BLE001
    nd = None
    ce = None
    print(f"[auto-censor] import failed: {e}")

_OUT_DIR = Path(tempfile.gettempdir()) / "forge_auto_censor"
_OUT_DIR.mkdir(exist_ok=True)

STYLES = ["mosaic", "blur", "bar", "barsV", "barsH", "manga", "glitch"]
SHAPES = ["auto", "rect", "ellipse"]
BG_EFFECTS = ["glitch", "grayscale", "blur", "none"]
PRESETS = ["None", "DLsite", "FANZA", "Pixiv", "Bar", "Master", "Both"]
QUICK = ["Exposed only", "All sensitive", "All", "None"]


def _bg_of(editor):
    if editor is None:
        return None
    if isinstance(editor, dict):
        return editor.get("background")
    return editor


def _mask_of(editor, w, h):
    """Union of brush layers' alpha -> L mask sized (w,h), or None if empty."""
    if not isinstance(editor, dict):
        return None
    layers = editor.get("layers") or []
    acc = None
    for lyr in layers:
        if lyr is None:
            continue
        a = np.asarray(lyr.convert("RGBA"))[:, :, 3]
        acc = a if acc is None else np.maximum(acc, a)
    if acc is None or acc.max() == 0:
        return None
    m = (acc > 10).astype(np.uint8) * 255
    return Image.fromarray(m, "L").resize((w, h), Image.NEAREST)


def _draw_preview(pil, boxes):
    img = pil.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for b in boxes:
        x1, y1, x2, y2 = b["x1"] * W, b["y1"] * H, b["x2"] * W, b["y2"] * H
        col = (255, 59, 59) if b["sensitive"] else (57, 211, 83)
        d.rectangle([x1, y1, x2, y2], outline=col, width=max(2, W // 320))
        d.text((x1 + 2, max(0, y1 - 12)), f"{b['label']} {int(b['score'] * 100)}%", fill=col)
    return img


def _label_str(b):
    return f"{b['label']}  {b['score']:.2f}"


def _detect(editor, conf):
    bg = _bg_of(editor)
    if nd is None or bg is None:
        return [], None, gr.update(choices=[], value=[]), "Load an image and click Detect."
    boxes = nd.detect(bg, float(conf))
    choices = [_label_str(b) for b in boxes]
    exposed = [_label_str(b) for b in boxes if b.get("exposed")]
    return boxes, _draw_preview(bg, boxes), gr.update(choices=choices, value=exposed), f"Detected {len(boxes)} regions."


def _quick(boxes, mode):
    if not boxes:
        return gr.update(value=[])
    def pick(pred):
        return [_label_str(b) for b in boxes if pred(b)]
    if mode == "Exposed only":
        v = pick(lambda b: b.get("exposed"))
    elif mode == "All sensitive":
        v = pick(lambda b: b.get("sensitive"))
    elif mode == "All":
        v = pick(lambda b: True)
    else:
        v = []
    return gr.update(value=v)


def _censor(editor, boxes, checked, mode, style, shape, mosaic_blocks, blur_strength,
            glitch_intensity, glitch_seed, bar_count, bar_color, padding, merge_gap,
            bg_effect, bg_intensity, box_frames, frame_labels, preset):
    bg = _bg_of(editor)
    if ce is None or bg is None:
        return None, None, "Load an image first."
    boxes = boxes or []
    if not boxes and nd is not None:
        boxes = nd.detect(bg)
    checkset = set(checked or [])
    sel = [b for b in boxes if _label_str(b) in checkset] if checkset else []
    mask = _mask_of(editor, *bg.size)
    opts = {
        "mode": mode.lower(), "style": style, "shape": shape, "barColor": bar_color,
        "mosaicBlocks": int(mosaic_blocks), "blurStrength": int(blur_strength),
        "glitchIntensity": int(glitch_intensity), "glitchSeed": int(glitch_seed),
        "barCount": int(bar_count), "padding": float(padding), "mergeGap": int(merge_gap),
        "bgEffect": bg_effect, "bgIntensity": int(bg_intensity),
        "boxFrames": bool(box_frames), "frameLabels": bool(frame_labels),
    }
    ts = int(time.time() * 1000)
    if preset and preset != "None":
        outs = ce.export_preset(bg, sel, preset, manual_mask=mask)
        files, first = [], None
        for name, im, fmt, dpi in outs:
            p = _OUT_DIR / f"censor_{ts}_{name}.{fmt}"
            im.save(p, dpi=(dpi, dpi))
            files.append(str(p))
            if first is None:
                first = im
        return first, files, f"Exported {preset}: {len(files)} file(s)."
    out = ce.apply_auto_censor(bg, sel, opts, manual_mask=mask)
    p = _OUT_DIR / f"censor_{ts}.png"
    out.save(p)
    if sel or mask is not None:
        msg = f"Censored {len(sel)} region(s)" + (" + brush mask" if mask is not None else "")
    else:
        msg = "Nothing to censor."
    return out, [str(p)], msg


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        boxes_state = gr.State([])
        with gr.Row():
            with gr.Column():
                inp = gr.ImageEditor(label="Input / Paint mask", type="pil",
                                     sources=["upload", "clipboard"],
                                     brush=gr.Brush(colors=["#ff2d2d"], default_size=40),
                                     eraser=gr.Eraser(), elem_id="auto_censor_input")
                with gr.Row():
                    detect_btn = gr.Button("🔍 Detect")
                    conf = gr.Slider(0.05, 0.6, value=0.22, step=0.01, label="Confidence")
                preview = gr.Image(label="Detected", interactive=False)
                quick = gr.Radio(QUICK, value="Exposed only", label="Quick select")
                classes = gr.CheckboxGroup([], label="Classes to censor")
            with gr.Column():
                out = gr.Image(label="Result", interactive=False)
                with gr.Row():
                    censor_btn = gr.Button("✨ CENSOR", variant="primary")
                    download = gr.File(label="Download")
                mode = gr.Radio(["Censor", "Stylize"], value="Censor", label="Mode")
                with gr.Row():
                    style = gr.Dropdown(STYLES, value="mosaic", label="Style")
                    shape = gr.Dropdown(SHAPES, value="auto", label="Shape")
                with gr.Accordion("Style options", open=False):
                    mosaic_blocks = gr.Slider(3, 40, value=10, step=1, label="Mosaic blocks")
                    blur_strength = gr.Slider(10, 100, value=70, step=1, label="Blur strength")
                    glitch_intensity = gr.Slider(10, 100, value=70, step=1, label="Glitch intensity")
                    glitch_seed = gr.Number(value=7, precision=0, label="Glitch seed")
                    bar_count = gr.Slider(2, 24, value=9, step=1, label="Bar count")
                    bar_color = gr.ColorPicker(value="#0a0a0a", label="Bar color")
                with gr.Accordion("Region", open=False):
                    padding = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Padding")
                    merge_gap = gr.Slider(0, 100, value=30, step=1, label="Merge gap")
                with gr.Accordion("Stylize background", open=False):
                    bg_effect = gr.Dropdown(BG_EFFECTS, value="glitch", label="Bg effect")
                    bg_intensity = gr.Slider(0, 100, value=70, step=1, label="Bg intensity")
                with gr.Row():
                    box_frames = gr.Checkbox(value=False, label="Box frames")
                    frame_labels = gr.Checkbox(value=True, label="Labels")
                preset = gr.Dropdown(PRESETS, value="None", label="Export preset")
                status = gr.HTML("")

        detect_btn.click(_detect, [inp, conf], [boxes_state, preview, classes, status])
        quick.change(_quick, [boxes_state, quick], [classes])
        censor_btn.click(
            _censor,
            [inp, boxes_state, classes, mode, style, shape, mosaic_blocks, blur_strength,
             glitch_intensity, glitch_seed, bar_count, bar_color, padding, merge_gap,
             bg_effect, bg_intensity, box_frames, frame_labels, preset],
            [out, download, status],
        )
    return [(tab, "\U0001f51e Censor", "auto_censor_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
