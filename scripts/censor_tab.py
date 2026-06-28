"""Auto-Censor Gradio tab for Forge: detect (NudeNet) + censor (engine) + brush mask."""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

from modules import script_callbacks

# Cross-tab "Send to Censor" support. In this fork the copypaste infra lives in
# modules.infotext_utils (older A1111 named it generation_parameters_copypaste).
try:
    from modules import infotext_utils as cp
except Exception:  # noqa: BLE001 - fallback for older naming
    try:
        from modules import generation_parameters_copypaste as cp
    except Exception:  # noqa: BLE001
        cp = None
try:
    from modules.ui_components import ToolButton
except Exception:  # noqa: BLE001
    ToolButton = None

_CENSOR_TABNAME = "auto_censor"
# Source galleries captured from txt2img / img2img output panels (filled by on_after_component).
_source_galleries = {}

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
BG_EFFECTS = ["glitch", "grayscale", "blur",
              "reverse glitch", "reverse grayscale", "reverse blur", "none"]
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
    painted = acc > 10
    if painted.mean() > 0.98:          # phantom full-canvas layer, not a real brush mask
        return None
    m = painted.astype(np.uint8) * 255
    return Image.fromarray(m, "L").resize((w, h), Image.Resampling.NEAREST)


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
            glitch_intensity, glitch_seed, bar_count, bar_thickness, bar_color, padding, merge_gap,
            bg_effect, bg_intensity, box_frames, frame_labels, preset, conf):
    bg = _bg_of(editor)
    if ce is None or bg is None:
        return None, None, "Load an image first."
    boxes = boxes or []
    if not boxes and nd is not None:
        boxes = nd.detect(bg, float(conf))
    checkset = set(checked or [])
    sel = [b for b in boxes if _label_str(b) in checkset] if checkset else []
    mask = _mask_of(editor, *bg.size)
    opts = {
        "mode": mode.lower(), "style": style, "shape": shape, "barColor": bar_color,
        "mosaicBlocks": int(mosaic_blocks), "blurStrength": int(blur_strength),
        "glitchIntensity": int(glitch_intensity), "glitchSeed": int(glitch_seed),
        "barCount": int(bar_count), "barThickness": float(bar_thickness),
        "padding": float(padding), "mergeGap": int(merge_gap),
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
        # Shrink the tab ~5% (Chromium zoom) and let the image panels be drag-resized
        # (grab the bottom-right corner of any of the three image boxes).
        # Only a gentle 5% shrink — no width/overflow clipping (it was cutting off
        # the ImageEditor's layer/brush toolbar). Zoom & pan are native: the editor
        # toolbar has a zoom control, and each result image has a fullscreen button.
        gr.HTML("<style>#auto_censor_tab{zoom:0.95;}</style>")
        # --- Top: three equal-sized image panels -------------------------------
        # No fixed height / equal_height: a fixed height squished the ImageEditor and
        # clipped its bottom Layer/brush toolbar. Natural sizing lets the toolbars fit
        # and shows the full image; zoom/pan are native (editor toolbar + fullscreen).
        with gr.Row():
            with gr.Column():
                inp = gr.ImageEditor(
                    label="1 · Image  (paint = extra censor mask)", type="pil",
                    sources=["upload", "clipboard"],
                    brush=gr.Brush(default_size=40, color_mode="defaults",
                                   colors=["#ff2d2d", "#39d353", "#3d7bff", "#ffffff", "#000000"]),
                    eraser=gr.Eraser(), elem_id="auto_censor_input")
                # Hidden paste target for the cross-tab "Send to Censor" buttons; its
                # .change copies the image into the ImageEditor background.
                paste_target = gr.Image(visible=False, elem_id="auto_censor_paste", type="pil")
            with gr.Column():
                preview = gr.Image(label="2 · Detected regions", interactive=False,
                                   elem_id="ac_preview")
            with gr.Column():
                out = gr.Image(label="3 · Result", interactive=False, elem_id="ac_result")
                download = gr.File(label="Download", file_count="multiple")

        # --- Prominent step buttons (Detect is easy to miss otherwise) ----------
        with gr.Row():
            detect_btn = gr.Button("🔍  Detect", variant="primary", size="lg")
            censor_btn = gr.Button("🩹  Apply Censor", variant="primary", size="lg")

        # --- All settings in one compact band (3 columns, no page scroll) -------
        with gr.Row():
            with gr.Column():
                conf = gr.Slider(0.05, 0.6, value=0.22, step=0.01, label="Detection sensitivity",
                                 info="Lower = find more (and fainter) regions. 0.22 is a good default.")
                quick = gr.Radio(QUICK, value="Exposed only", label="Quick pick",
                                 info="Bulk-select which detected regions to censor.")
                classes = gr.CheckboxGroup([], label="Regions to censor",
                                           info="Press DETECT first; only ticked regions are censored.")
            with gr.Column():
                mode = gr.Radio(["Censor", "Stylize"], value="Censor", label="Mode",
                                info="Censor = hide the regions. Stylize = effect the background, keep regions visible.")
                style = gr.Dropdown(STYLES, value="mosaic", label="Censor style",
                                    info="How regions are hidden: mosaic / blur / black bar / stripes / manga / glitch.")
                shape = gr.Dropdown(SHAPES, value="auto", label="Region shape",
                                    info="auto = oval over private parts, rectangle elsewhere.")
                # Stylize-only controls — appear right here when Mode = Stylize.
                with gr.Group(visible=False) as stylize_box:
                    bg_effect = gr.Dropdown(BG_EFFECTS, value="glitch", label="Stylize effect",
                                            info="'reverse *' effects the REGIONS instead of the background.")
                    bg_intensity = gr.Slider(0, 100, value=70, step=1, label="Stylize intensity")
                preset = gr.Dropdown(PRESETS, value="None", label="Export preset",
                                     info="JP mosaic presets (DLsite/FANZA/Pixiv) + Master/Both. Overrides the style.")
            with gr.Column():
                with gr.Accordion("Strength", open=True):
                    mosaic_blocks = gr.Slider(3, 40, value=10, step=1, label="Mosaic blocks",
                                              info="Fewer blocks = chunkier mosaic.")
                    blur_strength = gr.Slider(10, 100, value=70, step=1, label="Blur strength",
                                              info="Higher = blurrier.")
                    bar_count = gr.Slider(2, 24, value=9, step=1, label="Bar / stripe count",
                                          info="How many bars / stripes (bar / stripes / manga styles).")
                    bar_thickness = gr.Slider(0.1, 1.0, value=0.55, step=0.05, label="Bar thickness",
                                              info="Bar width (vertical) / height (horizontal & manga) as a fraction of the gap. 1.0 = solid.")
                    glitch_intensity = gr.Slider(10, 100, value=70, step=1, label="Glitch intensity")
                with gr.Accordion("More options", open=False):
                    padding = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Region padding",
                                        info="Grow each region a bit before censoring.")
                    merge_gap = gr.Slider(0, 100, value=30, step=1, label="Merge gap",
                                          info="Regions closer than this merge into one block.")
                    bar_color = gr.ColorPicker(value="#0a0a0a", label="Bar color")
                    glitch_seed = gr.Number(value=7, precision=0, label="Glitch seed",
                                            info="Change for a different random glitch pattern.")
                    with gr.Row():
                        box_frames = gr.Checkbox(value=False, label="Draw detection boxes",
                                                 info="Overlay detection rectangles on the result.")
                        frame_labels = gr.Checkbox(value=True, label="Box labels")
        status = gr.HTML("")

        detect_btn.click(_detect, [inp, conf], [boxes_state, preview, classes, status])
        quick.change(_quick, [boxes_state, quick], [classes])
        # Show the stylize controls only in Stylize mode.
        mode.change(lambda m: gr.update(visible=(m == "Stylize")), [mode], [stylize_box])
        censor_btn.click(
            _censor,
            [inp, boxes_state, classes, mode, style, shape, mosaic_blocks, blur_strength,
             glitch_intensity, glitch_seed, bar_count, bar_thickness, bar_color, padding, merge_gap,
             bg_effect, bg_intensity, box_frames, frame_labels, preset, conf],
            [out, download, status],
        )

        # When the copypaste infra drops an image into the hidden target, load it
        # into the ImageEditor background (keeps any future brush layers empty).
        def _to_editor(img):
            if img is None:
                return gr.update()
            return {"background": img, "layers": [], "composite": img}

        paste_target.change(_to_editor, [paste_target], [inp], show_progress=False)

        # Register the Censor tab as a paste destination so Forge wires the
        # "Send to Censor" buttons (image copy + JS switch_to_auto_censor).
        if cp is not None:
            try:
                cp.add_paste_fields(_CENSOR_TABNAME, paste_target, [])
            except Exception as e:  # noqa: BLE001
                print(f"[auto-censor] add_paste_fields failed: {e}")

    return [(tab, "\U0001f51e Censor", "auto_censor_tab")]


def _inject_send_button(tabname, gallery):
    """Add a 🔞 Send-to-Censor ToolButton into the current tool-button Row and
    register it with the copypaste infra (source=gallery, dest=auto_censor)."""
    label = "\U0001f51e"
    tooltip = "Send the selected image to the Censor tab."
    btn_id = f"{tabname}_send_to_auto_censor"
    if ToolButton is not None:
        btn = ToolButton(label, elem_id=btn_id, tooltip=tooltip)
    else:
        btn = gr.Button(label, elem_id=btn_id)
    # Pre-register the destination tabname so Forge's connect_paste_params_buttons()
    # never KeyErrors on it (which would break OTHER tabs' paste buttons) if the
    # Censor tab's on_ui_tabs fails to load before add_paste_fields runs. The real
    # add_paste_fields(paste_target) in on_ui_tabs overwrites this sentinel.
    try:
        if _CENSOR_TABNAME not in cp.paste_fields:
            cp.add_paste_fields(_CENSOR_TABNAME, None, [])
    except Exception:  # noqa: BLE001
        pass
    cp.register_paste_params_button(
        cp.ParamBinding(
            paste_button=btn,
            tabname=_CENSOR_TABNAME,
            source_image_component=gallery,
        )
    )


def on_after_component(component, **kwargs):
    if cp is None:
        return
    elem_id = getattr(component, "elem_id", None) or kwargs.get("elem_id")
    if not elem_id:
        return
    # Capture the txt2img / img2img output galleries (created before the buttons).
    if elem_id in ("txt2img_gallery", "img2img_gallery"):
        _source_galleries[elem_id[: -len("_gallery")]] = component
        return
    # Anchor on the existing "Send to extras" button so our button lands in the
    # same tool-button Row, immediately after it.
    for tabname in ("txt2img", "img2img"):
        if elem_id == f"{tabname}_send_to_extras":
            gallery = _source_galleries.get(tabname)
            if gallery is not None:
                try:
                    _inject_send_button(tabname, gallery)
                except Exception as e:  # noqa: BLE001
                    print(f"[auto-censor] inject send button failed ({tabname}): {e}")
            return


script_callbacks.on_after_component(on_after_component)
script_callbacks.on_ui_tabs(on_ui_tabs)
