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
    chw = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    sess = _get_session()
    out = sess.run(["output0"], {"images": chw})[0]
    return decode(out, sw, sh, scale, padX, padY, score_thr, bodypart_thr)
