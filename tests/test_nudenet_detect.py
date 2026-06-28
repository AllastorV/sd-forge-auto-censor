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
    assert padX == 0 and padY == 80


def test_decode_one_box_normalized():
    A, C = 2100, 18
    out = np.zeros((4 + C) * A, dtype=np.float32)
    a = 0
    out[0 * A + a] = 160.0
    out[1 * A + a] = 160.0
    out[2 * A + a] = 64.0
    out[3 * A + a] = 64.0
    out[(4 + 3) * A + a] = 0.9
    boxes = nd.decode(out, sw=320, sh=320, scale=1.0, padX=0, padY=0,
                      score_thr=0.22, bodypart_thr=0.1)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["label"] == "FEMALE_BREAST_EXPOSED" and b["sensitive"] and b["exposed"]
    assert abs(b["x1"] - (128/320)) < 1e-4 and abs(b["x2"] - (192/320)) < 1e-4


def test_decode_threshold_filters():
    A, C = 2100, 18
    out = np.zeros((4 + C) * A, dtype=np.float32)
    out[(4 + 3) * A + 0] = 0.1
    assert nd.decode(out, 320, 320, 1.0, 0, 0, 0.22, 0.1) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
