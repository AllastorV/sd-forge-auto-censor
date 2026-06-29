"""Generate the built-in censor stickers into stickers/builtin/ (RGBA PNGs).

Run once at build time:
    venv/Scripts/python.exe scripts/_gen_builtin_stickers.py
Output is committed so the feature works on a fresh clone.
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "stickers" / "builtin"
OUT.mkdir(parents=True, exist_ok=True)
S = 256  # canvas size


def _font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def heart():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = S * 0.24
    d.ellipse([S*0.18, S*0.22, S*0.18 + 2*r, S*0.22 + 2*r], fill=(230, 30, 60, 255))
    d.ellipse([S*0.52, S*0.22, S*0.52 + 2*r, S*0.22 + 2*r], fill=(230, 30, 60, 255))
    d.polygon([(S*0.12, S*0.46), (S*0.88, S*0.46), (S*0.5, S*0.9)], fill=(230, 30, 60, 255))
    im.save(OUT / "heart.png")


def star():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = cy = S / 2
    R, r = S * 0.46, S * 0.19
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = R if i % 2 == 0 else r
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=(255, 200, 30, 255))
    im.save(OUT / "star.png")


def censored_tape():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, S*0.36, S, S*0.64], fill=(15, 15, 15, 255))
    f = _font(int(S * 0.16))
    t = "CENSORED"
    l, tp, rr, bo = d.textbbox((0, 0), t, font=f)
    d.text(((S - (rr - l)) / 2, (S - (bo - tp)) / 2 - tp), t, fill=(245, 245, 245, 255), font=f)
    im.save(OUT / "censored.png")


def prohibited():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    w = int(S * 0.10)
    d.ellipse([w, w, S - w, S - w], outline=(220, 30, 30, 255), width=w)
    # diagonal slash
    import math as _m
    cx = cy = S / 2
    r = (S - 2 * w) / 2
    ox, oy = r * _m.cos(_m.radians(45)), r * _m.sin(_m.radians(45))
    d.line([(cx - ox, cy + oy), (cx + ox, cy - oy)], fill=(220, 30, 30, 255), width=w)
    im.save(OUT / "prohibited.png")


def blackout():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([S*0.08, S*0.3, S*0.92, S*0.7], radius=int(S*0.06), fill=(10, 10, 10, 255))
    im.save(OUT / "blackout.png")


if __name__ == "__main__":
    heart(); star(); censored_tape(); prohibited(); blackout()
    print(f"Wrote built-in stickers to {OUT}")
