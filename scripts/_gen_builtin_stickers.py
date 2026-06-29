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
    col = (230, 30, 60, 255)
    cx = S / 2
    r = S * 0.205          # lobe radius
    top = S * 0.36         # lobe centre y
    off = r * 0.92         # lobe centre x distance from middle (slight overlap → no gap)
    lx, rx = cx - off, cx + off
    # two symmetric top lobes
    d.ellipse([lx - r, top - r, lx + r, top + r], fill=col)
    d.ellipse([rx - r, top - r, rx + r, top + r], fill=col)
    # bottom V: outer edge of each lobe down to a single point
    d.polygon([(lx - r, top), (rx + r, top), (cx, S * 0.86)], fill=col)
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
    cx = cy = S / 2
    r = (S - 2 * w) / 2
    ox, oy = r * math.cos(math.radians(45)), r * math.sin(math.radians(45))
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
