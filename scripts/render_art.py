"""Render the launcher icon (assets/app.png), the README cover (docs/art/cover.png)
and the 4:3 MaixHub listing cover (docs/art/cover_4x3.png).

Run: uv run --with pillow==12.3.0 python3 scripts/render_art.py [--check]
Deterministic: --check renders everything twice and fails if the PNG bytes differ.
Needs network once, to fetch Noto Sans Bold at a pinned commit into
~/.cache/maixcam-yolo26/fonts/ (sha256 verified; the font is not committed).
The cover reads its two headline figures from model/MANIFEST.md and errors if
they cannot be parsed. Style constants and place_label_tab mirror
main/include/style.hpp BY HAND: the cover is illustrative, so the two can drift,
and both must be updated together.
"""
import hashlib
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_URL = ("https://raw.githubusercontent.com/notofonts/notofonts.github.io/"
            "28b15b4b43b7bed62b5cf6e6b0b5ff5846270535/fonts/NotoSans/full/ttf/NotoSans-Bold.ttf")
FONT_SHA256 = "3a08a47daa00cade516425c15c57615aef2fd418ec9811a7b9f465088f92cc05"
FONT_CACHE = Path.home() / ".cache/maixcam-yolo26/fonts/NotoSans-Bold.ttf"

ACCENT = (255, 90, 78)
PANEL = (29, 33, 39)
BG = (21, 24, 29)
WHITE = (255, 255, 255)
MUTED = (170, 176, 184)
PALETTE = [(229, 57, 44), (22, 137, 127), (183, 121, 31), (123, 63, 196),
           (31, 127, 174), (94, 127, 37), (194, 51, 127), (196, 90, 18)]
COCO80 = [n.strip() for n in (
    "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|"
    "parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|"
    "handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|"
    "surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|"
    "broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|"
    "mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|"
    "teddy bear|hair drier|toothbrush").split("|")]
assert len(COCO80) == 80


def font_path():
    if FONT_CACHE.exists() and hashlib.sha256(FONT_CACHE.read_bytes()).hexdigest() == FONT_SHA256:
        return str(FONT_CACHE)
    data = urllib.request.urlopen(FONT_URL, timeout=60).read()
    got = hashlib.sha256(data).hexdigest()
    if got != FONT_SHA256:
        sys.exit(f"font sha256 mismatch: {got} != {FONT_SHA256}")
    FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FONT_CACHE.write_bytes(data)
    return str(FONT_CACHE)


def font(px):
    return ImageFont.truetype(font_path(), px)


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, "PNG", optimize=False, compress_level=9)
    return b.getvalue()


# ---- icon -------------------------------------------------------------------
def render_icon(size=128):
    k = 4
    s = size * k
    u = s / 128
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    w = round(7 * u)
    paths = [[(12, 36), (12, 12), (36, 12)], [(92, 12), (116, 12), (116, 36)],
             [(116, 92), (116, 116), (92, 116)], [(36, 116), (12, 116), (12, 92)]]
    for p in paths:
        pts = [(x * u, y * u) for x, y in p]
        d.line(pts, fill=WHITE, width=w)
        for x, y in pts:                      # round caps and the corner join
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=WHITE)
    box = [26 * u, 38 * u, 102 * u, 90 * u]
    d.rounded_rectangle(box, radius=8 * u, fill=ACCENT)
    f = ImageFont.truetype(font_path(), round(46 * u))
    l, t, r, b = d.textbbox((0, 0), "26", font=f)
    cx, cy = 64 * u, 64 * u
    d.text((cx - (l + r) / 2, cy - (t + b) / 2), "26", font=f, fill=WHITE)
    return im.resize((size, size), Image.LANCZOS)


# ---- manifest figures ---------------------------------------------------------
def manifest_figures():
    text = (ROOT / "model/MANIFEST.md").read_text()

    def row(prefix):
        for line in text.splitlines():
            if line.startswith("| " + prefix):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) == 3:
                    return cells
        raise SystemExit(f"MANIFEST.md: row '{prefix}' not found")

    def first_number(cell, what):
        m = re.match(r"(\d+(?:\.\d+)?)", cell)
        if not m:
            raise SystemExit(f"MANIFEST.md: cannot parse {what} from {cell!r}")
        return float(m.group(1))

    ms = [first_number(c, f"{n} board ms") for c, n in zip(row("Board inference ms")[1:], ("320", "640"))]
    maps = [first_number(c, f"{n} INT8 mAP50") for c, n in zip(row("INT8 mAP50")[1:], ("320", "640"))]
    if any(m <= 0 for m in ms) or not all(0 < m < 1 for m in maps):
        raise SystemExit(f"MANIFEST.md: implausible figures ms={ms} map={maps}")
    return [f"~{1000.0 / m:.1f} fps \u00b7 mAP50 {a:.2f}" for m, a in zip(ms, maps)]


# ---- cover --------------------------------------------------------------------
def rounded_mask(size, radius, k=4):
    m = Image.new("L", (size[0] * k, size[1] * k), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] * k - 1, size[1] * k - 1], radius=radius * k, fill=255)
    return m.resize(size, Image.LANCZOS)


def paste_round(base, im, xy, radius):
    base.paste(im, xy, rounded_mask(im.size, radius))


def draw_round(base, box, radius, fill, k=4):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    layer = Image.new("RGB", (w, h), fill)
    base.paste(layer, (x0, y0), rounded_mask((w, h), radius))


def place_label_tab(box, tw, th, placed, sw):
    """Mirror of y26::place_label_tab in main/include/style.hpp (keep in sync)."""
    x0, y0, _, _ = box
    x = max(0, min(x0, sw - tw))
    above = (x, y0 - th, tw, th)
    hit = above[1] < 0 or any(above[0] < px + pw and px < above[0] + tw and above[1] < py + ph and py < above[1] + th
                              for px, py, pw, ph in placed)
    return (x, y0, tw, th) if hit else above


def draw_dets(photo, dets, scale, font_px):
    """Photo scaled by `scale` with the real INT8 detections, drawn like the app."""
    ph = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
    pd = ImageDraw.Draw(ph)
    ft = font(font_px)
    pad = round(font_px * 0.4)
    placed = []
    for x, y, w, h, cls, score in dets["dets"]:
        c = PALETTE[cls % 8]
        x, y, w, h = round(x * scale), round(y * scale), round(w * scale), round(h * scale)
        pd.rectangle([x, y, x + w - 1, y + h - 1], outline=c, width=max(3, round(3 * scale)))
        label = f"{COCO80[cls]} {max(0, min(100, round(score * 100)))}%"
        l, t, r, b = pd.textbbox((0, 0), label, font=ft)
        tw, th = r - l + 2 * pad, font_px + pad
        x, ty, _, _ = place_label_tab((x, y, w, h), tw, th, placed, ph.width)
        placed.append((x, ty, tw, th))
        pd.rectangle([x, ty, x + tw - 1, ty + th - 1], fill=c)
        pd.text((x + pad - l, ty + (th - (b - t)) // 2 - t), label, font=ft, fill=WHITE)
    return ph


def load_photo():
    dets = json.loads((ROOT / "docs/art/cover_dets.json").read_text())
    photo = Image.open(ROOT / "docs/art" / dets["file"]).convert("RGB")
    assert photo.size == (dets["width"], dets["height"]), photo.size
    return photo, dets


def draw_figure_row(im, d, x, y, tag, txt, first, chip, f):
    draw_round(im, (x, y, x + 64, y + 44), 10, ACCENT if first else (54, 60, 70))
    l, t, r, b = d.textbbox((0, 0), tag, font=chip)
    d.text((x + 32 - (l + r) / 2, y + 22 - (t + b) / 2), tag, font=chip, fill=WHITE)
    l, t, r, b = d.textbbox((0, 0), txt, font=f)
    d.text((x + 84 - l, y + 22 - (t + b) / 2), txt, font=f, fill=WHITE)


def render_cover():
    """Wide README banner, 1200x630."""
    rows = manifest_figures()
    photo, dets = load_photo()

    W, H = 1200, 630
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    ph = draw_dets(photo, dets, 1.0, 15)
    paste_round(im, ph, (W - ph.width - 40, (H - ph.height) // 2), 16)

    # left column
    icon = render_icon(150)
    im.paste(icon, (56, 64), icon)
    d.text((56, 236), "YOLO26 Detector", font=font(50), fill=WHITE)
    sub = font(22)
    d.text((56, 312), "Real-time object detection", font=sub, fill=MUTED)
    d.text((56, 342), "80 COCO classes · MaixCAM", font=sub, fill=MUTED)

    # figures tile
    tx0, ty0, tx1, ty1 = 56, 410, 56 + 430, 410 + 170
    draw_round(im, (tx0, ty0, tx1, ty1), 14, PANEL)
    for i, (tag, txt) in enumerate([("320", rows[0]), ("640", rows[1])]):
        draw_figure_row(im, d, tx0 + 20, ty0 + 22 + i * 72, tag, txt, i == 0, font(20), font(30))
    return im


def render_cover_4x3():
    """MaixHub listing cover, 4:3 (1200x900, shown at 400x300)."""
    rows = manifest_figures()
    photo, dets = load_photo()

    W, H = 1200, 900
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    # header: icon + title + subtitle
    icon = render_icon(150)
    im.paste(icon, (48, 36), icon)
    d.text((224, 44), "YOLO26 Detector", font=font(72), fill=WHITE)
    d.text((228, 140), "Real-time detection · 80 classes · MaixCAM", font=font(30), fill=MUTED)

    # photo: 4:3, full width below the header, figures overlaid along the bottom
    ph = draw_dets(photo, dets, 1104 / photo.width, 26)     # 1104x828 -> crop to fit
    top = 216
    ph = ph.crop((0, 0, ph.width, H - top - 48))
    paste_round(im, ph, (48, top), 20)

    # figures strip across the bottom of the photo
    sx0, sy1 = 48, H - 48
    sy0 = sy1 - 96
    draw_round(im, (sx0 + 20, sy0 - 20, sx0 + 1104 - 20, sy1 - 20), 16, PANEL)
    for i, (tag, txt) in enumerate([("320", rows[0]), ("640", rows[1])]):
        draw_figure_row(im, d, sx0 + 44 + i * 540, sy0 + 6, tag, txt, i == 0, font(22), font(30))
    return im


def main():
    check = "--check" in sys.argv
    outs = {ROOT / "assets/app.png": render_icon, ROOT / "docs/art/cover.png": render_cover,
            ROOT / "docs/art/cover_4x3.png": render_cover_4x3}
    for path, fn in outs.items():
        data = png_bytes(fn())
        if check and png_bytes(fn()) != data:
            sys.exit(f"NOT deterministic: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"{path.relative_to(ROOT)}  {hashlib.sha256(data).hexdigest()}"
              + ("  (deterministic x2)" if check else ""))


if __name__ == "__main__":
    main()
