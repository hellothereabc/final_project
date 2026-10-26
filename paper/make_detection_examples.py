"""Qualitative detection examples: annotation, YOLO11n and YOLO11x on the same
Laboro validation frames, boxes coloured by maturity stage, with per-stage
counts printed under each panel.

Needs ultralytics, so run it with the reproduce venv:

    /Users/chel/Projects/master_diploma/reproduce/.venv/bin/python \
        paper/make_detection_examples.py

Writes figures/detection_examples.jpg. The frames are chosen automatically:
the validation images with the most half-ripened fruit, since that is the stage
the paper's arguments turn on. Confidence 0.25, as in the other fixed-threshold
results.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
DATA = ROOT / "data" / "laboro_yolo"
NAMES = ["b_fully_ripened", "b_half_ripened", "b_green",
         "l_fully_ripened", "l_half_ripened", "l_green"]
STAGE = {n: n.split("_", 1)[1] for n in NAMES}
COLOUR = {"green": (49, 163, 84), "half_ripened": (253, 141, 60), "fully_ripened": (222, 45, 38)}
SHORT = {"green": "G", "half_ripened": "H", "fully_ripened": "F"}
MODELS = {
    "YOLO11n": ROOT / "runs" / "laboro_yolo11n_export" / "weights" / "best.pt",
    "YOLO11x": ROOT / "runs" / "laboro_yolo11x_exiffix_b4_mps" / "weights" / "best.pt",
}
PANEL_W = 520
N_FRAMES = 2


def ground_truth(path):
    label = DATA / "labels" / "val" / (path.stem + ".txt")
    boxes = []
    for line in label.read_text().splitlines():
        if line.strip():
            cls, cx, cy, w, h = line.split()[:5]
            boxes.append((STAGE[NAMES[int(cls)]], float(cx), float(cy), float(w), float(h)))
    return boxes


def panel(image, boxes, title):
    """boxes: (stage, cx, cy, w, h) normalised."""
    scale = PANEL_W / image.width
    im = image.resize((PANEL_W, int(image.height * scale)))
    draw = ImageDraw.Draw(im)
    for stage, cx, cy, w, h in boxes:
        x0, y0 = (cx - w / 2) * im.width, (cy - h / 2) * im.height
        x1, y1 = (cx + w / 2) * im.width, (cy + h / 2) * im.height
        draw.rectangle([x0, y0, x1, y1], outline=COLOUR[stage], width=4)
    counts = {s: sum(1 for b in boxes if b[0] == s) for s in COLOUR}
    caption = f"{title}   " + "  ".join(f"{SHORT[s]} {counts[s]}" for s in COLOUR)
    band = Image.new("RGB", (im.width, 44), "white")
    font = _font(26)
    ImageDraw.Draw(band).text((8, 7), caption, fill="black", font=font)
    out = Image.new("RGB", (im.width, im.height + band.height), "white")
    out.paste(im, (0, 0))
    out.paste(band, (0, im.height))
    return out


def _font(size):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    images = sorted((DATA / "images" / "val").iterdir())
    ranked = sorted(images, key=lambda p: -sum(1 for b in ground_truth(p) if b[0] == "half_ripened"))
    frames = ranked[:N_FRAMES]
    models = {name: YOLO(str(path)) for name, path in MODELS.items()}

    rows = []
    for path in frames:
        image = Image.open(path).convert("RGB")
        panels = [panel(image, ground_truth(path), "annotation")]
        for name, model in models.items():
            result = model.predict(str(path), conf=0.25, imgsz=640, verbose=False)[0]
            boxes = [(STAGE[NAMES[int(c)]], *map(float, xywhn))
                     for c, xywhn in zip(result.boxes.cls, result.boxes.xywhn)]
            panels.append(panel(image, boxes, name))
        row = Image.new("RGB", (sum(p.width for p in panels) + 16 * (len(panels) - 1),
                                max(p.height for p in panels)), "white")
        x = 0
        for p in panels:
            row.paste(p, (x, 0))
            x += p.width + 16
        rows.append(row)
        print(path.name)

    sheet = Image.new("RGB", (max(r.width for r in rows),
                              sum(r.height for r in rows) + 16 * (len(rows) - 1)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 16
    sheet.save(OUT / "detection_examples.jpg", quality=88)
    print("wrote", OUT / "detection_examples.jpg")


if __name__ == "__main__":
    main()
