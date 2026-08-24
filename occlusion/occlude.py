"""
Synthetic occlusion: hide a controlled fraction of every annotated fruit.

Occlusion is the one field condition a lab dataset cannot answer by itself -
Laboro and tomatOD are both shot with the fruit deliberately in view. The only
way to get a curve rather than an anecdote is to remove a known fraction of a
known fruit and re-measure.

Three things matter for the occluder to be an occluder and not an artefact:

  leaf     the strip is replaced with a patch cropped from the same image's
           background (leaves, netting, soil), so lighting and white balance
           carry over. A solid rectangle of any colour is a novel object; a
           leaf-coloured crop is what actually hides a tomato in a greenhouse.
  black    kept as a control. If the two occluders give the same curve, the
           model is losing the fruit, not reacting to the patch.
  side     which half is hidden is not cosmetic for this task. A half-ripened
           fruit is red on one side and green on the other, so hiding the red
           half and hiding the green half are two different experiments on the
           same fruit, and the ripeness label should move in opposite
           directions. `hide_red` / `hide_green` pick the half by median a* in
           CIELAB - the same axis cross_dataset/colour_stages.py measures
           stages on.

Geometry: the strip runs from one side of the box and covers `fraction` of the
box's area, so fraction is the visible-area loss, directly comparable across
fruit of different sizes.
"""
import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - install hint is the whole point
    raise SystemExit("needs opencv-python (it comes with ultralytics)") from exc

SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def read_labels(path, width, height):
    """YOLO label file -> [(class_index, [x1, y1, x2, y2] in pixels)].

    Handles both box lines (5 fields) and polygon lines (segmentation datasets
    such as the Camacho export), reducing polygons to their bounding box the
    same way ultralytics does when a segment model is evaluated as a detector.
    """
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            xc, yc, bw, bh = values
            x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
        else:
            xs, ys = values[0::2], values[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((cls, np.array([x1 * width, y1 * height,
                                     x2 * width, y2 * height], dtype=np.float32)))
    return boxes


def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def background(image, boxes, rng, tries=80):
    """A crop of the image that contains no annotated fruit, plus the median
    colour of everything outside the boxes as a fallback for dense frames."""
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=bool)
    for _, box in boxes:
        x1, y1, x2, y2 = np.clip(box, 0, [width, height, width, height]).astype(int)
        mask[y1:y2, x1:x2] = False
    colour = (np.median(image[mask], axis=0) if mask.any()
              else np.median(image.reshape(-1, 3), axis=0))

    patch_h, patch_w = max(8, height // 8), max(8, width // 8)
    for _ in range(tries):
        y = int(rng.integers(0, max(1, height - patch_h)))
        x = int(rng.integers(0, max(1, width - patch_w)))
        rect = (x, y, x + patch_w, y + patch_h)
        if not any(_rects_overlap(rect, box) for _, box in boxes):
            return image[y:y + patch_h, x:x + patch_w].copy(), colour
    return None, colour


def _median_a(lab, x1, y1, x2, y2):
    """Median CIELAB a* (green -> red) of a region; OpenCV stores a* offset by 128."""
    region = lab[y1:y2, x1:x2, 1]
    return float(np.median(region)) - 128.0 if region.size else 0.0


def choose_side(lab, box, mode, rng):
    """Which side of the box the occluder comes in from.

    random    a side drawn per fruit, so the sweep averages over geometry
    hide_red  the redder half disappears - "the ripe part is behind a leaf"
    hide_green the greener half disappears - the mirror experiment
    """
    if mode == "random":
        return SIDES[int(rng.integers(0, len(SIDES)))]

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    ymid, xmid = (y1 + y2) // 2, (x1 + x2) // 2
    vertical = _median_a(lab, x1, y1, x2, ymid) - _median_a(lab, x1, ymid, x2, y2)
    horizontal = _median_a(lab, x1, y1, xmid, y2) - _median_a(lab, xmid, y1, x2, y2)

    # Split along whichever axis separates colour best; a fruit lit from the
    # side splits horizontally, one ripening from the blossom end vertically.
    if abs(vertical) >= abs(horizontal):
        redder = "top" if vertical > 0 else "bottom"
    else:
        redder = "left" if horizontal > 0 else "right"
    return redder if mode == "hide_red" else OPPOSITE[redder]


def strip(box, side, fraction, width, height):
    """Pixel rect covering `fraction` of the box, entering from `side`."""
    x1, y1, x2, y2 = box
    if side in ("top", "bottom"):
        cut = (y2 - y1) * fraction
        y1, y2 = (y1, y1 + cut) if side == "top" else (y2 - cut, y2)
    else:
        cut = (x2 - x1) * fraction
        x1, x2 = (x1, x1 + cut) if side == "left" else (x2 - cut, x2)

    x1 = int(np.clip(round(x1), 0, width))
    x2 = int(np.clip(round(x2), 0, width))
    y1 = int(np.clip(round(y1), 0, height))
    y2 = int(np.clip(round(y2), 0, height))
    return x1, y1, x2, y2


def occlude(image, boxes, fraction, mode="random", occluder="leaf", seed=0):
    """Return a copy of `image` with `fraction` of every box hidden."""
    if fraction <= 0 or not boxes:
        return image.copy()

    rng = np.random.default_rng(seed)
    out = image.copy()
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if mode != "random" else None
    patch, colour = background(image, boxes, rng) if occluder == "leaf" else (None, None)

    for _, box in boxes:
        side = choose_side(lab, box, mode, rng)
        x1, y1, x2, y2 = strip(box, side, fraction, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        if occluder == "black":
            fill = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        elif patch is not None:
            fill = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        else:
            # No clean background in the frame: flat median colour plus grain,
            # so the strip is not a suspiciously smooth rectangle.
            noise = rng.normal(0, 6, size=(y2 - y1, x2 - x1, 3))
            fill = np.clip(colour + noise, 0, 255).astype(np.uint8)
        out[y1:y2, x1:x2] = fill

    return out
