"""
The occlusion already in the data: fruit hidden behind other fruit.

The synthetic sweep controls the occluded fraction but pays for it with an
artificial occluder pasted over a fruit that the annotator saw whole. This is
the opposite trade: no control over the fraction, but every case is real. For
each annotated fruit, the fraction of its box covered by other annotated boxes
is a lower bound on how much of it is hidden - a lower bound because leaves and
stems occlude too and carry no label, and because box overlap does not know
which fruit is in front.

Read together the two answer different halves of the same question: whether the
recall-vs-hidden-fraction curve measured with pasted patches also describes fruit
occluded by the scene itself.

Usage:
    python occlusion/natural_occlusion.py
    python occlusion/natural_occlusion.py --model mix --data tomatod
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

from occlude import read_labels  # noqa: E402
from detmetrics import match  # noqa: E402
from run_occlusion import MODELS, dataset_files  # noqa: E402
from cross_dataset.taxonomy import SHORT, STAGES, stage_of  # noqa: E402

# Bin 0 is "nothing else overlaps this fruit at all"; the rest are upper edges.
EDGES = [0.05, 0.15, 0.30, 0.50]
LABELS = ["none", "0-5%", "5-15%", "15-30%", "30-50%", ">50%"]
GRID = 64          # rasterisation resolution per box, ~0.02% area quantisation


def covered_fraction(box, others):
    """Share of `box` covered by the union of `others`, by rasterising the box."""
    x1, y1, x2, y2 = box
    width, height = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    mask = np.zeros((GRID, GRID), dtype=bool)
    for other in others:
        ox1 = int(np.clip((other[0] - x1) / width * GRID, 0, GRID))
        ox2 = int(np.clip((other[2] - x1) / width * GRID, 0, GRID))
        oy1 = int(np.clip((other[1] - y1) / height * GRID, 0, GRID))
        oy2 = int(np.clip((other[3] - y1) / height * GRID, 0, GRID))
        if ox2 > ox1 and oy2 > oy1:
            mask[oy1:oy2, ox1:ox2] = True
    return float(mask.mean())


def bin_of(fraction):
    if fraction <= 0:
        return 0
    for index, edge in enumerate(EDGES, start=1):
        if fraction <= edge:
            return index
    return len(LABELS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laboro3")
    ap.add_argument("--data", default="laboro3")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(MODELS.get(args.model, args.model))
    model = YOLO(str(weights))
    images, lbl_dir, class_names = dataset_files(args.data)
    print(f"model {args.model}  data {args.data}  {len(images)} images")

    found = defaultdict(int)
    total = defaultdict(int)
    stage_hits = defaultdict(int)
    confidences = defaultdict(list)
    by_stage = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    overlaps = []

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        if not labels:
            continue
        boxes = [box for _, box in labels]
        gt_stages = [stage_of(class_names[cls]) for cls, _ in labels]

        result = model.predict(str(path), conf=args.conf, device=args.device,
                               verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_conf = result.boxes.conf.cpu().numpy()
        pred_stages = [stage_of(model.names[int(c)])
                       for c in result.boxes.cls.cpu().numpy()]
        pairs = match(boxes, pred_boxes, pred_conf, args.iou_thr)

        for index, box in enumerate(boxes):
            fraction = covered_fraction(box, [b for j, b in enumerate(boxes) if j != index])
            overlaps.append(fraction)
            bucket = bin_of(fraction)
            total[bucket] += 1
            by_stage[gt_stages[index]][bucket][1] += 1
            if index in pairs:
                found[bucket] += 1
                by_stage[gt_stages[index]][bucket][0] += 1
                confidences[bucket].append(float(pred_conf[pairs[index]]))
                if pred_stages[pairs[index]] == gt_stages[index]:
                    stage_hits[bucket] += 1

    print(f"\nmedian overlap {np.median(overlaps):.3f}, "
          f"{100 * np.mean(np.asarray(overlaps) > 0.05):.0f}% of fruit overlap "
          f"another by more than 5%")
    print(f"\n{'covered by other fruit':<24}{'n':>7}{'recall':>9}{'stage-acc':>11}{'conf':>8}")
    rows = []
    for bucket, label in enumerate(LABELS):
        if not total[bucket]:
            continue
        recall = found[bucket] / total[bucket]
        accuracy = stage_hits[bucket] / found[bucket] if found[bucket] else float("nan")
        mean_conf = float(np.mean(confidences[bucket])) if confidences[bucket] else float("nan")
        print(f"{label:<24}{total[bucket]:>7}{recall:>9.3f}{accuracy:>11.3f}{mean_conf:>8.3f}")
        rows.append({"bin": label, "n": total[bucket], "recall": recall,
                     "stage_accuracy": accuracy, "mean_conf": mean_conf})

    print(f"\n{'recall by stage':<16}" + "".join(f"{label:>10}" for label in LABELS))
    for stage in STAGES:
        cells = ""
        for bucket in range(len(LABELS)):
            hit, seen = by_stage[stage][bucket]
            cells += f"{(hit / seen if seen else float('nan')):>10.2f}"
        print(f"{SHORT[stage]:<16}{cells}")

    out_dir = ROOT / "runs" / "occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"natural_{args.model}_{args.data}.json"
    out.write_text(json.dumps({"model": args.model, "data": args.data,
                               "conf": args.conf, "iou_threshold": args.iou_thr,
                               "bins": LABELS, "edges": EDGES, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
