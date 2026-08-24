"""
Score the shipped model on hand-counted phone photos.

Counting-MAE only, computed exactly as in evaluate.py and cross_dataset, so the
number sits next to the greenhouse ones: |predicted - counted| per stage, meaned
over images. Also reports signed bias per stage, because on a new domain the
direction is the finding - cross_dataset's transfer cells failed in opposite
directions and the MAE alone hid it.

Optionally breaks the result down by any column in labels.csv (light, distance,
occlusion), which is the point of filling those in.

Usage:
    python field_test/eval_field.py
    python field_test/eval_field.py --model laboro3 --by light
    python field_test/eval_field.py --annotate      # writes runs/field/annotated
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cross_dataset.taxonomy import SHORT, STAGES, stage_of  # noqa: E402

MODELS = {name: ROOT / "runs" / f"cross_{name}" / "weights" / "best.pt"
          for name in ("laboro3", "tomatod", "mix")}
MODELS["app"] = ROOT / "runs" / "app_yolo11n_det" / "weights" / "best.pt"

IMAGES = Path(__file__).parent / "data" / "images"
LABELS = Path(__file__).parent / "labels.csv"


def load_rows():
    assert LABELS.exists(), f"missing {LABELS} - run prepare_field.py first"
    rows = []
    for row in csv.DictReader(LABELS.open()):
        counts = {}
        for stage in STAGES:
            value = (row.get(stage) or "").strip()
            if value == "":
                counts = None
                break
            counts[stage] = int(value)
        if counts is None:
            continue          # not counted yet
        rows.append((row, counts))
    assert rows, "no counted rows yet - fill the count columns in labels.csv"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mix", help="shorthand or path to .pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--by", default="", help="labels.csv column to break down by")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "field" / "field.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(MODELS.get(args.model, args.model))
    assert weights.exists(), f"missing {weights}"
    model = YOLO(str(weights))
    rows = load_rows()
    print(f"model {args.model}  images {len(rows)}  conf {args.conf}")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_err, signed = defaultdict(list), defaultdict(list)
    groups = defaultdict(lambda: defaultdict(list))
    per_image = []

    for row, counts in rows:
        path = IMAGES / row["image"]
        result = model.predict(str(path), conf=args.conf, device=args.device,
                               verbose=False)[0]
        predicted = defaultdict(int)
        for cls in result.boxes.cls.tolist():
            predicted[stage_of(model.names[int(cls)])] += 1

        for stage in STAGES:
            error = predicted[stage] - counts[stage]
            abs_err[stage].append(abs(error))
            signed[stage].append(error)
            if args.by:
                groups[row.get(args.by, "") or "-"][stage].append(abs(error))

        per_image.append({"image": row["image"],
                          "gt": counts,
                          "pred": {s: predicted[s] for s in STAGES},
                          **{k: row.get(k, "") for k in
                             ("scene", "light", "distance", "occlusion", "notes")}})
        if args.annotate:
            annotated = out_dir / "annotated"
            annotated.mkdir(exist_ok=True)
            result.save(filename=str(annotated / row["image"]))

    print(f"\n{'stage':<15}{'counted':>9}{'predicted':>11}{'MAE':>8}{'bias':>8}")
    for stage in STAGES:
        counted = sum(c[stage] for _, c in rows)
        predicted = sum(item["pred"][stage] for item in per_image)
        print(f"{SHORT[stage]:<15}{counted:>9}{predicted:>11}"
              f"{np.mean(abs_err[stage]):>8.2f}{np.mean(signed[stage]):>+8.2f}")
    overall = float(np.mean([e for stage in STAGES for e in abs_err[stage]]))
    print(f"{'overall':<15}{'':>9}{'':>11}{overall:>8.2f}")

    if args.by:
        print(f"\nby {args.by}")
        for key, per_stage in sorted(groups.items()):
            values = [e for stage in STAGES for e in per_stage[stage]]
            print(f"  {key:<20} n={len(values) // len(STAGES):<4} "
                  f"MAE {np.mean(values):.2f}")

    Path(args.out).write_text(json.dumps({
        "model": args.model, "weights": str(weights), "conf": args.conf,
        "n_images": len(rows), "counting_mae": overall,
        "counting_mae_by_stage": {s: float(np.mean(abs_err[s])) for s in STAGES},
        "bias_by_stage": {s: float(np.mean(signed[s])) for s in STAGES},
        "images": per_image,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
