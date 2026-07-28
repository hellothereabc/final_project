"""
Evaluate trained Laboro Tomato YOLO model.

Reports:
  1. Detection metrics via ultralytics val (mAP50, mAP50-95, P, R) — 6 classes.
  2. Counting-MAE per ripeness stage (green / half / fully), collapsing size.
     For each val image: |predicted_count - gt_count| per stage, averaged.

Usage: python evaluate.py --weights runs/laboro_yolo11n/weights/best.pt
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "laboro_yolo"
YAML = DATA / "data.yaml"

# class idx -> ripeness stage (size collapsed). Order from prepare_data.py:
# 0 b_fully 1 b_half 2 b_green 3 l_fully 4 l_half 5 l_green
STAGE = {0: "fully", 1: "half", 2: "green", 3: "fully", 4: "half", 5: "green"}
STAGES = ["green", "half", "fully"]


def gt_counts(label_path):
    c = defaultdict(int)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                c[STAGE[int(line.split()[0])]] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/laboro_yolo11n/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO(args.weights)

    # --- 1. detection metrics ---
    print("=" * 60)
    print("DETECTION METRICS (ultralytics val, 6 classes)")
    print("=" * 60)
    metrics = model.val(data=str(YAML), device=args.device, verbose=False)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"mAP75   : {metrics.box.map75:.4f}")
    print(f"mean P  : {metrics.box.mp:.4f}")
    print(f"mean R  : {metrics.box.mr:.4f}")

    # --- 2. counting-MAE per ripeness stage ---
    print("=" * 60)
    print(f"COUNTING-MAE per ripeness stage (conf={args.conf})")
    print("=" * 60)
    img_dir = DATA / "images" / "val"
    lbl_dir = DATA / "labels" / "val"
    imgs = sorted(img_dir.glob("*.jpg"))

    abs_err = defaultdict(list)   # stage -> [|pred-gt| per image]
    tot_pred = defaultdict(int)
    tot_gt = defaultdict(int)

    for img in imgs:
        gt = gt_counts(lbl_dir / (img.stem + ".txt"))
        res = model.predict(str(img), conf=args.conf, device=args.device, verbose=False)[0]
        pred = defaultdict(int)
        for cls in res.boxes.cls.tolist():
            pred[STAGE[int(cls)]] += 1
        for s in STAGES:
            abs_err[s].append(abs(pred[s] - gt[s]))
            tot_pred[s] += pred[s]
            tot_gt[s] += gt[s]

    all_err = []
    print(f"{'stage':>7} | {'MAE':>6} | {'GT tot':>7} | {'pred tot':>8}")
    print("-" * 40)
    for s in STAGES:
        mae = float(np.mean(abs_err[s]))
        all_err += abs_err[s]
        print(f"{s:>7} | {mae:6.3f} | {tot_gt[s]:7d} | {tot_pred[s]:8d}")
    print("-" * 40)
    print(f"{'overall':>7} | {float(np.mean(all_err)):6.3f}  (mean over image×stage)")

    # ripe-only counting error (the metric that matters for the study)
    ripe_mae = float(np.mean(abs_err["fully"]))
    print(f"\nRIPE (fully_ripened) counting-MAE per image: {ripe_mae:.3f}")


if __name__ == "__main__":
    main()
