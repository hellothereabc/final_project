"""
Counting-MAE across confidence thresholds for every trained model size, on the
clean six-class Laboro val split.

The n/m pair was swept this way for model_size_results.md
(runs/laboro_model_size_thresholds.json); this repeats it for n s m l x in one
pass so the table has a tuned counting column per size. Same reason as
conf_sweep.py: at one fixed threshold a counting comparison is a calibration
comparison, and a bigger model fires more freely.

Each model is predicted once at conf 0.05 and thresholded in Python afterwards.

Usage:
    python runtime_bench/size_thresholds.py
    python runtime_bench/size_thresholds.py --models n x --confs 0.25 0.5
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lighting"))
from cross_dataset.taxonomy import STAGES, stage_of  # noqa: E402
from run_lighting import MODELS, dataset_files  # noqa: E402
from occlusion.occlude import read_labels  # noqa: E402

CONF_FLOOR = 0.05


def sweep(model, images, lbl_dir, class_names, confs, device):
    import cv2

    abs_err = {c: defaultdict(list) for c in confs}
    pred_tot = {c: 0 for c in confs}
    gt_tot = 0
    for path in images:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        gt = defaultdict(int)
        for cls, _ in labels:
            gt[stage_of(class_names[cls])] += 1
        gt_tot += len(labels)
        result = model.predict(str(path), conf=CONF_FLOOR, device=device, verbose=False)[0]
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        stages = np.array([stage_of(model.names[int(c)]) for c in cls], dtype=object)
        for c in confs:
            kept = stages[conf >= c]
            pred = defaultdict(int)
            for s in kept:
                pred[s] += 1
            pred_tot[c] += len(kept)
            for s in STAGES:
                abs_err[c][s].append(abs(pred[s] - gt[s]))
    rows = []
    for c in confs:
        by_stage = {s: float(np.mean(abs_err[c][s])) for s in STAGES}
        rows.append({
            "conf": c,
            "counting_mae": float(np.mean([e for s in STAGES for e in abs_err[c][s]])),
            "ripe_mae": by_stage["fully_ripened"],
            "stage_mae": by_stage,
            "pred_over_gt": pred_tot[c] / max(1, gt_tot),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--confs", type=float, nargs="+",
                    default=[round(0.05 * i, 2) for i in range(1, 19)])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(ROOT / "runs" / "laboro_model_size_thresholds_nsmlx.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    images, lbl_dir, class_names = dataset_files()
    out_rows = []
    print(f"{'model':>8} {'best MAE':>9} {'@conf':>6} {'ripe MAE':>9} {'@conf':>6} {'MAE@.25':>8}")
    for name in args.models:
        weights = MODELS[name]
        if not weights.exists():
            print(f"{name:>8}  missing {weights}")
            continue
        rows = sweep(YOLO(str(weights)), images, lbl_dir, class_names, args.confs, args.device)
        for row in rows:
            row["model"] = f"YOLO11{name}"
        out_rows += rows
        best = min(rows, key=lambda r: r["counting_mae"])
        ripe = min(rows, key=lambda r: r["ripe_mae"])
        at25 = next((r for r in rows if abs(r["conf"] - 0.25) < 1e-6), best)
        print(f"{name:>8} {best['counting_mae']:>9.3f} {best['conf']:>6.2f}"
              f" {ripe['ripe_mae']:>9.3f} {ripe['conf']:>6.2f} {at25['counting_mae']:>8.3f}")

    Path(args.out).write_text(json.dumps({"confs": args.confs, "rows": out_rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
