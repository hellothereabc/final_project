"""
Counting-MAE across confidence thresholds, because at one threshold it is not a
model comparison.

Everything in sections 3 and 4 counts at conf 0.25, the value evaluate.py fixed.
That is fine for one model on its own domain and misleading the moment two
models or two domains are compared: a detector that fires more freely
over-counts at a low threshold and under-counts at a high one, so a fixed
threshold reports how a model is calibrated as much as how well it finds fruit.

mAP does not have this problem - it integrates over confidence - which is
exactly why the two metrics disagree about which model is better.

Usage:
    python conf_sweep.py --models cross_laboro3 aug_no_colour --data tomatod
    python conf_sweep.py --models cross_mix --data laboro3 --confs 0.25 0.5
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "occlusion"))

from detmetrics import Score  # noqa: E402
from occlude import read_labels  # noqa: E402
from run_occlusion import dataset_files  # noqa: E402
from cross_dataset.taxonomy import STAGES, stage_of  # noqa: E402


def sweep(model, images, lbl_dir, class_names, conf, device):
    score = Score()
    for path in images:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        result = model.predict(str(path), conf=conf, device=device, verbose=False)[0]
        score.add([stage_of(class_names[cls]) for cls, _ in labels],
                  [box for _, box in labels],
                  [stage_of(model.names[int(c)]) for c in result.boxes.cls.cpu().numpy()],
                  result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
    row = score.summary()
    row["pred_over_gt"] = (sum(row["pred_totals"].values())
                           / max(1, sum(row["gt_totals"].values())))
    row["conf"] = conf
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cross_laboro3"],
                    help="run directory names under runs/")
    ap.add_argument("--data", default="laboro3", choices=["laboro3", "tomatod", "camacho"])
    ap.add_argument("--confs", type=float, nargs="+",
                    default=[0.25, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    images, lbl_dir, class_names = dataset_files(args.data)
    print(f"data {args.data} ({len(images)} images)\n")
    print(f"{'model':<20}{'conf':>6}{'recall':>9}{'stage-acc':>11}{'MAE':>7}{'pred/GT':>10}")

    rows = []
    for name in args.models:
        weights = ROOT / "runs" / name / "weights" / "best.pt"
        assert weights.exists(), f"missing {weights}"
        model = YOLO(str(weights))
        for conf in args.confs:
            row = sweep(model, images, lbl_dir, class_names, conf, args.device)
            row["model"] = name
            rows.append(row)
            print(f"{name:<20}{conf:>6}{row['recall']:>9.3f}"
                  f"{row['stage_accuracy']:>11.3f}{row['counting_mae']:>7.2f}"
                  f"{row['pred_over_gt']:>9.2f}x")

    print("\nbest counting-MAE per model, and where it sits:")
    for name in args.models:
        best = min((r for r in rows if r["model"] == name),
                   key=lambda r: r["counting_mae"])
        print(f"  {name:<20} MAE {best['counting_mae']:.2f} at conf {best['conf']} "
              f"(recall {best['recall']:.3f}, {best['pred_over_gt']:.2f}x ground truth)")

    out_dir = ROOT / "runs" / "conf_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"{args.data}.json"
    out.write_text(json.dumps({"data": args.data, "confs": args.confs,
                               "stages": STAGES, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
