"""
Every trained model against every dataset's val set - the 3x2 matrix.

Diagonal = in-domain, off-diagonal = transfer, bottom row = effect of mixing.

Two metrics, because they can disagree and that's the interesting part: mAP from
ultralytics, and counting-MAE per stage. Counting logic follows ../evaluate.py,
with the hardcoded 6-class Laboro mapping replaced by taxonomy.py.

Usage:
    python cross_dataset/eval_cross.py                 # full matrix
    python cross_dataset/eval_cross.py --models laboro3 --data tomatod
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import SHORT, STAGES  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "cross"
RUNS = ROOT / "runs"

MODELS = {name: RUNS / f"cross_{name}" / "weights" / "best.pt"
          for name in ("laboro3", "tomatod", "mix")}
VAL_SETS = {"laboro3": DATA / "laboro3", "tomatod": DATA / "tomatod"}


def counting_mae(model, data_dir, conf, device):
    """Per-stage |predicted - GT| counts, averaged over images. Per image and not
    per dataset, since over- and undercounting would cancel in a total."""
    img_dir, lbl_dir = data_dir / "images" / "val", data_dir / "labels" / "val"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    abs_err = defaultdict(list)
    totals = {"gt": defaultdict(int), "pred": defaultdict(int)}

    for img in images:
        gt = defaultdict(int)
        label = lbl_dir / (img.stem + ".txt")
        if label.exists():
            for line in label.read_text().splitlines():
                if line.strip():
                    gt[STAGES[int(line.split()[0])]] += 1

        result = model.predict(str(img), conf=conf, device=device, verbose=False)[0]
        pred = defaultdict(int)
        for cls in result.boxes.cls.tolist():
            pred[STAGES[int(cls)]] += 1

        for stage in STAGES:
            abs_err[stage].append(abs(pred[stage] - gt[stage]))
            totals["gt"][stage] += gt[stage]
            totals["pred"][stage] += pred[stage]

    per_stage = {stage: float(np.mean(abs_err[stage])) for stage in STAGES}
    overall = float(np.mean([e for stage in STAGES for e in abs_err[stage]]))
    return {
        "per_stage": per_stage,
        "overall": overall,
        "gt_totals": dict(totals["gt"]),
        "pred_totals": dict(totals["pred"]),
        "n_images": len(images),
    }


def evaluate(model_name, data_name, conf, device):
    from ultralytics import YOLO

    weights = MODELS[model_name]
    assert weights.exists(), f"missing {weights} - train {model_name} first"
    data_dir = VAL_SETS[data_name]

    print(f"\n--- model={model_name}  val={data_name} "
          f"({'in-domain' if model_name == data_name else 'cross'}) ---")

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_dir / "data.yaml"), device=device,
                        verbose=False, plots=False)
    counting = counting_mae(model, data_dir, conf, device)

    row = {
        "model": model_name,
        "val": data_name,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "counting": counting,
    }
    stages = "  ".join(f"{SHORT[s]}={counting['per_stage'][s]:.2f}" for s in STAGES)
    print(f"mAP50-95 {row['map50_95']:.4f}  mAP50 {row['map50']:.4f}  "
          f"P {row['precision']:.4f}  R {row['recall']:.4f}")
    print(f"counting-MAE overall {counting['overall']:.3f}   {stages}")
    return row


def render(rows):
    print(f"\n{'=' * 78}\nMATRIX - mAP50-95 / counting-MAE (overall)\n{'=' * 78}")
    cols = list(VAL_SETS)
    print(f"{'model \\ val':<14}" + "".join(f"{c:>22}" for c in cols))
    for model_name in MODELS:
        cells = []
        for data_name in cols:
            row = next((r for r in rows
                        if r["model"] == model_name and r["val"] == data_name), None)
            cells.append("-" if row is None
                         else f"{row['map50_95']:.3f} / {row['counting']['overall']:.2f}")
        print(f"{model_name:<14}" + "".join(f"{c:>22}" for c in cells))
    print("\ndiagonal = in-domain, off-diagonal = domain shift, "
          "bottom row = effect of mixing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--data", nargs="+", choices=list(VAL_SETS), default=list(VAL_SETS))
    ap.add_argument("--conf", type=float, default=0.25, help="matches ../evaluate.py")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(ROOT / "runs" / "cross_matrix.json"))
    args = ap.parse_args()

    rows = [evaluate(m, d, args.conf, args.device) for m in args.models for d in args.data]
    render(rows)

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
