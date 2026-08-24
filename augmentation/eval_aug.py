"""
Score every augmentation ablation on the three things they could plausibly buy.

  in-domain    laboro3 val - the number the recipe was tuned for
  transfer     tomatOD val - augmentation is supposed to buy generalisation, and
               generalisation only shows up off the training distribution
  occlusion    laboro3 val with 30% and 50% of every fruit hidden, the sweep from
               occlusion/ at two points - the one robustness axis the deployed
               app actually meets

The third column is why the ablation includes `occlusion_aug` (cutmix + mixup).
Those paste foreign pixels over the fruit at training time, which is synthetic
occlusion by another name; if that transfers, this is where it shows, not on the
clean in-domain row.

Usage:
    python augmentation/eval_aug.py
    python augmentation/eval_aug.py --runs default no_colour --levels 0.5
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "occlusion"))

from run_occlusion import dataset_files, sweep_level  # noqa: E402
from cross_dataset.taxonomy import STAGES  # noqa: E402

RUNS = ROOT / "runs"
# `default` is cross_dataset's laboro3 run: same protocol, stock augmentation.
BASELINE = "default"


def available_runs():
    runs = {BASELINE: RUNS / "cross_laboro3"}
    runs.update({d.name[4:]: d for d in sorted(RUNS.glob("aug_*")) if d.is_dir()})
    return {name: path for name, path in runs.items()
            if (path / "weights" / "best.pt").exists()}


def evaluate(name, run_dir, levels, conf, device):
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    model = YOLO(str(weights))
    print(f"\n--- {name} ({weights.relative_to(ROOT)}) ---", flush=True)

    metrics = model.val(data=str(ROOT / "data" / "cross" / "laboro3" / "data.yaml"),
                        device=device, verbose=False, plots=False)
    row = {
        "run": name,
        "weights": str(weights),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall_val": float(metrics.box.mr),
    }

    for data_name in ("laboro3", "tomatod"):
        images, lbl_dir, class_names = dataset_files(data_name)
        clean = sweep_level(model, images, lbl_dir, class_names, 0.0, "none",
                            "leaf", conf, device, seed=0)
        row[data_name] = clean
        print(f"  {data_name:<8} recall {clean['recall']:.3f}  "
              f"stage-acc {clean['stage_accuracy']:.3f}  "
              f"MAE {clean['counting_mae']:.2f}", flush=True)

    images, lbl_dir, class_names = dataset_files("laboro3")
    row["occlusion"] = []
    for fraction in levels:
        hit = sweep_level(model, images, lbl_dir, class_names, fraction, "random",
                          "leaf", conf, device, seed=0)
        row["occlusion"].append(hit)
        print(f"  occl {fraction * 100:>3.0f}% recall {hit['recall']:.3f}  "
              f"stage-acc {hit['stage_accuracy']:.3f}  "
              f"MAE {hit['counting_mae']:.2f}", flush=True)
    return row


def render(rows, levels):
    print(f"\n{'=' * 104}\naugmentation ablation - laboro3 protocol, 60 epochs, seed 0"
          f"\n{'=' * 104}")
    header = (f"{'run':<15}{'mAP50-95':>9}{'in-dom R':>10}{'MAE':>7}"
              f"{'tomatOD R':>11}{'MAE':>7}")
    for fraction in levels:
        header += f"{f'occl{fraction * 100:.0f} R':>10}{'MAE':>7}"
    print(header)

    base = next((r for r in rows if r["run"] == BASELINE), None)
    for row in rows:
        line = (f"{row['run']:<15}{row['map50_95']:>9.3f}"
                f"{row['laboro3']['recall']:>10.3f}{row['laboro3']['counting_mae']:>7.2f}"
                f"{row['tomatod']['recall']:>11.3f}{row['tomatod']['counting_mae']:>7.2f}")
        for hit in row["occlusion"]:
            line += f"{hit['recall']:>10.3f}{hit['counting_mae']:>7.2f}"
        print(line)
    if base:
        print(f"\n{BASELINE} = runs/cross_laboro3 (stock ultralytics 8.4 augmentation); "
              "every other row changes one group and nothing else")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(ROOT / "runs" / "aug_matrix.json"))
    args = ap.parse_args()

    runs = available_runs()
    if args.runs:
        missing = [name for name in args.runs if name not in runs]
        assert not missing, f"no trained weights for {missing} - train_aug.py first"
        runs = {name: runs[name] for name in args.runs}
    print(f"scoring {len(runs)} runs: {', '.join(runs)}")

    rows = [evaluate(name, path, args.levels, args.conf, args.device)
            for name, path in runs.items()]
    render(rows, args.levels)

    Path(args.out).write_text(json.dumps(
        {"protocol": "laboro3, YOLO11n detect, 60 epochs, imgsz 640, batch 16, seed 0",
         "levels": args.levels, "conf": args.conf, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
