"""
Lighting sweep: what the detector loses when the light changes and the fruit
does not.

Same shape as occlusion/run_occlusion.py - the same weights over the same
validation images many times, one photometric knob turned per run (see
relight.py) - and the same three numbers, because they fail at different rates:

  detection   class-agnostic recall at IoU 0.5 - was the fruit found
  stage       among found fruit, is the ripeness stage still right
  counting    counting-MAE per image

Every level is predicted once at a low confidence floor and scored at each
threshold in --confs from that one pass. Section 6 of the README showed that a
counting comparison at one fixed threshold compares calibration rather than
models, and a change of exposure moves confidences wholesale, so the sweep is
not optional here: the table reports conf 0.25 next to the best threshold per
level.

Ground truth is the original annotation. Nothing here moves a pixel.

Usage:
    python lighting/run_lighting.py                          # n baseline
    python lighting/run_lighting.py --model m
    python lighting/run_lighting.py --model light_hsv_v08 --axis exposure
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "occlusion"))
from relight import EXPOSURE_STOPS, WB_SHIFTS, mean_luma, relight  # noqa: E402
from occlude import read_labels  # noqa: E402
from detmetrics import Score  # noqa: E402
from cross_dataset.taxonomy import STAGES, stage_of  # noqa: E402

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "laboro_yolo"

# Shorthands for the runs this section compares. Everything is the six-class
# Laboro protocol of train.py; the n/m rows are the section 5 model-size pair.
MODELS = {
    "n": RUNS / "laboro_yolo11n_export" / "weights" / "best.pt",
    "s": RUNS / "laboro_yolo11s_exiffix_b16_mps" / "weights" / "best.pt",
    "m": RUNS / "laboro_yolo11m_exiffix_b8_mps" / "weights" / "best.pt",
    "l": RUNS / "laboro_yolo11l_exiffix_b8_mps" / "weights" / "best.pt",
    "x": RUNS / "laboro_yolo11x_exiffix_b4_mps" / "weights" / "best.pt",
}

AXES = {
    "exposure": [("stops", s) for s in EXPOSURE_STOPS],
    "white_balance": [("wb", w) for w in WB_SHIFTS],
}

# Predict once this low, threshold in Python afterwards.
CONF_FLOOR = 0.05


def dataset_files():
    img_dir, lbl_dir = DATA / "images" / "val", DATA / "labels" / "val"
    assert img_dir.is_dir(), f"missing {img_dir}"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    line = next(l for l in (DATA / "data.yaml").read_text().splitlines()
                if l.startswith("names"))
    names = [n.strip(" '\"") for n in line.split("[", 1)[1].rsplit("]", 1)[0].split(",")]
    return images, lbl_dir, names


def sweep_level(model, images, lbl_dir, class_names, knob, value, confs, device):
    scores = {conf: Score() for conf in confs}
    lumas = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        gt_stages = [stage_of(class_names[cls]) for cls, _ in labels]
        gt_boxes = [box for _, box in labels]

        view = relight(image, **{knob: value})
        lumas.append(mean_luma(view))
        result = model.predict(view, conf=CONF_FLOOR, device=device, verbose=False)[0]
        cls = result.boxes.cls.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        stages = np.array([stage_of(model.names[int(c)]) for c in cls], dtype=object)
        for threshold, score in scores.items():
            keep = conf >= threshold
            score.add(gt_stages, gt_boxes, list(stages[keep]), xyxy[keep], conf[keep])

    by_conf = {conf: score.summary() for conf, score in scores.items()}
    best_conf = min(by_conf, key=lambda c: by_conf[c]["counting_mae"])
    # Paired tests need the per-image series, not the pooled summary. Keep it
    # for the two thresholds every table reports: the fixed one and the best.
    per_image = {str(c): scores[c].per_image() for c in {0.25, best_conf} if c in scores}
    row = dict(by_conf[0.25] if 0.25 in by_conf else by_conf[best_conf])
    row.update({
        "knob": knob, "value": value, "mean_luma": float(np.mean(lumas)),
        "best_conf": best_conf, "best_counting_mae": by_conf[best_conf]["counting_mae"],
        "best_recall": by_conf[best_conf]["recall"],
        "by_conf": {str(c): {k: v[k] for k in ("recall", "stage_accuracy", "counting_mae",
                                                "counting_mae_by_stage", "pred_totals")}
                    for c, v in by_conf.items()},
        "per_image": per_image,
    })
    return row


def render(rows, title):
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")
    print(f"{'level':>7} {'luma':>6} {'recall':>7} {'green':>6} {'half':>6} {'fully':>6}"
          f" {'stage-acc':>10} {'MAE@.25':>8} {'best MAE':>9} {'@conf':>6} {'pred/gt':>8}")
    for row in rows:
        by_stage = row["recall_by_stage"]
        cells = "".join(f"{(by_stage[s] if by_stage[s] is not None else float('nan')):>6.2f}"
                        for s in STAGES)
        ratio = sum(row["pred_totals"].values()) / max(1, sum(row["gt_totals"].values()))
        print(f"{row['value']:>+7.2f} {row['mean_luma']:>6.1f} {row['recall']:>7.3f}{cells}"
              f" {row['stage_accuracy']:>10.3f} {row['counting_mae']:>8.2f}"
              f" {row['best_counting_mae']:>9.2f} {row['best_conf']:>6.2f} {ratio:>8.2f}")


def save_examples(images, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(images[0]))
    for knob, levels in (("stops", EXPOSURE_STOPS), ("wb", WB_SHIFTS)):
        for value in levels:
            cv2.imwrite(str(out_dir / f"{knob}_{value:+.2f}.jpg"), relight(image, **{knob: value}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="n", help="shorthand, runs/<name>, or path to .pt")
    ap.add_argument("--axis", nargs="+", default=list(AXES), choices=list(AXES))
    ap.add_argument("--confs", type=float, nargs="+",
                    default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="first N images, for smoke runs")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-examples", action="store_true",
                    help="skip the example frames; they are written once by the first pass")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(MODELS.get(args.model, args.model))
    if not weights.exists() and (RUNS / args.model / "weights" / "best.pt").exists():
        weights = RUNS / args.model / "weights" / "best.pt"
    assert weights.exists(), f"missing {weights}"
    model = YOLO(str(weights))

    images, lbl_dir, class_names = dataset_files()
    if args.limit:
        images = images[:args.limit]
    print(f"model {args.model} ({weights})  {len(images)} val images")

    rows = {}
    for axis in args.axis:
        rows[axis] = []
        for knob, value in AXES[axis]:
            row = sweep_level(model, images, lbl_dir, class_names, knob, value,
                              args.confs, args.device)
            rows[axis].append(row)
            print(f"  {axis:<14} {value:>+5.2f}  recall {row['recall']:.3f}"
                  f"  stage-acc {row['stage_accuracy']:.3f}"
                  f"  MAE {row['counting_mae']:.2f}  best {row['best_counting_mae']:.2f}"
                  f"@{row['best_conf']:.2f}")
        render(rows[axis], f"{args.model}: {axis}")

    out_dir = RUNS / "lighting"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"{args.model}.json"
    out.write_text(json.dumps({
        "model": args.model, "weights": str(weights), "n_images": len(images),
        "confs": args.confs, "conf_floor": CONF_FLOOR, "axes": rows,
    }, indent=2))
    if not args.no_examples:
        save_examples(images, out_dir / "examples")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
