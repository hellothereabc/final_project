"""
Occlusion sweep: what does the detector lose as fruit goes behind leaves.

Runs the same weights over the same validation set many times, hiding a growing
fraction of every annotated fruit (see occlude.py), and reports three things
that fail at different rates:

  detection   is the fruit found at all - class-agnostic recall at IoU 0.5
  stage       given it was found, is the ripeness stage still right
  counting    counting-MAE per image, the metric this thesis is actually about

Detection and stage are separated on purpose. A model that keeps finding fruit
but starts calling half-ripened green has a very different failure than one that
stops seeing fruit, and counting-MAE alone cannot tell them apart - the same
protocol point cross_dataset made about domain shift.

Ground truth is left untouched: the box is where the fruit is, occluded or not,
so recall answers "did it find the fruit" rather than "did it find the visible
part".

Usage:
    python occlusion/run_occlusion.py                          # laboro3, leaf
    python occlusion/run_occlusion.py --data tomatod --occluder black
    python occlusion/run_occlusion.py --model app --data camacho
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from occlude import occlude, read_labels  # noqa: E402
from detmetrics import Score  # noqa: E402
from cross_dataset.taxonomy import STAGES, stage_of  # noqa: E402

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"

MODELS = {
    "laboro3": RUNS / "cross_laboro3" / "weights" / "best.pt",
    "tomatod": RUNS / "cross_tomatod" / "weights" / "best.pt",
    "mix": RUNS / "cross_mix" / "weights" / "best.pt",
    "app": RUNS / "app_yolo11n_det" / "weights" / "best.pt",
}

DATASETS = {
    "laboro3": ROOT / "data" / "cross" / "laboro3",
    "tomatod": ROOT / "data" / "cross" / "tomatod",
    "camacho": ROOT / "camacho_replication" / "data" / "camacho_yolo",
}

# Directory names differ between the two preparations (val vs valid/test).
SPLITS = {"laboro3": "val", "tomatod": "val", "camacho": "test"}


def dataset_files(name):
    root = DATASETS[name]
    split = SPLITS[name]
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    assert img_dir.is_dir(), f"missing {img_dir}"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    names = _class_names(root / "data.yaml")
    return images, lbl_dir, names


def _class_names(yaml_path):
    """names: [...] out of a data.yaml, without a yaml dependency."""
    line = next(l for l in yaml_path.read_text().splitlines() if l.startswith("names"))
    return [n.strip(" '\"") for n in line.split("[", 1)[1].rsplit("]", 1)[0].split(",")]


def sweep_level(model, images, lbl_dir, class_names, fraction, mode, occluder,
                conf, device, seed, iou_threshold=0.5):
    score = Score(iou_threshold)

    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        gt_stages = [stage_of(class_names[cls]) for cls, _ in labels]
        gt_boxes = [box for _, box in labels]

        # seed per image so the same fruit gets the same random side at every
        # level - the curve is then a function of fraction alone
        view = occlude(image, labels, fraction, mode=mode, occluder=occluder,
                       seed=seed + index)
        result = model.predict(view, conf=conf, device=device, verbose=False)[0]
        pred_stages = [stage_of(model.names[int(c)])
                       for c in result.boxes.cls.cpu().numpy()]
        score.add(gt_stages, gt_boxes, pred_stages,
                  result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())

    row = score.summary()
    row.update({"fraction": fraction, "mode": mode, "iou_threshold": iou_threshold})
    return row


def save_examples(images, lbl_dir, modes, fraction, occluder, seed, out_dir):
    """One frame per mode at a single level, for the write-up."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = images[0]
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
    cv2.imwrite(str(out_dir / "clean.jpg"), image)
    for mode in modes:
        view = occlude(image, labels, fraction, mode=mode, occluder=occluder, seed=seed)
        cv2.imwrite(str(out_dir / f"{mode}_{occluder}_{int(fraction * 100)}.jpg"), view)


def render(rows, title):
    print(f"\n{'=' * 86}\n{title}\n{'=' * 86}")
    print(f"{'hidden':>7} {'recall':>8} {'green':>7} {'half':>7} {'fully':>7}"
          f" {'stage-acc':>10} {'conf':>6} {'MAE':>6}")
    for row in rows:
        by_stage = row["recall_by_stage"]
        cells = "".join(f"{(by_stage[s] if by_stage[s] is not None else float('nan')):>7.2f}"
                        for s in STAGES)
        print(f"{row['fraction'] * 100:>6.0f}% {row['recall']:>8.3f}{cells}"
              f" {row['stage_accuracy']:>10.3f} {row['mean_conf']:>6.3f}"
              f" {row['counting_mae']:>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laboro3", help="shorthand or path to .pt")
    ap.add_argument("--data", default="laboro3", choices=list(DATASETS))
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--modes", nargs="+", default=["random", "hide_red", "hide_green"])
    ap.add_argument("--occluder", default="leaf", choices=["leaf", "black"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    # Ground truth is the whole fruit, so a detector that boxes only the visible
    # half scores IoU ~= 1 - fraction against it. At fraction 0.5 that lands
    # exactly on the default threshold, and the recall curve drops for a
    # bookkeeping reason rather than a perceptual one. Re-running at 0.3
    # separates the two.
    ap.add_argument("--iou-thr", type=float, default=0.5,
                    help="IoU a detection needs against the full-fruit box")
    ap.add_argument("--limit", type=int, default=0, help="first N images, for smoke runs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(MODELS.get(args.model, args.model))
    assert weights.exists(), f"missing {weights}"
    model = YOLO(str(weights))

    images, lbl_dir, class_names = dataset_files(args.data)
    if args.limit:
        images = images[:args.limit]
    print(f"model {args.model} ({weights})  data {args.data} "
          f"({len(images)} images, split {SPLITS[args.data]})  occluder {args.occluder}")

    rows = []
    # fraction 0 is the same image whatever the mode, so it is measured once
    if 0.0 in args.levels:
        base = sweep_level(model, images, lbl_dir, class_names, 0.0, "none",
                           args.occluder, args.conf, args.device, args.seed,
                           args.iou_thr)
        rows.append(base)
    for mode in args.modes:
        for fraction in [f for f in args.levels if f > 0]:
            rows.append(sweep_level(model, images, lbl_dir, class_names, fraction,
                                    mode, args.occluder, args.conf, args.device,
                                    args.seed, args.iou_thr))
            row = rows[-1]
            print(f"  {mode:<11} {fraction * 100:>3.0f}%  recall {row['recall']:.3f}"
                  f"  stage-acc {row['stage_accuracy']:.3f}"
                  f"  MAE {row['counting_mae']:.2f}")

    for mode in args.modes:
        subset = [r for r in rows if r["mode"] in ("none", mode)]
        render(subset, f"{args.model} on {args.data}, occluder={args.occluder}, "
                       f"mode={mode}")

    out_dir = RUNS / "occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"{args.model}_{args.data}_{args.occluder}_iou{int(args.iou_thr * 100)}.json"
    out.write_text(json.dumps({
        "model": args.model, "weights": str(weights), "data": args.data,
        "split": SPLITS[args.data], "n_images": len(images), "conf": args.conf,
        "occluder": args.occluder, "seed": args.seed,
        "iou_threshold": args.iou_thr, "levels": rows,
    }, indent=2))
    save_examples(images, lbl_dir, args.modes, 0.4, args.occluder, args.seed,
                  out_dir / "examples")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
