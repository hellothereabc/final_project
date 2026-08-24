"""
Same fruit, two occlusions: does hiding the red half and hiding the green half
move the ripeness call in opposite directions?

run_occlusion.py answers this only in aggregate, and aggregates hide the effect:
the fruit that survive occlusion are a biased subset, and averaging over green
fruit - which have no red half to hide - dilutes whatever happens to the
half-ripened ones. This is the paired version. Every annotated fruit is scored
twice, once with its redder half hidden and once with its greener half hidden,
and only fruit detected in *both* passes are compared, so each fruit is its own
control.

The number reported is a shift on the stage scale (green 0, half 1, fully 2):

    shift = mean(stage | green half hidden) - mean(stage | red half hidden)

Positive means the same fruit reads riper when its green side is covered, which
is the direction the colour boundaries in cross_dataset/colour_stages.py
predict. If the shift is ~0, the model is not reading ripeness off the visible
colour of the fruit at all, and its stage calls survive occlusion for a reason
that needs a different explanation.

Usage:
    python occlusion/colour_flip.py --fraction 0.4
    python occlusion/colour_flip.py --model mix --fraction 0.5 --data tomatod
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

from occlude import occlude, read_labels  # noqa: E402
from detmetrics import match  # noqa: E402
from run_occlusion import MODELS, dataset_files  # noqa: E402
from cross_dataset.taxonomy import SHORT, STAGES, stage_of  # noqa: E402

RANK = {stage: index for index, stage in enumerate(STAGES)}


def per_fruit(model, images, lbl_dir, class_names, fraction, mode, conf, device, seed):
    """{(image, gt index): predicted stage or None} for one occlusion mode."""
    calls = {}
    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        if not labels:
            continue
        gt_boxes = [box for _, box in labels]

        view = occlude(image, labels, fraction, mode=mode, occluder="leaf",
                       seed=seed + index)
        result = model.predict(view, conf=conf, device=device, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_conf = result.boxes.conf.cpu().numpy()
        pred_stages = [stage_of(model.names[int(c)])
                       for c in result.boxes.cls.cpu().numpy()]
        pairs = match(gt_boxes, pred_boxes, pred_conf)

        for gt_index, (cls, _) in enumerate(labels):
            key = (path.name, gt_index)
            hit = pairs.get(gt_index)
            calls[key] = (stage_of(class_names[cls]),
                          pred_stages[hit] if hit is not None else None)
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laboro3")
    ap.add_argument("--data", default="laboro3")
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(MODELS.get(args.model, args.model))
    model = YOLO(str(weights))
    images, lbl_dir, class_names = dataset_files(args.data)
    if args.limit:
        images = images[:args.limit]
    print(f"model {args.model}  data {args.data}  fraction {args.fraction}  "
          f"{len(images)} images")

    passes = {}
    for mode in ("hide_red", "hide_green"):
        print(f"  pass {mode} ...", flush=True)
        passes[mode] = per_fruit(model, images, lbl_dir, class_names, args.fraction,
                                 mode, args.conf, args.device, args.seed)

    red, green = passes["hide_red"], passes["hide_green"]
    rows, flips = {}, defaultdict(Counter)
    for key, (gt_stage, red_call) in red.items():
        green_call = green.get(key, (None, None))[1]
        if red_call is None or green_call is None:
            continue          # only fruit found in both passes are comparable
        rows.setdefault(gt_stage, []).append((red_call, green_call))
        flips[gt_stage][(red_call, green_call)] += 1

    print(f"\nfruit detected in both passes: "
          f"{sum(len(v) for v in rows.values())} of {len(red)}")
    print(f"\n{'GT stage':<14}{'n':>6}{'mean|red hidden':>17}{'mean|green hidden':>19}"
          f"{'shift':>8}{'disagree':>10}")
    summary = {}
    for stage in STAGES:
        pairs = rows.get(stage, [])
        if not pairs:
            continue
        red_rank = np.mean([RANK[a] for a, _ in pairs])
        green_rank = np.mean([RANK[b] for _, b in pairs])
        disagree = sum(1 for a, b in pairs if a != b) / len(pairs)
        print(f"{SHORT[stage]:<14}{len(pairs):>6}{red_rank:>17.3f}{green_rank:>19.3f}"
              f"{green_rank - red_rank:>+8.3f}{disagree:>10.1%}")
        summary[stage] = {"n": len(pairs), "mean_hide_red": float(red_rank),
                          "mean_hide_green": float(green_rank),
                          "shift": float(green_rank - red_rank),
                          "disagreement": float(disagree)}

    for stage in STAGES:
        pairs = flips[stage]
        moved = {k: v for k, v in pairs.items() if k[0] != k[1]}
        if not moved:
            continue
        print(f"\n{SHORT[stage]} fruit that changed call "
              f"(hide_red -> hide_green), top 4:")
        for (a, b), count in sorted(moved.items(), key=lambda kv: -kv[1])[:4]:
            print(f"    {SHORT[a]:>6} when red hidden  ->  {SHORT[b]:<6} "
                  f"when green hidden   x{count}")

    out_dir = ROOT / "runs" / "occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else \
        out_dir / f"colour_flip_{args.model}_{args.data}_{int(args.fraction * 100)}.json"
    out.write_text(json.dumps({
        "model": args.model, "data": args.data, "fraction": args.fraction,
        "conf": args.conf, "by_stage": summary,
        "flips": {stage: {f"{a}->{b}": count for (a, b), count in counts.items()}
                  for stage, counts in flips.items()},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
