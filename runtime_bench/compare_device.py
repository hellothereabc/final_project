"""
Line up the phone's own numbers against the laptop's, on identical images.

BenchmarkView in the iOS app writes a JSON blob (Copy JSON on the Benchmark
screen); this reads it, re-runs the same .mlpackage here over the same bundled
sample images, and reports where the two disagree:

  latency    phone per compute-unit setting vs laptop per compute-unit setting
  counts     per class, per image - the number the app actually shows a user
  geometry   IoU between the phone's boxes and the laptop's, matched greedily

Preprocessing is the reason to bother. Vision letterboxes with `.scaleFit`
(black padding) while ultralytics trains and validates with grey 114, so this is
also the measurement of what that difference costs: `--pad 0` reproduces Vision,
`--pad 114` reproduces training, and whichever agrees with the phone is the one
the app is really running.

Usage:
    python runtime_bench/compare_device.py phone.json
    python runtime_bench/compare_device.py phone.json --pad 0
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime_bench"))

from detmetrics import iou_matrix, match  # noqa: E402
from cross_dataset.taxonomy import stage_of  # noqa: E402
from bench import CoreMLRunner, MLPACKAGE  # noqa: E402

SAMPLES = ROOT / "ios" / "TomatoRipeness" / "Resources" / "samples"


def phone_boxes(entry):
    """Vision rects (normalised, origin bottom-left) -> xyxy pixels."""
    width, height = entry["width"], entry["height"]
    boxes, labels, scores = [], [], []
    for det in entry["detections"]:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        boxes.append([x * width, (1 - y - h) * height,
                      (x + w) * width, (1 - y) * height])
        labels.append(det["label"])
        scores.append(det["conf"])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), labels, np.asarray(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phone_json", help="JSON copied out of the app's Benchmark screen")
    ap.add_argument("--units", default="CPU_ONLY",
                    help="laptop compute units to compare geometry against")
    ap.add_argument("--pad", type=int, default=0,
                    help="0 reproduces Vision .scaleFit, 114 reproduces training")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    phone = json.loads(Path(args.phone_json).read_text())
    print(f"phone: {phone.get('device')} / {phone.get('os')}  "
          f"repeats {phone.get('repeats')}")

    runner = CoreMLRunner(MLPACKAGE, args.units, args.conf, args.iou, args.pad)

    print(f"\n{'latency':<22}{'median ms':>11}{'fps':>8}{'thermal':>12}")
    for config in phone["configs"]:
        print(f"phone {config['computeUnits']:<16}{config['median_ms']:>11.1f}"
              f"{1000 / config['median_ms']:>8.1f}{config.get('thermal_state', '-'):>12}")

    reference = next((c for c in phone["configs"] if c["computeUnits"] == "cpuOnly"),
                     phone["configs"][0])

    print(f"\n{'image':<12}{'phone':>18}{'laptop':>18}{'matched':>9}{'mean IoU':>10}"
          f"{'label ==':>10}")
    timings = []
    for entry in reference["images"]:
        path = SAMPLES / f"{entry['name']}.jpg"
        assert path.exists(), f"missing {path}"
        image = cv2.imread(str(path))
        stages, boxes, scores, elapsed = runner(image)
        timings.append(elapsed)

        p_boxes, p_labels, p_scores = phone_boxes(entry)
        pairs = match(p_boxes, boxes, scores)
        ious = iou_matrix(p_boxes, boxes)
        agree = sum(1 for gt_idx, pred_idx in pairs.items()
                    if stage_of(p_labels[gt_idx]) == stages[pred_idx])
        mean_iou = (np.mean([ious[g, p] for g, p in pairs.items()]) if pairs else float("nan"))
        print(f"{entry['name']:<12}{len(p_boxes):>18}{len(boxes):>18}"
              f"{len(pairs):>9}{mean_iou:>10.3f}"
              f"{(agree / len(pairs) if pairs else float('nan')):>10.2f}")

    laptop_ms = float(np.median(timings))
    print(f"\nlaptop coreml-{args.units} (pad={args.pad}): median {laptop_ms:.1f} ms "
          f"({1000 / laptop_ms:.1f} fps)")
    print(f"phone  {reference['computeUnits']}: median {reference['median_ms']:.1f} ms "
          f"({1000 / reference['median_ms']:.1f} fps)  "
          f"-> phone is {reference['median_ms'] / laptop_ms:.2f}x the laptop")


if __name__ == "__main__":
    sys.exit(main())
