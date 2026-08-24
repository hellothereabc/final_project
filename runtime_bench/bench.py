"""
The same weights, four ways to run them: does the phone see what the laptop sees?

Shipping a detector to a phone changes three things at once - the weights are
cast to fp16, NMS moves inside the model, and the image is letterboxed by Vision
instead of by ultralytics. Any of the three can move a count, and none of them
show up in the training log. This script measures accuracy and latency for every
laptop-side runtime of one checkpoint, so the only unmeasured step left is the
phone itself (runtime_bench/README.md says how that half is collected).

Backends:
  torch-mps / torch-cpu   the checkpoint as trained, ultralytics pre/post
  coreml-<units>          the exported .mlpackage, fp16, NMS baked in, run
                          through CoreML on the requested compute units

Both are scored by detmetrics.py on the same ground truth, so the numbers are
comparable to cross_dataset and to the occlusion sweep.

Preprocessing is a deliberate variable. ultralytics letterboxes with grey 114 at
train and test time; the iOS app asks Vision for `.scaleFit`, which pads with
black. `--pad 0` reproduces the app's version, `--pad 114` the training one, and
the gap between them is what the app's preprocessing choice costs.

Usage:
    python runtime_bench/bench.py --data laboro3
    python runtime_bench/bench.py --data tomatod --backends torch-mps coreml-ALL
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "occlusion"))

from detmetrics import Score  # noqa: E402
from cross_dataset.taxonomy import STAGES, stage_of  # noqa: E402
from occlude import read_labels  # noqa: E402

WEIGHTS = ROOT / "runs" / "cross_mix" / "weights" / "best.pt"
MLPACKAGE = ROOT / "ios" / "TomatoRipeness" / "Resources" / "TomatoRipeness.mlpackage"
DATASETS = {"laboro3": (ROOT / "data" / "cross" / "laboro3", "val"),
            "tomatod": (ROOT / "data" / "cross" / "tomatod", "val")}
COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")


def dataset_files(name):
    root, split = DATASETS[name]
    images = sorted(p for p in (root / "images" / split).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    line = next(l for l in (root / "data.yaml").read_text().splitlines()
                if l.startswith("names"))
    names = [n.strip(" '\"") for n in line.split("[", 1)[1].rsplit("]", 1)[0].split(",")]
    return images, root / "labels" / split, names


def letterbox(image, size=640, pad=114):
    """Scale to fit, pad to square. `pad`=114 is ultralytics, 0 is Vision's."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    dw, dh = (size - new_w) // 2, (size - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, ratio, dw, dh


class CoreMLRunner:
    """The exported pipeline: image in, post-NMS boxes out."""

    def __init__(self, path, compute_units, conf, iou, pad):
        import coremltools as ct

        self.model = ct.models.MLModel(
            str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
        self.conf, self.iou, self.pad = conf, iou, pad
        names = self.model.get_spec().description.metadata.userDefined["names"]
        self.names = [part.split(":")[-1].strip(" '\"{}")
                      for part in names.split(",")]

    def __call__(self, image):
        canvas, ratio, dw, dh = letterbox(image, pad=self.pad)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        started = time.perf_counter()
        out = self.model.predict({"image": pil,
                                  "iouThreshold": self.iou,
                                  "confidenceThreshold": self.conf})
        elapsed = (time.perf_counter() - started) * 1000

        # The exported NMS pipeline always declares 80 confidence columns - a
        # leftover of the COCO head it was built from - and fills only the
        # first len(names). Slicing rather than reshaping keeps that harmless.
        confidence = np.atleast_2d(np.asarray(out["confidence"]))[:, :len(self.names)]
        coordinates = np.asarray(out["coordinates"]).reshape(-1, 4)
        if not len(confidence):
            return [], np.zeros((0, 4)), np.zeros(0), elapsed

        classes = confidence.argmax(axis=1)
        scores = confidence.max(axis=1)
        # normalised cxcywh on the 640 canvas -> xyxy in original pixels
        cx, cy, w, h = (coordinates[:, i] * 640 for i in range(4))
        boxes = np.stack([(cx - w / 2 - dw) / ratio, (cy - h / 2 - dh) / ratio,
                          (cx + w / 2 - dw) / ratio, (cy + h / 2 - dh) / ratio], axis=1)
        keep = scores >= self.conf
        stages = [stage_of(self.names[c]) for c in classes[keep]]
        return stages, boxes[keep], scores[keep], elapsed


class TorchRunner:
    """The checkpoint as trained, through ultralytics' own pre/post-processing."""

    def __init__(self, path, device, conf, iou):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.device, self.conf, self.iou = device, conf, iou

    def __call__(self, image):
        started = time.perf_counter()
        result = self.model.predict(image, conf=self.conf, iou=self.iou,
                                    device=self.device, verbose=False)[0]
        elapsed = (time.perf_counter() - started) * 1000
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        stages = [stage_of(self.model.names[int(c)])
                  for c in result.boxes.cls.cpu().numpy()]
        return stages, boxes, scores, elapsed


def build(backend, conf, iou, pad):
    if backend.startswith("torch-"):
        return TorchRunner(WEIGHTS, backend.split("-", 1)[1], conf, iou)
    units = backend.split("-", 1)[1]
    assert units in COMPUTE_UNITS, f"unknown compute units {units}"
    return CoreMLRunner(MLPACKAGE, units, conf, iou, pad)


def run(backend, images, lbl_dir, class_names, conf, iou, pad, warmup):
    runner = build(backend, conf, iou, pad)
    score = Score()
    timings = []

    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        labels = read_labels(lbl_dir / (path.stem + ".txt"), width, height)
        gt_stages = [stage_of(class_names[cls]) for cls, _ in labels]
        gt_boxes = [box for _, box in labels]

        stages, boxes, scores, elapsed = runner(image)
        score.add(gt_stages, gt_boxes, stages, boxes, scores)
        if index >= warmup:  # first frames pay for lazy graph compilation
            timings.append(elapsed)

    row = score.summary()
    row.update({
        "backend": backend,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_p90": sorted(timings)[int(0.9 * len(timings))],
        "fps_median": 1000 / statistics.median(timings),
        "pad": pad,
    })
    return row


def render(rows, title):
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")
    print(f"{'backend':<18}{'ms':>8}{'fps':>7}{'recall':>9}{'stage-acc':>11}"
          f"{'MAE':>7}{'conf':>7}{'IoU':>7}   counts g/h/f")
    for row in rows:
        totals = "/".join(str(row["pred_totals"].get(s, 0)) for s in STAGES)
        print(f"{row['backend']:<18}{row['latency_ms_median']:>8.1f}"
              f"{row['fps_median']:>7.1f}{row['recall']:>9.3f}"
              f"{row['stage_accuracy']:>11.3f}{row['counting_mae']:>7.2f}"
              f"{row['mean_conf']:>7.3f}{row['mean_iou']:>7.3f}   {totals}")
    gt = "/".join(str(rows[0]["gt_totals"].get(s, 0)) for s in STAGES)
    print(f"{'ground truth':<18}{'':>8}{'':>7}{'':>9}{'':>11}{'':>7}{'':>7}{'':>7}   {gt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="laboro3", choices=list(DATASETS))
    ap.add_argument("--backends", nargs="+",
                    default=["torch-mps", "torch-cpu", "coreml-CPU_ONLY",
                             "coreml-CPU_AND_NE", "coreml-ALL"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--pad", type=int, default=114,
                    help="letterbox fill: 114 = ultralytics, 0 = Vision .scaleFit")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    images, lbl_dir, class_names = dataset_files(args.data)
    if args.limit:
        images = images[:args.limit]
    print(f"weights {WEIGHTS.relative_to(ROOT)}  mlpackage {MLPACKAGE.name}\n"
          f"data {args.data} ({len(images)} images)  conf {args.conf}  "
          f"iou {args.iou}  pad {args.pad}")

    rows = []
    for backend in args.backends:
        print(f"  running {backend} ...", flush=True)
        rows.append(run(backend, images, lbl_dir, class_names, args.conf,
                        args.iou, args.pad, args.warmup))
    render(rows, f"one checkpoint, {len(args.backends)} runtimes - {args.data}, pad={args.pad}")

    out_dir = ROOT / "runs" / "runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"bench_{args.data}_pad{args.pad}.json"
    out.write_text(json.dumps({"data": args.data, "weights": str(WEIGHTS),
                               "mlpackage": str(MLPACKAGE), "conf": args.conf,
                               "iou": args.iou, "pad": args.pad, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
