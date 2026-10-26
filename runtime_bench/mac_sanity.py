"""
Laptop-side sanity benchmark of every exported model size, on the three sample
images the phone benchmark uses.

Same protocol as the phone's Benchmark screen - each image predicted `repeats`
times per compute-unit setting after a warm-up, median and p90 over all timed
frames - so the two columns of model_size_results.md line up. Also records
package size and load time, the two numbers that grow fastest with model size
and matter on a phone more than on a laptop.

Usage:
    python runtime_bench/mac_sanity.py                     # n s m l x
    python runtime_bench/mac_sanity.py --models n m --repeats 20
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime_bench"))
from bench import CoreMLRunner  # noqa: E402

SAMPLES = ROOT / "ios" / "TomatoRipeness" / "Resources" / "samples"
PACKAGES = {
    "n": ROOT / "ios_model" / "yolo11n_laboro_fp16" / "best.mlpackage",
    "s": ROOT / "ios_model" / "yolo11s_fp16" / "best.mlpackage",
    "m": ROOT / "ios_model" / "yolo11m_fp16" / "best.mlpackage",
    "l": ROOT / "ios_model" / "yolo11l_fp16" / "best.mlpackage",
    "x": ROOT / "ios_model" / "yolo11x_fp16" / "best.mlpackage",
}
UNITS = ["CPU_ONLY", "CPU_AND_NE"]


def package_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def machine():
    return subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True,
                          text=True).stdout.strip() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(PACKAGES))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pad", type=int, default=0, help="0 reproduces Vision's letterbox")
    ap.add_argument("--out", default=str(ROOT / "runs" / "runtime" / "mac_model_sizes.json"))
    args = ap.parse_args()

    images = [cv2.imread(str(p)) for p in sorted(SAMPLES.glob("*.jpg"))]
    rows = []
    print(f"{machine()}  {len(images)} images x {args.repeats} repeats")
    print(f"{'model':>6} {'MB':>7} {'units':<11} {'load ms':>8} {'median':>8} {'p90':>8} {'fps':>6} {'boxes':>6}")
    for name in args.models:
        path = PACKAGES[name]
        if not path.exists():
            print(f"{name:>6}  missing {path}")
            continue
        for units in UNITS:
            started = time.perf_counter()
            runner = CoreMLRunner(path, units, conf=0.25, iou=0.45, pad=args.pad)
            load_ms = (time.perf_counter() - started) * 1000
            for _ in range(args.warmup):
                for image in images:
                    runner(image)
            timings, boxes = [], 0
            for _ in range(args.repeats):
                for image in images:
                    stages, _, _, elapsed = runner(image)
                    timings.append(elapsed)
                    boxes += len(stages)
            row = {
                "model": f"YOLO11{name}", "units": units, "package_mb": package_mb(path),
                "load_ms": load_ms, "median_ms": float(np.median(timings)),
                "p90_ms": float(np.percentile(timings, 90)),
                "fps": 1000.0 / float(np.median(timings)),
                "boxes_per_pass": boxes / args.repeats, "frames": len(timings),
            }
            rows.append(row)
            print(f"{name:>6} {row['package_mb']:>7.1f} {units:<11} {load_ms:>8.0f}"
                  f" {row['median_ms']:>8.2f} {row['p90_ms']:>8.2f} {row['fps']:>6.1f}"
                  f" {row['boxes_per_pass']:>6.0f}")
            del runner

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": machine(), "repeats": args.repeats,
                               "warmup": args.warmup, "pad": args.pad, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
