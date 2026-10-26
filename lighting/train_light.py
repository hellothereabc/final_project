"""
Lighting augmentation: does turning up HSV jitter at training time buy anything
under a real change of light at test time.

Ultralytics' colour augmentation is an HSV jitter - hsv_h, hsv_s, hsv_v are the
half-widths of a uniform gain on each channel, defaults 0.015 / 0.7 / 0.4. That
is the only colour knob in the pipeline, so "simulate lighting with HSV/HSL" in
this codebase means moving those three numbers. The question is whether a model
trained under wider jitter holds its recall and stage calls further along the
exposure and white-balance sweep in run_lighting.py than the stock recipe does.

README section 6 already learned the opposite lesson the hard way: removing the
jitter (`no_colour`) looked like a win at conf 0.25 and was not once the
threshold was swept. So the floor is re-trained here under this protocol and
every row is scored with a sweep.

Protocol is train.py's, unchanged, so the reference row is the existing
laboro_yolo11n_exiffix run: YOLO11n, six classes, 60 epochs, imgsz 640,
batch 16, MPS, patience 20. One group changes per config.

Seeds. The first pass of this section was one seed per config, which cannot
separate a real effect from training noise. `--seed` repeats a config; the
`stock` config is the unchanged recipe, so the spread of `stock` across seeds
is the noise floor every other difference has to clear. Run directories are
`runs/light_<config>_s<seed>`, except for the seed-0 runs of the first pass,
which keep their original `runs/light_<config>` names and are never touched.
The seed-0 `stock` row is the existing `runs/laboro_yolo11n_export`.

Usage:
    python lighting/train_light.py --config hsv_v08
    python lighting/train_light.py --all                 # ~45 min per config on M4 Pro
    python lighting/train_light.py --all --seed 1 2      # repeat every config
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "laboro_yolo" / "data.yaml"

# Reference row: the stock recipe, already trained.
BASELINE_RUN = ROOT / "runs" / "laboro_yolo11n_export"

CONFIGS = {
    # the unchanged recipe: the seed-repeat of this config is the noise floor
    "stock": {},
    # brightness jitter doubled, hue and saturation stock - exposure only
    "hsv_v08": {"hsv_v": 0.8},
    # brightness doubled and hue widened ~3x: a stand-in for white balance,
    # since a hue rotation is the nearest thing HSV has to a colour cast
    "hsv_wide": {"hsv_h": 0.05, "hsv_v": 0.8},
    # no colour jitter at all: the floor, section 6's control under this protocol
    "no_colour": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
}


def completed(run_dir, epochs):
    results = run_dir / "results.csv"
    weights = run_dir / "weights" / "best.pt"
    if not results.exists() or not weights.exists():
        return False
    # patience 20 may stop early; a run counts as done once last.pt exists and
    # ultralytics wrote its final results row
    return (run_dir / "weights" / "last.pt").exists() and sum(1 for _ in results.open()) > 1


def run_dir_for(name, seed):
    """Seed 0 keeps the first pass's directory names; everything else is suffixed."""
    legacy = ROOT / "runs" / f"light_{name}"
    if seed == 0 and legacy.exists():
        return legacy
    return ROOT / "runs" / f"light_{name}_s{seed}"


def run(name, seed, args):
    from ultralytics import YOLO

    overrides = CONFIGS[name]
    run_dir = run_dir_for(name, seed)
    print(f"\n{'=' * 70}\n{name} seed {seed}: {overrides} -> {run_dir.name}\n{'=' * 70}")
    YOLO(args.model).train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        seed=seed,
        project=str(ROOT / "runs"),
        name=run_dir.name,
        exist_ok=False,
        patience=20,
        val=True,
        **overrides,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, nargs="+", default=[0],
                    help="one run per seed per config")
    args = ap.parse_args()

    if not args.all and not args.config:
        ap.error("pass --config or --all")
    assert DATA.exists(), f"missing {DATA} - run prepare_data.py first"
    assert BASELINE_RUN.exists(), f"missing {BASELINE_RUN} - the reference row is train.py's n run"

    for seed in args.seed:
        for name in (list(CONFIGS) if args.all else [args.config]):
            if name == "stock" and seed == 0:
                print("skipping stock seed 0: it is runs/laboro_yolo11n_export")
                continue
            run_dir = run_dir_for(name, seed)
            if run_dir.exists() and completed(run_dir, args.epochs):
                print(f"skipping {name} seed {seed}: runs/{run_dir.name} already complete")
                continue
            if run_dir.exists():
                raise SystemExit(
                    f"runs/{run_dir.name} exists but is incomplete; move it aside before rerunning")
            run(name, seed, args)


if __name__ == "__main__":
    sys.exit(main())
