"""
Augmentation ablation: which of the default augmentations this task actually needs.

Two questions, one protocol.

1. Ripeness is a colour judgement, and the default recipe jitters colour hard
   (hsv_s 0.7, hsv_v 0.4). cross_dataset showed the green/half/fully boundary
   sits on a* in CIELAB and that the two datasets place it differently. So
   colour jitter is not a free win here the way it is on COCO - it may be
   smearing exactly the signal the stage label depends on.

2. README section 2 blames the unclosed recall gap in the Camacho replication on
   "newer defaults adding augmentations (random erasing, cutmix)". Reading
   ultralytics 8.4.115 rather than its docs, that is wrong: `erasing` and
   `auto_augment` are consumed only by ClassificationDataset, and `cutmix`
   defaults to 0.0. Detection training here runs mosaic + hsv + translate/scale
   + fliplr and nothing else. This ablation bounds how much recall any
   augmentation change can move at all - if the whole recipe is worth less than
   the 5-point gap, the augmentation explanation is dead however 8.0.39 was
   configured.

Everything except the augmentation hyperparameters is frozen at the
cross_dataset protocol (YOLO11n detect, laboro3, 60 epochs, imgsz 640, batch 16,
seed 0, patience 0), so `default` needs no run of its own: it is the existing
runs/cross_laboro3. Each config below changes one group and nothing else.

Usage:
    python augmentation/train_aug.py --config no_colour
    python augmentation/train_aug.py --all           # ~45 min per config
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# The reference row: same protocol, already trained by cross_dataset.
BASELINE_RUN = ROOT / "runs" / "cross_laboro3"

CONFIGS = {
    # colour jitter off - the group this task has most reason to suspect
    "no_colour": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
    # mosaic off - four-image tiling changes fruit scale and context density
    "no_mosaic": {"mosaic": 0.0},
    # geometry off - translate/scale/flip
    "no_geometry": {"translate": 0.0, "scale": 0.0, "fliplr": 0.0, "degrees": 0.0},
    # nothing but the letterbox: the floor the rest is measured against
    "none": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "mosaic": 0.0,
             "translate": 0.0, "scale": 0.0, "fliplr": 0.0, "degrees": 0.0,
             "erasing": 0.0},
    # occlusion-shaped augmentation, on purpose: cutmix pastes a patch of another
    # image over this one, which is the training-time version of the occlusion/
    # sweep. If it buys anything, it should show up there rather than here.
    "occlusion_aug": {"cutmix": 0.15, "mixup": 0.1},
    # more of everything, to check the ablation is not just measuring "less is better"
    "heavy": {"hsv_h": 0.03, "hsv_s": 0.9, "hsv_v": 0.6, "scale": 0.8,
              "translate": 0.2, "degrees": 10.0, "mosaic": 1.0, "cutmix": 0.15},
}

DATA = ROOT / "data" / "cross" / "laboro3" / "data.yaml"


def run(name, args):
    from ultralytics import YOLO

    overrides = CONFIGS[name]
    print(f"\n{'=' * 70}\n{name}: {overrides or 'ultralytics defaults'}\n{'=' * 70}")
    YOLO(args.model).train(
        data=str(DATA),
        task="detect",
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=args.device,
        seed=0,
        patience=0,
        project=str(ROOT / "runs"),
        name=f"aug_{name}",
        exist_ok=False,
        val=True,
        **overrides,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if not args.all and not args.config:
        ap.error("pass --config or --all")
    assert DATA.exists(), f"missing {DATA} - run cross_dataset/prepare.py first"
    assert BASELINE_RUN.exists(), (
        f"missing {BASELINE_RUN} - the `default` row is cross_dataset's laboro3 run")

    for name in (list(CONFIGS) if args.all else [args.config]):
        if (ROOT / "runs" / f"aug_{name}").exists():
            print(f"skipping {name}: runs/aug_{name} already exists")
            continue
        run(name, args)


if __name__ == "__main__":
    sys.exit(main())
