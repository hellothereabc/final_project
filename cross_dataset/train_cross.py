"""
One detector per dataset plus one on the mix, identical protocol.

The protocol is the comparison, so nothing but the training data changes between
the three runs. patience=0 keeps all three at exactly 60 epochs instead of
stopping at data-dependent points. Matches train_app_model.py.

Usage:
    python cross_dataset/train_cross.py --dataset laboro3
    python cross_dataset/train_cross.py --dataset tomatod
    python cross_dataset/train_cross.py --dataset mix
    python cross_dataset/train_cross.py --all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "cross"

DATASETS = {
    "laboro3": DATA / "laboro3" / "data.yaml",
    "tomatod": DATA / "tomatod" / "data.yaml",
    "mix": DATA / "mix.yaml",
}


def run(dataset, args):
    from ultralytics import YOLO

    yaml = DATASETS[dataset]
    assert yaml.exists(), f"missing {yaml} - run cross_dataset/prepare.py first"

    print(f"\n{'=' * 60}\ntraining {dataset}  ({yaml})\n{'=' * 60}")
    YOLO(args.model).train(
        data=str(yaml),
        task="detect",
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=args.device,
        seed=0,
        patience=0,      # all three runs end at the same epoch
        project=str(ROOT / "runs"),
        name=f"cross_{dataset}",
        exist_ok=False,
        val=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--all", action="store_true", help="all three, in order")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    if not args.all and not args.dataset:
        ap.error("pass --dataset or --all")

    for name in (list(DATASETS) if args.all else [args.dataset]):
        run(name, args)


if __name__ == "__main__":
    sys.exit(main())
