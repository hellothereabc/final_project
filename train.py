"""
Train a YOLO detection baseline on Laboro Tomato (MPS).

Reproduce-first baseline: off-the-shelf YOLO11n, single protocol, fixed seed.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "laboro_yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="laboro_yolo11n")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=0,
        project=str(ROOT / "runs"),
        name=args.name,
        patience=20,
        val=True,
    )


if __name__ == "__main__":
    main()
