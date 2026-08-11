"""
Train the detector that ships inside the iOS app.

Separate from train_camacho.py on purpose: that one reproduces the paper's
protocol verbatim and must not drift. This one optimises for something else —
a small model that exports cleanly to CoreML with NMS baked in.

Detection, not segmentation: only detect checkpoints can be exported with the
CoreML NMS pipeline, which is what lets Vision return ready-made
VNRecognizedObjectObservation instead of raw tensors the app must decode.

Trains on the Camacho split (already on disk after prepare_camacho.py), 3
classes — b_green / b_half_ripened / b_fully_ripened — which map one-to-one onto
the app's three counters. Ultralytics derives boxes from the polygon labels, so
no separate detection export of the dataset is needed.

Usage:
    python camacho_replication/prepare_camacho.py     # once
    python train_app_model.py --epochs 60
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "camacho_replication" / "data" / "camacho_yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="app_yolo11n_det")
    args = ap.parse_args()

    assert DATA.exists(), f"missing {DATA} — run camacho_replication/prepare_camacho.py first"

    YOLO(args.model).train(
        data=str(DATA),
        task="detect",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=0,
        project=str(ROOT / "runs"),
        name=args.name,
        patience=0,   # fixed-length run so the epoch count is predictable
        val=True,
    )


if __name__ == "__main__":
    main()
