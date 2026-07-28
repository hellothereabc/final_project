"""
Replicate the Camacho & Morocho-Cayamcela (TICEC 2023) training run, then repeat
the identical protocol with YOLO11n-seg as the single-variable modification.

Their protocol, recovered from the console output saved in
`YOLOv8 - custom.ipynb` (the paper's CLI cell says epochs=100, but the log shows
the run that produced the reported numbers went to 200):

    yolo task=segment mode=train model=yolov8n-seg.pt
         epochs=200 imgsz=640 optimizer=Adam patience=0
    # batch 16 (309 train images -> 20 iterations/epoch in their log)
    # ultralytics 8.0.39, torch 1.13.1, Tesla T4, 1.584 h

Usage:
    python train_camacho.py --model yolov8n-seg.pt --name camacho_v8n_repl
    python train_camacho.py --model yolo11n-seg.pt --name camacho_v11n_mod
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "camacho_yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--name", default="camacho_v8n_repl")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    YOLO(args.model).train(
        data=str(DATA),
        task="segment",
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        optimizer="Adam",
        patience=0,      # their setting: never early-stop
        seed=0,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
