"""
Export a trained YOLO checkpoint to CoreML for the on-device iOS app.

Bakes NMS and class names into the .mlpackage where the task allows it, so the
app can consume it through Vision as a plain object detector
(VNRecognizedObjectObservation) with no manual post-processing.

Notes:
  - Detection checkpoints support nms=True -> Vision-ready pipeline.
  - Segmentation checkpoints do not: CoreML gets raw outputs and the app has to
    decode boxes/masks itself. The script says which one it produced.
  - Legacy (ultralytics 8.0.x) checkpoints are loadable via legacy_shim.

Usage:
    python export_coreml.py --weights camacho_replication/repo/YOLOv8-seg-trained.pt
    python export_coreml.py --weights runs/<name>/weights/best.pt --half
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="fp16 — halves the file")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default=str(ROOT / "ios_model"))
    args = ap.parse_args()

    weights = Path(args.weights)
    assert weights.exists(), f"missing {weights}"

    # 8.0.x checkpoints pickle classes under the removed ultralytics.yolo.*
    sys.path.insert(0, str(ROOT / "camacho_replication"))
    import legacy_shim  # noqa: F401

    from ultralytics import YOLO

    model = YOLO(str(weights))
    task = model.task
    names = list(model.names.values())
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"loaded   : {weights.name}")
    print(f"task     : {task}")
    print(f"classes  : {len(names)} {names}")
    print(f"params   : {n_params:,}")

    # nms=True builds the Vision-consumable detection pipeline; it is only
    # available for detect. Asking for it on segment fails the export.
    nms = task == "detect"
    if not nms:
        print(f"\n!! task={task}: exporting without the NMS pipeline.")
        print("   Vision will NOT return VNRecognizedObjectObservation for this "
              "model — the app must decode raw outputs itself.")

    out = model.export(
        format="coreml",
        imgsz=args.imgsz,
        half=args.half,
        nms=nms,
        **({"conf": args.conf, "iou": args.iou} if nms else {}),
    )

    dst_dir = Path(args.out)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(out).name
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.move(str(out), str(dst))

    size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"\nwrote    : {dst}  ({size_mb:.1f} MB)")
    print(f"nms baked: {nms}")
    print(f"labels   : {names}")


if __name__ == "__main__":
    main()
