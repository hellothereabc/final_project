# Tomato ripeness detection and counting

MSc work. Public datasets only.

Order of work: baseline first, then replicate a published paper as published, then modify.

## 1. Baseline on Laboro Tomato

`reproduce_laboro.ipynb` — YOLO11n, 60 epochs, seed 0, MPS, native 643/161 split, 6 classes.

| | ours | Laboro published baseline |
|---|:--:|:--:|
| Box AP (mAP50-95) | 0.706 | 0.643 |
| mAP50 | 0.842 | — |
| ripe counting-MAE per image | 0.82 | — |

Reference: [laboroai/LaboroTomato](https://github.com/laboroai/LaboroTomato) (Mask R-CNN R-50-FPN, MMDetection, 48 epochs, 4xV100).

About 40% of Laboro images have an EXIF orientation=6 tag. The pixels are landscape, the COCO labels are upright, and the YOLO loader ignores EXIF, so boxes did not match the fruit. `prepare_data.py` bakes the rotation in. mAP barely changed (train and val were both wrong), counting-MAE went 0.93 -> 0.82.

This is a baseline anchor, not a replication.

## 2. Replication

Paper:

> Camacho, J.C., Morocho-Cayamcela, M.E. (2023). Mask R-CNN and YOLOv8 Comparison to Perform Tomato Maturity Recognition Task. TICEC 2023, CCIS vol. 1885, pp. 382-396. Springer. [doi:10.1007/978-3-031-45438-7_26](https://doi.org/10.1007/978-3-031-45438-7_26)

Their code and data: [JeanN00B/Tomato-maturity-recognition---Mask-R-CNN-YOLOv8](https://github.com/JeanN00B/Tomato-maturity-recognition---Mask-R-CNN-YOLOv8)

Their protocol, unchanged: YOLOv8n-seg, 200 epochs, imgsz 640, Adam, patience 0, their 309/67 split, 3 classes.

| Box, all classes | published | replicated | YOLO11n-seg |
|---|:--:|:--:|:--:|
| mAP50 | 0.893 | 0.864 | 0.863 |
| mAP50-95 | 0.750 | 0.718 | 0.721 |
| Precision | 0.808 | 0.806 | 0.811 |
| Recall | 0.866 | 0.816 | 0.771 |
| parameters | 3.26M | 3.26M | 2.84M |

200 epochs took 1.06 h on an M4 Pro; theirs took 1.58 h on a Tesla T4.

Precision matches to 0.002, recall is lower on every class. The gap is recall-driven. Likely cause is ultralytics version drift: they used 8.0.39, this uses 8.4.94, and the newer defaults add augmentations (random erasing, cutmix) that were not there before. Not tuned to close the gap.

Swapping YOLOv8n-seg for YOLO11n-seg under the same protocol changes nothing on mAP50 and loses recall, with 13% fewer parameters. The recall gap survives the architecture change, so it comes from the pipeline rather than the model.

Two things found by re-running rather than citing:

1. The weights they released are a different model from the one in the paper: YOLOv8l-seg, 6 classes, 500 epochs, SGD, ~1300 images, started from an undocumented tuned checkpoint. The released weights cannot verify the published numbers.
2. Their Roboflow export stripped EXIF orientation automatically, so they never hit the bug from section 1.

Write-up: [`camacho_replication/replicate_camacho.ipynb`](camacho_replication/replicate_camacho.ipynb).

## Running it

```bash
python -m venv .venv && .venv/bin/pip install ultralytics nbformat nbconvert

# baseline (download Laboro Tomato first)
python prepare_data.py && python train.py && python evaluate.py

# replication
cd camacho_replication
git clone --depth 1 https://github.com/JeanN00B/Tomato-maturity-recognition---Mask-R-CNN-YOLOv8.git repo
python prepare_camacho.py
python train_camacho.py --model yolov8n-seg.pt --name camacho_v8n_repl
python train_camacho.py --model yolo11n-seg.pt --name camacho_v11n_mod
```

Datasets, weights and runs are gitignored. Laboro Tomato is CC BY-NC-SA 4.0, not redistributed here.

## Next

- Their colour-analysis step, to get the per-stage R2 (0.809 / 0.897 / 0.968). Counting is the axis of this work, not mAP.
- Check the augmentation hypothesis behind the recall gap.
- Own contribution: counting error under controlled ripeness-label noise, plus on-device inference (CoreML/SwiftUI app already runs the detector).
