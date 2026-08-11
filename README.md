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

## 3. Two datasets, one label space

Sections 1 and 2 both ran on Laboro Tomato - one annotation team, one definition
of "half ripened". This section adds [tomatOD](https://github.com/up2metric/tomatOD)
(277 images, Greek greenhouse, independent annotators) and asks what survives the
change of source.

Both datasets collapse onto a shared 3-stage space (`cross_dataset/taxonomy.py`),
native splits, one protocol: YOLO11n detect, 60 epochs, imgsz 640, batch 16,
seed 0, patience 0. Three training sets, two validation sets.

**mAP50-95 / counting-MAE per image** (bold = in-domain):

| model \ val | Laboro (161) | tomatOD (55) |
|---|:--:|:--:|
| laboro3 | **0.698** / **1.25** | 0.389 / 3.29 |
| tomatod | 0.308 / 2.14 | **0.563** / **1.00** |
| mix | 0.704 / 1.14 | 0.546 / 1.24 |

The 6-class baseline in section 1 (0.706) and `laboro3` are not comparable -
fewer classes is an easier task.

### The datasets disagree about what "ripe" means

Every annotated box reduced to its CIELAB a\* (green->red), measured against its
own image's background so white balance cancels out:

| stage | Laboro | tomatOD |
|---|:--:|:--:|
| green | -5.6 | -4.2 |
| half | +18.9 | +6.6 |
| fully | +41.7 | +13.6 |

They agree on green and diverge above it. Fitting each dataset's own boundaries
and swapping them, **98% of what tomatOD calls fully ripe falls inside Laboro's
half-ripened band**; 68% of Laboro's half would be fully ripe to tomatOD. On raw
a\* the green medians are 10 points apart and that gap disappears under the
background correction - the half/fully gap widens instead. It is the annotation,
not the camera.

A single-source model inherits its source's convention: `laboro3` on tomatOD
over-predicts half by **+353%** and under-predicts fully by **-51%** while
over-predicting everything else. The only class that collapses is the one the
colour boundaries said would, in the predicted direction. The reverse direction
looks nothing like it - `tomatod` on Laboro just fails to find fruit (recall
0.88 -> 0.48) and undercounts every stage alike.

### Mixing resolves the conflict rather than blurring it

This contradicts the prediction recorded before the run. The mix matches the
Laboro specialist on Laboro (0.704 vs 0.698), costs 0.017 mAP on tomatOD, and
lifts the cross cells from 0.308 -> 0.704 and 0.389 -> 0.546. It also does not
inherit the schema bias: where `laboro3` put tomatOD's fully at -51%, the mix
puts it at +36%, the same direction as tomatOD's own specialist, despite Laboro
supplying 80% of the boxes.

The domains are visually distinguishable, so the network never has to learn one
global rule - it learns "ripe begins here in images like these, there in images
like those". **Systematic label noise that correlates with an identifiable domain
is close to harmless**; the damaging kind is noise not recoverable from the image.
Perturbing a fixed percentage of labels at random, the usual protocol, mixes the
two regimes and cannot separate them.

Write-up: [`cross_dataset/cross_dataset.ipynb`](cross_dataset/cross_dataset.ipynb).

Two defects found by converting rather than reading: `tomatOD_test.json` declares
a single category whose name is a list of all three, while its annotations use
three ids (reading categories per split silently collapses the test set onto one
class); and the shipped annotations hold 2 421 boxes against the 2 418 its README
advertises.

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

# cross-dataset (downloads both, ~1.8 GB; the whole chain is one script)
bash cross_dataset/run_all.sh
python cross_dataset/colour_stages.py
```

Datasets, weights and runs are gitignored. Laboro Tomato and tomatOD are both
CC BY-NC-SA 4.0, not redistributed here.

## Next

- Their colour-analysis step, to get the per-stage R2 (0.809 / 0.897 / 0.968). Counting is the axis of this work, not mAP.
- Check the augmentation hypothesis behind the recall gap.
- A third annotation source (KUTomaData) - two datasets give one pairwise comparison, which cannot separate "schemas generally disagree" from "one of these two is idiosyncratic". Three enable leave-one-out. Camacho's data does not count: it is Laboro re-exported through Roboflow, not independent annotation.
- Own contribution: counting error under label noise, now sharpened by section 3 - separate noise that is recoverable from the image (a domain the model can identify) from noise that is not. The usual protocol, flipping a fixed percentage of labels at random, conflates the two.
- On-device inference (CoreML/SwiftUI app already runs the detector).
