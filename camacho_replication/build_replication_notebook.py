"""Build replicate_camacho.ipynb (then execute it once so outputs are saved)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# Replicating a published tomato-ripeness model

Supervisor's instruction: first take an existing published work and run the same
thing, then modify. This notebook does that.

Paper:

> Camacho, J.C., Morocho-Cayamcela, M.E. (2023). Mask R-CNN and YOLOv8 Comparison
> to Perform Tomato Maturity Recognition Task. TICEC 2023, CCIS vol. 1885,
> pp. 382-396. Springer.
> [doi:10.1007/978-3-031-45438-7_26](https://doi.org/10.1007/978-3-031-45438-7_26)

Their code and data:
[JeanN00B/Tomato-maturity-recognition---Mask-R-CNN-YOLOv8](https://github.com/JeanN00B/Tomato-maturity-recognition---Mask-R-CNN-YOLOv8)

Why this paper: it can actually be re-run. It uses a public dataset (Laboro
Tomato) and publishes the converted data, the training notebooks with console
output still in them, and weights. Most newer tomato/YOLO11 papers train on
private greenhouse data and release no code, so they can only be cited.

What they report: YOLOv8n-seg vs Mask R-CNN, normal-size fruit only, 3 ripeness
classes, box/mask mAP, plus a separate colour-analysis step giving per-stage
$R^2$ (YOLOv8: 0.809 / 0.897 / 0.968 for ripe / half-ripe / green).""")

md("""## 0. Where the protocol came from

Their CLI cell says `epochs=100`, but the saved console output in the same
notebook shows the run went to `200/200`. So the protocol below is taken from
their executed logs, not from the text.

| | value | source |
|---|---|---|
| task | `segment` | their CLI cell |
| model | YOLOv8n-seg, 3 258 649 params | their val log |
| epochs | 200 | their training log |
| imgsz / optimizer / patience | 640 / Adam / 0 | their CLI cell |
| batch | 16 (309 images -> 20 iters/epoch) | inferred from their log |
| classes | `b_fully_ripened`, `b_green`, `b_half_ripened` | their `data.yaml` |
| split | 309 train / 67 val / 66 test | their log + COCO files |
| framework | ultralytics 8.0.39, torch 1.13.1 | their log header |
| hardware / time | Tesla T4, 1.584 h | their log footer |

The Springer chapter itself is paywalled and has not been read yet, so anything
cited from it in the dissertation still needs checking against the PDF.""")

md("## 1. Environment")
code("""import sys, platform, torch, ultralytics
print("python     ", sys.version.split()[0])
print("platform   ", platform.platform())
print("torch      ", torch.__version__, "| MPS:", torch.backends.mps.is_available())
print("ultralytics", ultralytics.__version__, " (theirs: 8.0.39)")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device     ", DEVICE)""")

md("""## 2. Data

Their repo ships `Big_tomatoes-dataset-2` as COCO only; the YOLO export they
trained on was never committed. `prepare_camacho.py` converts COCO -> YOLO
segmentation labels keeping their splits and class order.

The counts have to match their logged `67 images, 453 instances`.""")
code("""from pathlib import Path
import yaml
DATA = Path("data/camacho_yolo")
cfg = yaml.safe_load((DATA / "data.yaml").read_text())
print("classes:", cfg["names"])
for split in ("train", "valid", "test"):
    imgs = list((DATA / "images" / split).glob("*.jpg"))
    objs = sum(len([l for l in p.read_text().splitlines() if l.strip()])
               for p in (DATA / "labels" / split).glob("*.txt"))
    print(f"{split:>6}: {len(imgs):>4} images, {objs:>5} objects")
print("\\ntheir logged val split: 67 images, 453 instances")""")

md("""### The released weights are not the model from the paper

Cheapest check first: run their published `YOLOv8-seg-trained.pt` on their own
val split. It does not reproduce their table, because the checkpoint is from a
different experiment.

Loading it needs `legacy_shim.py` — it was written by ultralytics 8.0.x, whose
modules lived under the now-removed `ultralytics.yolo.*`.""")
code("""import sys; sys.path.insert(0, ".")
import legacy_shim  # aliases ultralytics.yolo.* -> ultralytics.*
import torch

ck = torch.load("repo/YOLOv8-seg-trained.pt", map_location="cpu", weights_only=False)
ta = ck.get("train_args", {})
print("classes in checkpoint :", list(ck["model"].names.values()))
print("parameters            :", f"{sum(p.numel() for p in ck['model'].parameters()):,}")
print("epochs / optimizer    :", ta.get("epochs"), "/", ta.get("optimizer"))
print("lr0 / batch           :", round(ta.get("lr0", 0), 5), "/", ta.get("batch"))
print("trained on            :", Path(str(ta.get("data", ""))).parent.name)
print("initialised from      :", Path(str(ta.get("model", ""))).name)
print("ultralytics / date    :", ck.get("version"), "/", str(ck.get("date"))[:10])""")

md("""| | released weights | the paper's run |
|---|---|---|
| classes | 6 (`b_*` and `l_*`) | 3 (`b_*` only) |
| parameters | 45.9 M (YOLOv8l-seg) | 3.26 M (YOLOv8n-seg) |
| epochs | 500 | 200 |
| optimizer | SGD, lr0 ~0.0239 | Adam |
| training data | `tomato-dataset-1.3kimg` | `Big_tomatoes` (442 images) |
| initialised from | `last-best-tuned-yolov8.pt` | COCO-pretrained |

So the released artefact cannot verify the published numbers, and there was an
undocumented tuning stage (`last-best-tuned-*`) the paper does not mention. Their
logged notebook output is the only record of the reported table, which is what
this replication targets.""")

md("""### EXIF orientation was already handled upstream

An earlier part of this project found that ~40% of Laboro images carry
`EXIF orientation = 6`: stored pixels landscape, COCO labels upright, so a loader
that ignores EXIF trains on misaligned boxes (see `../RESULTS.md`). The
assumption was that these authors had hit the same thing.""")
code("""print(Path("repo/Tomato-Dataset/Big_tomatoes-dataset-2/README.roboflow.txt")
      .read_text().split("The following pre-processing")[1].split("No image")[0].strip())""")

md("""They exported through Roboflow, which applied auto-orientation automatically, so
the defect never reached their loader. Two consequences: the manual fix is
independently confirmed as a real defect, and their images were also resized to
640x640 by stretching rather than letterboxing, which the paper does not mention
but a replication has to keep.

One more thing visible from the filenames: their val split comes entirely from
one of Laboro's two cameras (`IMG_20191215_*`) while train mixes both. So their
val set is not representative of the dataset. Any comparison against their
numbers inherits that.""")

md("""## 3. Training

Run once via `train_camacho.py`; the cell is left non-executing so the notebook
renders fast.

```bash
python train_camacho.py --model yolov8n-seg.pt --name camacho_v8n_repl
```

which is their configuration verbatim: `task=segment`, `epochs=200`, `imgsz=640`,
`batch=16`, `optimizer=Adam`, `patience=0`, `seed=0`.""")
code("""WEIGHTS = "runs/camacho_v8n_repl/weights/best.pt"
print("weights :", WEIGHTS)
print("ours    : 200 epochs in 1.056 h, Apple M4 Pro (MPS)")
print("theirs  : 200 epochs in 1.584 h, NVIDIA Tesla T4")""")

md("## 4. Results vs the published table")
code("""from ultralytics import YOLO
model = YOLO(WEIGHTS)
m = model.val(data=str(DATA / "data.yaml"), split="val", device=DEVICE, verbose=False)
print(f"Box  mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f} | P {m.box.mp:.3f} | R {m.box.mr:.3f}")
print(f"Mask mAP50 {m.seg.map50:.3f} | mAP50-95 {m.seg.map:.3f}")""")

code("""import pandas as pd

published = {  # their logged val table (best.pt)
    "all":             (0.808, 0.866, 0.893, 0.750),
    "b_fully_ripened": (0.814, 0.853, 0.863, 0.729),
    "b_green":         (0.837, 0.882, 0.918, 0.765),
    "b_half_ripened":  (0.772, 0.864, 0.898, 0.756),
}
ours = {"all": (m.box.mp, m.box.mr, m.box.map50, m.box.map)}
for i, n in enumerate(m.names.values()):
    ours[n] = m.box.class_result(i)

rows = [{"class": cls, "metric": metric, "published": pub[j],
         "ours": round(ours[cls][j], 3), "delta": round(ours[cls][j] - pub[j], 3)}
        for cls, pub in published.items()
        for j, metric in enumerate(("P", "R", "mAP50", "mAP50-95"))]
pd.DataFrame(rows).set_index(["class", "metric"])""")

md("""### Reading it

| Box, all classes | published | replicated | delta |
|---|:--:|:--:|:--:|
| Precision | 0.808 | 0.806 | -0.002 |
| Recall | 0.866 | 0.816 | -0.050 |
| mAP50 | 0.893 | 0.864 | -0.029 |
| mAP50-95 | 0.750 | 0.718 | -0.032 |

Within about 3 points of mAP. The shape of the gap says more than its size:
precision matches almost exactly, recall is lower on every class by a similar
margin (-0.037, -0.050, -0.052, -0.061), and per-class ordering is preserved
(`b_fully_ripened` hardest in both). Random variation would scatter the deltas
both ways. This says the replicated model finds fewer fruit but labels what it
finds just as well, and that the whole mAP gap is recall-driven.

Most likely cause is framework drift. They ran ultralytics 8.0.39, this runs
8.4.94, about eighteen months of changed defaults, including augmentations that
did not exist in 8.0.x (random erasing at 0.4, cutmix) and different NMS/loss
settings. Random erasing is the obvious suspect: occluding parts of objects
during training costs recall and leaves precision alone.

Testable by re-running with those augmentations off. If recall recovers, it
becomes a point worth making in the dissertation: replicability of
ultralytics-based results is bounded by silent changes in framework defaults.""")

md("""## 5. Deviations

Stated rather than tuned away:

| | theirs | here | effect |
|---|---|---|---|
| ultralytics | 8.0.39 | 8.4.94 | changed augmentation/NMS defaults; leading explanation for the recall gap |
| torch / backend | 1.13.1 + CUDA | 2.13.0 + MPS | kernel-level numeric differences |
| hardware | Tesla T4 | Apple M4 Pro | none expected; ran faster (1.06 h vs 1.58 h) |
| label conversion | their unreleased YOLO export | COCO -> YOLO via `prepare_camacho.py` | object counts verified identical |

Not replicated yet: their colour-analysis stage, which produces the per-stage
$R^2$ (0.809 / 0.897 / 0.968). Their scripts are in `repo/Color analysis/`. That
is the next step, since counting rather than mAP is the axis of this work.""")

md("""## 6. Modification: same protocol, newer detector

One variable changed: `yolov8n-seg` -> `yolo11n-seg`. Same split, epochs, imgsz,
optimizer, patience, seed.

```bash
python train_camacho.py --model yolo11n-seg.pt --name camacho_v11n_mod
```""")
code("""import csv
from pathlib import Path

RUN11 = Path("runs/camacho_v11n_mod")
W11, CSV11 = RUN11 / "weights" / "best.pt", RUN11 / "results.csv"
EPOCHS = 200
done = sum(1 for _ in csv.DictReader(open(CSV11))) if CSV11.exists() else 0

if not W11.exists():
    print("run not started")
elif done < EPOCHS:
    # ultralytics writes best.pt continuously, so don't report a mid-run checkpoint
    print(f"still training: epoch {done}/{EPOCHS}, rebuild this notebook when it finishes")
else:
    m11 = YOLO(str(W11)).val(data=str(DATA / "data.yaml"), split="val",
                             device=DEVICE, verbose=False)
    print("                Box mAP50   mAP50-95        P        R")
    print(f"YOLOv8n-seg    {m.box.map50:9.3f} {m.box.map:10.3f} {m.box.mp:8.3f} {m.box.mr:8.3f}   (replication)")
    print(f"YOLO11n-seg    {m11.box.map50:9.3f} {m11.box.map:10.3f} {m11.box.mp:8.3f} {m11.box.mr:8.3f}   (modification)")
    print("\\nparameters: YOLOv8n-seg 3,258,649  vs  YOLO11n-seg 2,835,153")""")

md("""| Box, all | published | replication (v8n) | modification (11n) |
|---|:--:|:--:|:--:|
| mAP50 | 0.893 | 0.864 | 0.863 |
| mAP50-95 | 0.750 | 0.718 | 0.721 |
| Precision | 0.808 | 0.806 | 0.811 |
| Recall | 0.866 | 0.816 | 0.771 |
| parameters | 3.26 M | 3.26 M | 2.84 M |
| training time | 1.58 h (T4) | 1.06 h (M4 Pro) | 1.44 h (M4 Pro) |

The newer detector does not help. mAP50 is unchanged to a thousandth, mAP50-95
and precision move by 0.003 and 0.005 (noise on a 67-image val set), recall drops
another 0.045. It does use 13% fewer parameters, which is a real efficiency gain,
but training took 36% longer on MPS. On 309 training images a newer backbone buys
no accuracy.

This also helps with the recall question. Both architectures sit at recall
0.77-0.82 against their 0.866. If the deficit were specific to YOLOv8n it should
have moved when the model changed; it did not. That points at the pipeline and
framework defaults rather than the model, which is what the version hypothesis
predicts.""")

md("""## 7. Summary

Their protocol was recovered from their logged notebook output, their data
conversion reproduced with matching object counts, and their run repeated in
1.06 h. Result: Box mAP50 0.864 / mAP50-95 0.718 against their published 0.893 /
0.750, with precision matching to 0.002 and the gap carried entirely by recall,
consistent with ultralytics default drift.

Two things came out of re-running rather than citing: the weights they released
are a different model from the one the paper reports, and their Roboflow export
had already stripped EXIF orientation, confirming the data defect found earlier
in this project.

The first modification is a negative result: YOLO11n-seg matches YOLOv8n-seg on
mAP50 and loses recall, at 13% fewer parameters.

Next: replicate their colour-analysis stage for the per-stage $R^2$; test the
augmentation hypothesis; then move on to this project's own contribution,
counting error under controlled ripeness-label noise and on-device inference.""")

nb["cells"] = c
nbf.write(nb, "replicate_camacho.ipynb")
print("wrote replicate_camacho.ipynb —", len(c), "cells")
