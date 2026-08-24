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

Precision matches to 0.002, recall is lower on every class. The gap is recall-driven, and the cause is still open. An earlier note here blamed augmentation defaults drifting between their 8.0.39 and this 8.4.x; reading the installed source rather than the docs, that is not it - `erasing` and `auto_augment` are read only by `ClassificationDataset`, and `cutmix` defaults to 0.0, so detection training here runs mosaic + hsv + translate/scale/fliplr and nothing else. `augmentation/` bounds how much any augmentation change can move recall at all.

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

The counting numbers in this section are all taken at conf 0.25. Section 6 shows
that threshold is doing more of the transfer penalty than the domain shift is:
swept, laboro3's counting-MAE on tomatOD goes from 3.29 to 1.34. The mAP cells
are unaffected.

Write-up: [`cross_dataset/cross_dataset.ipynb`](cross_dataset/cross_dataset.ipynb).

Two defects found by converting rather than reading: `tomatOD_test.json` declares
a single category whose name is a list of all three, while its annotations use
three ids (reading categories per split silently collapses the test set onto one
class); and the shipped annotations hold 2 421 boxes against the 2 418 its README
advertises.

## 4. Occlusion

Sections 1-3 measure fruit an annotator could see whole; the app is pointed at
fruit behind leaves. `occlusion/` hides a known fraction of every annotated box -
with a patch cropped from the same frame's background, so lighting and white
balance carry over - and re-measures. Ground truth stays the whole fruit, so
recall answers "did it find the fruit" rather than "did it find what is left".

`laboro3` on Laboro's own val split, conf 0.25, matching at IoU 0.5:

| hidden | recall | stage accuracy | counting-MAE |
|---|:--:|:--:|:--:|
| 0% | 0.884 | 0.944 | 1.25 |
| 20% | 0.778 | 0.940 | 1.33 |
| 40% | 0.571 | 0.920 | 1.77 |
| 60% | 0.160 | 0.900 | 2.43 |
| 70% | 0.057 | 0.860 | 2.99 |

**Detection and staging fail independently.** Recall loses 94% of its value
across the sweep while stage accuracy loses 9%: the model stops *finding* fruit
long before it starts misreading ripeness, and counting-MAE - the metric this
work is about - is driven almost entirely by the first. A single error number
cannot say which of the two produced it, the same objection section 3 raised
about domain shift. The 0% row is a check rather than a result: counting-MAE 1.25
is section 3's laboro3/Laboro cell, reproduced through different code.

### Half of the collapse is bookkeeping

Recall falls off a shelf between 40% and 50% hidden, which is where the metric
turns over rather than the model. A detector that boxes the visible part scores
IoU about `1 - fraction` against a whole-fruit ground truth, and at 50% that sits
exactly on the matching threshold. The identical sweep matched at IoU 0.3:

| hidden | recall @ IoU 0.5 | recall @ IoU 0.3 | counting-MAE |
|---|:--:|:--:|:--:|
| 30% | 0.694 | 0.726 | 1.49 |
| 40% | 0.571 | 0.642 | 1.77 |
| 50% | 0.312 | 0.516 | 2.06 |
| 60% | 0.160 | 0.344 | 2.43 |

Counting-MAE is identical in both runs, as it must be - counts do not depend on
how detections are matched - which makes the pair a clean decomposition of "not
found" from "found, boxed what was left".

### Which half is hidden decides the stage

A half-ripened fruit is red on one side and green on the other. `hide_red` and
`hide_green` cover the half chosen by median CIELAB a\*, the axis section 3
measured the stage boundaries on. Paired - the same fruit scored under both, only
fruit found in both counted, 40% hidden:

| GT stage | n | mean stage, red hidden | green hidden | shift | changed call |
|---|:--:|:--:|:--:|:--:|:--:|
| green | 493 | 0.010 | 0.018 | +0.008 | 1.8% |
| half | 156 | 1.122 | 1.205 | **+0.083** | **17.3%** |
| fully | 187 | 1.941 | 1.957 | +0.016 | 4.8% |

(stage scale: green 0, half 1, fully 2)

**Which side is visible decides the stage of a half-ripened fruit and almost
nothing else.** 17% of them change their call between the two conditions against
2% of green and 5% of fully ripened ones - the classes that look the same from
either side - and among those that move, 20 read riper with the green side
covered against 7 the other way (McNemar p = 0.019). The mean shift is small
because most fruit do not move at all; the effect is concentrated in exactly the
class the two datasets disagree about.

### The occluder is part of the measurement

Two controls, same geometry. A flat black patch instead of a background crop:

| hidden | recall, background patch | recall, black | MAE, patch | MAE, black |
|---|:--:|:--:|:--:|:--:|
| 20% | 0.778 | 0.709 | 1.33 | 1.43 |
| 30% | 0.694 | 0.609 | 1.49 | 1.72 |
| 40% | 0.571 | 0.461 | 1.77 | 1.88 |

Nothing about the fruit differs between the two runs, only what covers it, so
that gap is the model reacting to the occluder rather than to the missing pixels
- worth about ten points of recall, more than the spread between several models
in section 3's matrix. The other control moves nothing: the same sweep on
`cross_mix`, the checkpoint the app ships, stays within 0.03 recall of `laboro3`
at every level, so the curve describes the setup rather than one checkpoint.

And the occlusion that was already in the data - fruit overlapped by other fruit,
untouched frames, coverage measured from the annotations:

| covered by other fruit | n | recall | stage accuracy |
|---|:--:|:--:|:--:|
| none | 423 | 0.891 | 0.947 |
| 5-15% | 377 | 0.912 | 0.968 |
| 15-30% | 503 | 0.913 | 0.939 |
| 30-50% | 357 | 0.888 | 0.934 |
| >50% | 187 | 0.701 | 0.908 |

Real overlap barely costs anything until it passes half the box, where the pasted
patch has already taken a third of the detections. Two reasons, not exclusive:
box overlap overstates occlusion for round fruit (two boxes can share a third of
their area while neither fruit hides any of the other), and a pasted rectangle is
out of distribution in a way fruit-behind-fruit is not. Either way, **a synthetic
occlusion number is a property of the occluder as much as of the model**, and
which occluder was used is rarely reported.

Write-up: [`occlusion/occlusion.ipynb`](occlusion/occlusion.ipynb).

## 5. The same model on a laptop and on a phone

The app ships `cross_mix` as a 16-bit CoreML package with NMS baked in
(`export_coreml.py`), which changes three things at once: precision, where NMS
runs, and who letterboxes the image. `runtime_bench/bench.py` runs one checkpoint
through every laptop-side runtime over the same 161 images and scores them with
the same code.

| runtime | ms/frame | fps | recall | stage accuracy | counting-MAE |
|---|:--:|:--:|:--:|:--:|:--:|
| torch, MPS | 28.1 | 36 | 0.875 | 0.947 | 1.04 |
| torch, CPU | 55.0 | 18 | 0.875 | 0.947 | 1.04 |
| CoreML, CPU only | 16.4 | 61 | 0.871 | 0.948 | 1.03 |
| CoreML, CPU + Neural Engine | **6.0** | **166** | 0.873 | 0.949 | 1.03 |
| CoreML, all units | 6.0 | 167 | 0.873 | 0.949 | 1.03 |

- **The export costs nothing measurable.** fp16 and a baked NMS move recall by
  0.002 and counting-MAE by 0.01, with per-class totals within 1%.
- **It is also 4.7x faster than the framework it was trained in.** The Neural
  Engine path runs the same model in 6 ms. The app currently forces `.cpuOnly`
  because both GPU paths crashed on device with an `MLIR pass manager failed`
  assertion; on this Mac that setting costs 2.7x, which is now a measured reason
  to retry rather than a guess.
- **Vision's letterbox padding does not matter.** ultralytics pads with grey 114,
  Vision's `.scaleFit` pads with black; re-running with black padding gives
  counting-MAE 1.04 against 1.03. A hypothesis closed rather than confirmed.
- **Section 3 reported 1.14 for this model, not 1.03.** That number used
  ultralytics' default NMS IoU of 0.7 while the export bakes 0.45; re-running at
  0.7 reproduces 1.14 exactly. The shipped threshold happens to count slightly
  better than the evaluation default.

### The phone half

iPhone18,1 on iOS 26.5.2, Release build, thermal state nominal, ten repeats over
the three bundled samples, collected by the app's Benchmark screen:

| runtime | ms/frame | fps | model load |
|---|:--:|:--:|:--:|
| phone, CPU only | 11.7 | 86 | 36 ms |
| phone, CPU + Neural Engine | **3.8** | **262** | 600 ms |
| laptop, CoreML CPU only | 16.4 | 61 | - |
| laptop, CoreML CPU + Neural Engine | 6.0 | 166 | - |

**The phone is faster than the laptop on both paths**, and its timing is the more
inclusive of the two - it covers Vision's own scaling, while the laptop figure
times only the CoreML call. On-device inference is not the compromise half of
this system; at 262 fps the detector is nowhere near the camera's frame rate, and
the app's forced `.cpuOnly` is costing 3.1x for nothing. Loading onto the Neural
Engine costs 600 ms against 36 ms, which belongs in app startup rather than in
the per-frame budget.

The two machines agree on what they see: 30 of 30 detections matched across the
three images, mean IoU 0.992 to 0.997, every label identical
(`runtime_bench/compare_device.py`).

Two things this run cannot say, and one it can:

- The bundled samples are 640x640, so letterboxing is a no-op on them. The
  padding question stays answered only by the laptop-side run above.
- Ten repeats on three images is a latency number, not a throughput-under-load
  number; a camera session sustains inference for minutes and will throttle.
- Compute units are not bit-identical. Between `cpuOnly` and
  `cpuAndNeuralEngine` on the same phone, confidences move by up to 0.098, and
  one borderline box flips from `fully_ripened` (0.522) to `half_ripened`
  (0.506). One box out of thirty, but it is a count changing because of where
  the model ran.

How the phone half is collected:
[`runtime_bench/README.md`](runtime_bench/README.md).

## 6. Which augmentations this task needs, and a threshold that was doing the work

The protocol is frozen at section 3's - laboro3, YOLO11n detect, 60 epochs, imgsz
640, batch 16, seed 0 - and one augmentation group changes per run. `default` is
section 3's own laboro3 run. Two of the six configs in `augmentation/train_aug.py`
have been trained so far.

Ripeness is a colour judgement and the stock recipe jitters colour hard
(`hsv_s 0.7`), so `no_colour` is the group this task has most reason to suspect.
`occlusion_aug` adds cutmix and mixup, which paste foreign pixels over the fruit -
section 4's occluder, applied at training time - so if that transfers, it should
show under occlusion rather than on clean frames.

recall / counting-MAE at the fixed conf 0.25 the earlier sections used:

| run | mAP50-95 | Laboro | tomatOD | 30% hidden | 50% hidden |
|---|:--:|:--:|:--:|:--:|:--:|
| default | 0.698 | 0.884 / 1.25 | 0.981 / 3.29 | 0.694 / 1.49 | 0.312 / 2.06 |
| no_colour | **0.706** | 0.873 / **1.08** | 0.746 / **1.50** | 0.656 / 1.44 | 0.228 / 2.03 |
| occlusion_aug | 0.701 | 0.865 / 1.19 | 0.976 / 3.01 | 0.671 / 1.51 | 0.283 / 2.01 |

Read at face value this says colour jitter is expensive: dropping it halves the
transfer counting error. That reading is wrong, and finding out why is the more
useful result.

### The fixed threshold was doing the work

On tomatOD the default model predicts 1.92x as many fruit as exist while
`no_colour` predicts 0.93x, and their stage accuracy is the same to within
0.006 - so the gap is how freely each model fires, not what it understands.
Confidence is the knob for that, and it had been held at 0.25 since
`evaluate.py`. Sweeping it (`conf_sweep.py`):

| model | best counting-MAE on tomatOD | at conf | recall there | predicted/actual |
|---|:--:|:--:|:--:|:--:|
| default | **1.34** | 0.70 | 0.799 | 0.88x |
| no_colour | 1.50 | 0.25 | 0.746 | 0.93x |
| occlusion_aug | 1.32 | 0.70 | 0.776 | 0.85x |

Tuned, the ordering reverses: the colour-jittered model counts *better* on the
unfamiliar domain, and keeps more recall doing it. `no_colour` was never better
at counting - it was merely calibrated closer to where the threshold happened to
sit. In-domain a smaller advantage survives the sweep (0.93 against 1.03 at each
model's own best threshold), consistent with the +0.008 mAP, but that is one seed
on one dataset and is not worth a claim yet.

**This applies backwards to section 3.** Its transfer cells were counted at
conf 0.25 too, and the threshold alone takes laboro3's tomatOD counting-MAE from
3.29 to 1.34 - more than half of the "counting error triples under transfer"
penalty is the operating point rather than the domain shift. The mAP half of that
section is unaffected: mAP integrates over confidence, which is precisely why the
two metrics disagreed. **Counting-MAE at a fixed threshold compares calibration,
not models**, and every counting comparison here now needs a sweep next to it.

That is the third time this section has run into the same shape of problem: a
protocol that looks like it measures the model measures the harness instead -
the occluder in section 4, the confidence threshold here, the two regimes of
label noise in section 3.

### What cutmix and mixup bought

Nothing where it was supposed to. Under 30% occlusion `occlusion_aug` gets recall
0.671 against the default's 0.694, and 0.283 against 0.312 at 50%. Pasting
rectangles of other images over the fruit at training time does not transfer to
having rectangles pasted over it at test time, even though the two operations are
nearly identical. Its small win on tuned tomatOD counting (1.32 vs 1.34) is well
inside one-seed noise.

Four configs remain: `no_mosaic`, `no_geometry`, `none`, `heavy`.

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

# occlusion - no training, runs on the section 3 weights
bash occlusion/run_all.sh

# laptop side of the runtime comparison; the phone side is runtime_bench/README.md
python runtime_bench/bench.py --data laboro3
python runtime_bench/bench.py --data laboro3 --pad 0

# augmentation ablation - one 60-epoch run per config, ~1 h each
python augmentation/train_aug.py --all && python augmentation/eval_aug.py

# counting-MAE across confidence thresholds - run this next to any counting comparison
python conf_sweep.py --models cross_laboro3 aug_no_colour --data tomatod
```

Datasets, weights and runs are gitignored. Laboro Tomato and tomatOD are both
CC BY-NC-SA 4.0 and are not redistributed here, with one exception: the three
demo frames bundled with the iOS app
(`ios/TomatoRipeness/Resources/samples/`) are Laboro Tomato images, shared under
the same CC BY-NC-SA 4.0 licence and credited to
[laboroai/LaboroTomato](https://github.com/laboroai/LaboroTomato). The CoreML
model shipped in the app is trained on Laboro Tomato and tomatOD and inherits
the non-commercial terms.

## Next

- Their colour-analysis step, to get the per-stage R2 (0.809 / 0.897 / 0.968). Counting is the axis of this work, not mAP.
- Finish the augmentation ablation: `no_mosaic`, `no_geometry`, `none` and `heavy` are written but not trained. The two that ran say less about augmentation than about thresholds.
- Re-count section 3's matrix with a confidence sweep per cell, now that a fixed threshold is known to carry more than half the transfer penalty. Same for the occlusion sweep in section 4, whose counting column is also at conf 0.25.
- Repeat the ablation on a second seed. Every difference it found in-domain is inside the range one seed could produce.
- Field test: `field_test/` takes phone photos and hand counts and reports counting-MAE against them. Nothing here has been measured outside two research greenhouses.
- Occlusion on tomatOD, and coverage recomputed from Laboro's instance masks rather than from box overlap, which overstates it for round fruit.
- A third annotation source (KUTomaData) - two datasets give one pairwise comparison, which cannot separate "schemas generally disagree" from "one of these two is idiosyncratic". Three enable leave-one-out. Camacho's data does not count: it is Laboro re-exported through Roboflow, not independent annotation.
- Own contribution: counting error under label noise, now sharpened by section 3 - separate noise that is recoverable from the image (a domain the model can identify) from noise that is not. The usual protocol, flipping a fixed percentage of labels at random, conflates the two. Section 4 says the same thing about occlusion protocols.
