# Larger-model phone experiment

## Research question

Does moving from YOLO11n to YOLO11m improve ripeness detection and
counting enough to justify its latency, storage, and power cost on the target
iPhone?

YOLO11m was selected as a clearly non-nano comparison. The exported models have
2.58M parameters and 6.3 GFLOPs for YOLO11n versus 20.03M parameters and 67.7
GFLOPs for YOLO11m. This creates a strong accuracy/resource contrast on the
phone rather than testing two nearby compact variants.

## Controlled protocol

Keep the data and training protocol fixed as closely as the hardware permits:

- training data: Laboro Tomato's native 643/161 split;
- task: six-class detection, with fruit size collapsed into three ripeness
  stages for counting;
- image size: 640;
- epochs: 60;
- physical batch: 16 for YOLO11n and 8 for YOLO11m; both retain Ultralytics'
  `nbs=64` effective nominal batch. Batch 16 made YOLO11m consume about 15.4 GB
  and thrash unified memory, while batch 8 used about 8.6 GB;
- seed: 0;
- patience: 20; both runs completed all 60 epochs;
- export: Core ML FP16 with NMS, confidence 0.25, IoU 0.45;
- phone: the same iPhone18,1, Release build, iOS version recorded;
- compute units: `cpuOnly` and `cpuAndNeuralEngine`;
- benchmark images and repetitions: the same three images and 10 repetitions,
  plus a sustained 4,500-frame YOLO11m run for thermal behaviour.

Train and export under distinct run names so existing results are not
overwritten.

Suggested commands after restoring the datasets and environment:

```bash
/Users/chel/Projects/master_diploma/reproduce/.venv/bin/python train.py \
  --model yolo11m.pt --name laboro_yolo11m_exiffix_b8_mps --batch 8

/Users/chel/Projects/master_diploma/reproduce/.venv/bin/python export_coreml.py \
  --weights runs/laboro_yolo11m_exiffix_b8_mps/weights/best.pt \
  --half --out ios_model/yolo11m_fp16
```

## Measurements

For each model and compute-unit setting, report:

- mAP50--95 on the Laboro validation set;
- best counting-MAE after a confidence sweep on the Laboro validation set;
- model package size and load time;
- median and p90 inference latency, and FPS as `1000 / median_ms`;
- per-class count agreement between PyTorch, Mac Core ML, and iPhone Core ML;
- peak memory if available from Xcode;
- initial and final thermal state during a fixed-duration run.

## Phone results

On iPhone18,1 with iOS 26.5.2, YOLO11n ran at a median 4.05 ms (246.7 FPS) on
CPU + Neural Engine, while YOLO11m ran at 13.47 ms (74.2 FPS). CPU-only
latencies were 12.43 ms and 48.20 ms respectively. The package grew from 5.4
MB to 40.3 MB. A sustained 4,500-frame YOLO11m run reached a median 13.86 ms
(72.2 FPS), p90 15.77 ms, and remained in the `fair` thermal state.

Power consumption is omitted because automated Instruments Power Profiler runs
ended unreliably and did not yield a defensible measurement.

## Decision rule

YOLO11m improves mAP50--95 from 0.706 to 0.782 and tuned overall counting MAE
from 0.963 to 0.882, while retaining about 74 FPS on the target phone. It does
not improve the best fully-ripe-only MAE and costs 3.32x Neural Engine latency
and 7.5x package size. It is therefore viable when overall detection quality is
the priority; YOLO11n remains preferable when storage, load time, or maximum
throughput matters.
