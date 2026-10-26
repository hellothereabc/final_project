# Tomato ripeness - progress for 22 09 2026

Two questions from the 19 September meeting, both answered and measured:
*"try bigger models until it stops working"* and *"data augmentation using
HSV/HSL to simulate light conditions"*.

Two figures, one per question: **`01_models.png`** and **`02_lighting.png`**.
Everything else is in `extra/`, for questions.

Everything below is the six-class Laboro Tomato protocol, 643 train / 161 val
images, 60 epochs, image size 640, seed 0, **one seed per model**: treat
differences under ~0.02 recall or ~0.01 mAP as noise. Latency is inference-only
Core ML (FP16, embedded NMS) on the same three images - no camera capture or
rendering. Mac is a Mac16,8; the phone is an iPhone 17 Pro on iOS 27.0.

## 1. How far up the size ladder is it worth going?  -  figure 01

YOLO11n, s, m, l, x trained under one protocol, exported to Core ML, swept over
confidence, and benchmarked on the Mac and on the phone.

| Model | mAP50-95 | Best counting MAE | Best fully-ripe MAE | Core ML FP16 | iPhone NE load |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 0.706 | 0.963 at 0.40 | 0.590 at 0.60 | 5.4 MB | 0.71 s |
| YOLO11s | 0.747 | 0.919 at 0.50 | 0.640 at 0.60 | 19.1 MB | 1.40 s |
| YOLO11m | 0.782 | 0.882 at 0.50 | 0.627 at 0.60 | 40.3 MB | 1.58 s |
| YOLO11l | 0.780 | 0.859 at 0.60 | 0.658 at 0.40 | 51.0 MB | 2.16 s |
| YOLO11x | 0.774 | 0.874 at 0.60 | 0.602 at 0.60 | 114.1 MB | 2.73 s |

- **Accuracy stops at m.** 0.706 -> 0.747 -> 0.782, then l is 0.002 below and x
  0.008 below m: inside one-seed noise.
- **The fully-ripe count never improves.** n's 0.590 is the best on the ladder;
  every larger model lands between 0.602 and 0.658. The class this work is
  about gains nothing from capacity on this split.
- **Cost keeps growing after the accuracy stops.** m -> x is 2.8x the package,
  2.0x the Neural Engine latency and 1.7x the load time, for no mAP.
- **Where it stops working**, two different answers: accuracy stops improving at
  **m**; real time on the phone stops at **x**, which has no margin left -
  33.8 FPS over a sustained 4,500-frame run, p90 31.8 ms against a 33.3 ms
  budget, and the largest thermal drift on the ladder (+8.8%).

On the phone's CPU rather than its Neural Engine the ladder breaks three sizes
earlier: s is the last size above 30 FPS (38.3), m is at 20.4, x at 8.9. The
laptop is no substitute for measuring on the phone, but it is close: the phone
trails the Mac by a flat ~1.5x on CPU, while on the Neural Engine the gap
closes as the model grows - 3.85x at n down to 1.21x at x - so a small model
looks far worse on the phone than the laptop suggests. Both devices see the
same fruit: every matched box has IoU 0.99-1.00 and the same stage label.
Full tables in `extra/`, `device_tables.md`.

## 2. Does HSV augmentation buy robustness to lighting?  -  figure 02

The test perturbation is deliberately *not* HSV jitter, so no model is scored on
its own augmentation: exposure in stops and a red/blue white-balance shift,
both applied in linear light, labels untouched.

Each cell is recall at conf 0.25 / counting MAE at the best threshold.

Each cell is recall at conf 0.25 / counting MAE at the best threshold.

| Model | clean mAP50-95 | clean | -3 stops | +3 stops | wb -1.5 (cool) | wb +1.5 (warm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n (stock HSV) | 0.706 | 0.877 / 0.96 | 0.860 / 1.13 | 0.696 / 1.78 | 0.797 / 1.31 | 0.817 / 1.35 |
| n, no colour jitter | 0.718 | 0.877 / 1.00 | 0.765 / 1.47 | 0.608 / 2.07 | 0.751 / 1.52 | 0.782 / 1.71 |
| n, hsv_v08 (brightness x2) | 0.699 | 0.874 / 1.01 | 0.870 / 1.04 | 0.692 / 1.74 | 0.810 / 1.19 | 0.820 / 1.35 |
| n, hsv_wide (+ hue) | 0.684 | 0.871 / 1.05 | 0.868 / 1.07 | 0.704 / 1.82 | 0.824 / 1.09 | 0.847 / 1.30 |
| YOLO11s | 0.747 | 0.916 / 0.92 | 0.904 / 0.97 | 0.739 / 1.70 | 0.842 / 1.12 | 0.883 / 1.12 |
| YOLO11m | 0.782 | 0.935 / 0.88 | 0.901 / 1.05 | 0.786 / 1.68 | 0.870 / 1.11 | 0.898 / 1.27 |
| YOLO11l | 0.780 | 0.929 / 0.86 | 0.912 / 0.98 | 0.782 / 1.75 | 0.877 / 1.26 | 0.901 / 1.04 |
| YOLO11x | 0.774 | 0.935 / 0.87 | 0.914 / 0.93 | 0.730 / 1.81 | 0.865 / 1.23 | 0.899 / 1.06 |

- **The stock HSV jitter is what buys lighting robustness.** Turning it off is a
  wash on clean frames - it even has the best clean mAP - and loses about 10
  recall points at every extreme.
- **Widening it beyond stock buys nothing.** hsv_v08 sits on the baseline
  everywhere; hsv_wide gains ~0.03 at the white-balance extremes and pays 0.02
  clean mAP.
- **Under-exposure is nearly free down to -3 stops; over-exposure breaks
  everyone**, big models included: clipping removes the very saturation the
  ripeness stage is read from - the +3 stops frame in figure 02 shows it.
- A **cool** cast costs more than a warm one; a **warm** cast costs counting
  more than detection, because it pushes green fruit toward "riper".

Answer to the question: the augmentation asked for is already in the default
recipe and is doing its job. It should be kept, not strengthened.

## Caveats to state out loud

- One seed per model; no confidence intervals.
- Laboro Tomato only, six classes. The cross-dataset (tomatOD) models are not
  part of this comparison.
- The phone moved from iOS 26.5.2 to 27.0 between the two measurement days:
  YOLO11n's Neural Engine latency went 4.05 -> 6.4-6.8 ms (repeated, not noise)
  while m was unchanged. The YOLO11m sustained run is still from 26.5.2.
- Power consumption is not reported: the Instruments runs on 14 September ended
  unreliably, so no defensible number exists.

## If a question needs more (`extra/`)

| File | What it shows |
|---|---|
| `size_accuracy.png` | mAP and tuned counting MAE per size, next to package size and phone load time |
| `accuracy_vs_latency.png` | Figure 01 with the Mac alongside the phone and the sustained runs marked |
| `device_latency.png` | Mac CPU / Mac NE / iPhone CPU / iPhone NE per size, log axis, 33.3 ms budget, ratios |
| `lighting_examples.jpg` | All five exposure and five white-balance levels of the sample frame |
| `lighting_sizes.png` | Recall and counting MAE along both lighting axes for n-x |
| `lighting_hsv.png` | The same for all three HSV training variants |

Sources: every number is generated from saved result files by
`runtime_bench/make_device_tables.py`, `lighting/make_tables.py`,
`paper/make_talk_figures.py` and `paper/make_size_lighting_figures.py`; the full
tables are `runtime_bench/device_tables.md` and `lighting/results.md`, the
written record is `runtime_bench/model_size_results.md`.
