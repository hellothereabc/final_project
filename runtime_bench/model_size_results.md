# YOLO11n versus YOLO11m on iPhone

Both models were trained for 60 epochs on the same 643/161-image Laboro Tomato
split and exported to FP16 Core ML with embedded NMS. Phone measurements used
the same three bundled images, a Release build, and an iPhone18,1 running iOS
26.5.2. FPS is `1000 / median latency` and measures model inference rather than
the complete camera pipeline.

| Model | Parameters | GFLOPs | mAP50-95 | Best overall counting MAE | Core ML size | iPhone CPU | iPhone CPU + Neural Engine |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 2.58 M | 6.3 | 0.706 | 0.963 at conf. 0.40 | 5.4 MB | 12.43 ms / 80.5 FPS | 4.05 ms / 246.7 FPS |
| YOLO11m | 20.03 M | 67.7 | 0.782 | 0.882 at conf. 0.50 | 40.3 MB | 48.20 ms / 20.7 FPS | 13.47 ms / 74.2 FPS |

## The full size ladder: n, s, m, l, x

YOLO11s, l, and x were trained on 20 September 2026 under the same protocol
(60 epochs, 640, seed 0, MPS, `nbs=64`), with physical batch 16 / 8 / 4 so
that each fits the 24 GB of unified memory (s peaked at 9.5 GB, l at 10.7 GB).
Wall time: s 1 h 00 min, l 1 h 57 min, x 3 h 17 min. Confidence sweeps are in
`runs/laboro_model_size_thresholds_nsmlx.json` (`size_thresholds.py`); the n
and m rows reproduce the earlier sweep exactly. Mac figures are from
`mac_sanity.py` (Mac16,8, three bundled images, 10 repeats after 3 warm-up
passes, Vision-style black padding) and are stored in
`runs/runtime/mac_model_sizes.json`.

| Model | Parameters | GFLOPs | mAP50-95 | Best overall MAE | Best fully-ripe MAE | Core ML size | Mac CPU | Mac CPU + NE | NE load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 2.58 M | 6.3 | 0.706 | 0.963 at conf. 0.40 | 0.590 at conf. 0.60 | 5.4 MB | 9.08 ms / 110 FPS | 1.76 ms / 570 FPS | 656 ms |
| YOLO11s | 9.43 M | 21.3 | 0.747 | 0.919 at conf. 0.50 | 0.640 at conf. 0.60 | 19.1 MB | 16.95 ms / 59 FPS | 3.25 ms / 307 FPS | 846 ms |
| YOLO11m | 20.03 M | 67.7 | 0.782 | 0.882 at conf. 0.50 | 0.627 at conf. 0.60 | 40.3 MB | 32.41 ms / 31 FPS | 9.48 ms / 106 FPS | 1237 ms |
| YOLO11l | 25.32 M | 86.6 | 0.780 | 0.859 at conf. 0.60 | 0.658 at conf. 0.40 | 51.0 MB | 42.19 ms / 24 FPS | 11.11 ms / 90 FPS | 1683 ms |
| YOLO11x | 56.88 M | 194.4 | 0.774 | 0.874 at conf. 0.60 | 0.602 at conf. 0.60 | 114.1 MB | 69.05 ms / 14 FPS | 22.51 ms / 44 FPS | 2109 ms |

Three things the ladder says, all on one seed:

- **Accuracy stops at m.** mAP50-95 rises 0.706 -> 0.747 -> 0.782 and then
  stays there: l is 0.002 below m and x 0.008 below, both inside single-seed
  noise. Tuned overall counting keeps creeping down to l (0.859) but the
  ordering among m, l, and x is not something one seed can establish.
- **The fully-ripe count never improves.** n's 0.590 is the best value on the
  ladder; every larger model lands between 0.602 and 0.658. The class this
  work is about does not benefit from capacity at all on this split.
- **Cost keeps growing after the accuracy stops.** From m to x the package
  grows 2.8x, Neural Engine latency 2.4x, and load time 1.7x, for no mAP.

### The same ladder on the phone

Measured on 20 September 2026 on the same iPhone18,1, now on iOS 27.0
(Build 24A437), with `phone_bench.sh`: Release build, three bundled images,
10 repeats per compute-unit setting, phone unattended. Raw reports are
`phone_iPhone18-1_ios27_yolo11{n,s,m,l,x}_laboro.json`. n and m were re-run
on this OS so the whole ladder is one build on one day.

| Model | iPhone CPU only | CPU p90 | iPhone CPU + Neural Engine | NE p90 | NE load |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 13.59 ms / 73.6 FPS | 17.22 ms | 6.76 ms / 147.9 FPS | 7.91 ms | 715 ms |
| YOLO11s | 26.09 ms / 38.3 FPS | 37.32 ms | 8.20 ms / 122.0 FPS | 10.52 ms | 1404 ms |
| YOLO11m | 48.98 ms / 20.4 FPS | 61.45 ms | 13.72 ms / 72.9 FPS | 15.32 ms | 1582 ms |
| YOLO11l | 59.01 ms / 16.9 FPS | 70.49 ms | 15.22 ms / 65.7 FPS | 16.68 ms | 2159 ms |
| YOLO11x | 111.98 ms / 8.9 FPS | 123.38 ms | 27.21 ms / 36.7 FPS | 28.30 ms | 2727 ms |

Sustained Neural Engine runs, 4,500 frames each
(`phone_iPhone18-1_ios27_yolo11{x,l}_sustained.json`):

| Model | Median | p90 | vs. short run | Thermal state at end |
|---|---:|---:|---:|---|
| YOLO11l | 15.63 ms / 64.0 FPS | 17.69 ms | +2.7% | fair |
| YOLO11x | 29.60 ms / 33.8 FPS | 31.81 ms | +8.8% | nominal |

The phone and the Mac agree on what they see at every size
(`compare_device.py --pad 0 --package ...`, `runs/logs/phone_compare_ios27.log`):
every matched box has mean IoU 0.990-1.000 and an identical stage label. Two
borderline boxes differ in count - n finds 28 where the Mac finds 29, l finds
27 where the Mac finds 26 - the same one-box-out-of-thirty the earlier
comparison saw.

**Where it stops working.** Against the three limits:

- *30 FPS camera budget on the Neural Engine.* Everything up to l clears it with
  room (l: 65 FPS short, 64 sustained). x is the first model to touch it:
  36.7 FPS cold, 33.8 FPS sustained, p90 31.8 ms against a 33.3 ms budget. It
  works, but with no margin at all, and the 8.8% sustained drift is the largest
  on the ladder. One more step up - or a warmer phone - and it would not.
- *CPU fallback.* The app shipped with `.cpuOnly` forced. On CPU the ladder
  breaks at m: s is the last size above 30 FPS (38.3), m is at 20.4, x at 8.9.
- *Load and storage.* Neural Engine load grows from 0.7 s (n) to 2.7 s (x), and
  the package from 5 MB to 114 MB. Neither breaks anything; both are the cost of
  capacity that section above showed buys no accuracy past m.

So the answer to "try bigger models until it stops working" is two different
sizes depending on what "working" means: accuracy stops improving at **m**,
and real-time on the phone's Neural Engine stops at **x**. The two sizes in
between, l and x, cost 1.1x and 2.0x m's latency for no mAP and no better
fully-ripe count.

**iOS 27 versus 26.5.2.** n's Neural Engine latency is 6.76 ms here against
4.05 ms measured on 26.5.2 on 14 September (repeat run on 27.0: 6.37 ms, so it
is not noise), while m is unchanged (13.72 vs 13.47 ms) and CPU-only n is also
unchanged (13.59 vs 12.43 ms). The small graph got slower on the Neural Engine
across the OS update and the large one did not, which reads as a fixed
per-inference overhead that only the smallest model notices. The raw 26.5.2
reports for n and m were lost to an overwrite during this session; their
summary figures survive in the table at the top of this file and in
`SESSION_STATUS.md`.

## Mac versus iPhone

| Model | Compute units | Mac median / FPS | iPhone median / FPS | iPhone slowdown |
|---|---|---:|---:|---:|
| YOLO11n | CPU only | 10.01 ms / 99.9 FPS | 12.43 ms / 80.5 FPS | 1.24x |
| YOLO11m | CPU only | 34.31 ms / 29.1 FPS | 48.20 ms / 20.7 FPS | 1.40x |
| YOLO11n | CPU + Neural Engine | 2.05 ms / 487.1 FPS | 4.05 ms / 246.7 FPS | 1.98x |
| YOLO11m | CPU + Neural Engine | 9.85 ms / 101.5 FPS | 13.47 ms / 74.2 FPS | 1.37x |

These are inference-only Core ML measurements on the same three bundled
images, using FP16 exports and 10 repeats. They do not include camera capture,
display, or application rendering time.

YOLO11m gains 0.076 mAP50-95 and reduces tuned overall counting MAE by 0.081.
The cost is a 7.5x larger Core ML package and 3.32x higher Neural Engine
latency. It still exceeds a 30 FPS camera budget. For fully ripe fruit alone,
the larger model does not improve the tuned error: its best MAE is 0.627,
compared with 0.590 for YOLO11n, both at confidence 0.60.

The short phone benchmark used 30 timed frames per compute setting. YOLO11n
had p90 latencies of 13.57 ms on CPU and 4.64 ms on CPU + Neural Engine;
YOLO11m had 53.59 ms and 13.99 ms respectively. Cross-device output checks
matched 28/28 YOLO11n detections (mean box IoU 0.9963) and 29/29 YOLO11m
detections (mean box IoU 0.9931), with full maturity-stage agreement.

A sustained YOLO11m Neural Engine run processed 4,500 frames. Median latency
was 13.86 ms (72.2 FPS), p90 was 15.77 ms, and the thermal state remained
`fair`. The 2.9% increase from the short-run median gives no evidence of
material throttling during this run.

Power consumption is not reported. Apple Instruments could launch the app but
the automated Power Profiler runs ended unreliably, so no defensible power
number was produced. The latency, thermal, accuracy, and package-size results
remain valid independently of that failed measurement.

Raw phone reports:

- `phone_iPhone18-1_yolo11n_laboro.json`
- `phone_iPhone18-1_yolo11m_laboro.json`
- `phone_iPhone18-1_yolo11m_sustained.json`

## Generated tables and figures

The tables above are the written record; the same numbers are regenerated from
the saved measurements by `python runtime_bench/make_device_tables.py` into
`device_tables.md` (accuracy and cost, Mac against iPhone at every size, and
each size against the 30 FPS budget).

Figures, from `python paper/make_size_lighting_figures.py` into `paper/figures/`:

- `size_accuracy.pdf/png` - mAP50-95 and tuned counting MAE per size next to
  package size and Neural Engine load time.
- `device_latency.pdf/png` - Mac CPU, Mac NE, iPhone CPU, iPhone NE per size on
  a log axis against the 33.3 ms frame budget, plus the phone/Mac ratios.
- `accuracy_vs_latency.pdf/png` - accuracy against Neural Engine latency on both
  devices, marker area the Core ML package, with the sustained runs marked.
