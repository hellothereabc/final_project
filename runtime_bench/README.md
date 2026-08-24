# Phone half of the runtime comparison

`bench.py` measures the exported model on the laptop across CoreML compute
units. The phone cannot be driven from here, so its half is collected by hand
and merged afterwards.

1. Open `ios/TomatoRipeness` in Xcode and Run on the device (not the simulator -
   simulator inference runs on the Mac's CPU and would just re-measure the
   laptop).
2. Tap the speedometer button, leave **Include GPU paths** off for the first run
   (`.cpuAndGPU` and `.all` have both crashed on this model with an `MLIR pass
   manager failed` assertion inside `handler.perform`, which traps rather than
   throwing), set repeats, Start.
3. Copy JSON, put it in a file, then:

```bash
python runtime_bench/bench.py --data laboro3          # laptop, training-style padding
python runtime_bench/bench.py --data laboro3 --pad 0  # laptop, the padding Vision uses
python runtime_bench/compare_device.py phone.json --pad 0
```

`compare_device.py` re-runs the same `.mlpackage` here on the same bundled sample
images and lines up latency, per-class counts and box IoU against what the phone
reported.

Two things worth watching in the result:

- **Which padding matches.** The app asks Vision for `.scaleFit`, which pads with
  black; ultralytics trains and validates with grey 114. If the phone's boxes
  line up with `--pad 0` and not with `--pad 114`, the app is running the model
  under conditions it was never evaluated under, and the fix is in the app, not
  in the weights.
- **Thermal state.** Reported per config in the JSON. A phone that has warmed up
  throttles, so a latency taken at `serious` is not comparable with one taken at
  `nominal` - re-run cold if they differ.

## Collected so far

`phone_iPhone18-1.json` - iPhone18,1, iOS 26.5.2, Release build, thermal state
nominal, 10 repeats. Kept in the repository rather than under `runs/` because a
device measurement cannot be reproduced by re-running a script here.

```bash
python runtime_bench/compare_device.py runtime_bench/phone_iPhone18-1.json --pad 114
```

Run the laptop side on an otherwise idle machine. A training job in the
background inflates the CoreML figure by roughly 50% - the first comparison run
reported 25.2 ms for CPU-only against 16.4 ms measured idle.
