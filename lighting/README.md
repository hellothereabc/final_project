# Lighting

Two questions, both about the light rather than the fruit.

1. **How far does a change of light move the detector?** `run_lighting.py`
   re-scores the same weights over the Laboro val split (161 images, six-class
   protocol) with one photometric knob turned per run - exposure in stops and a
   red/blue white-balance shift, both applied in linear light (`relight.py`).
   Labels are untouched. Each level reports class-agnostic recall, stage
   accuracy among found fruit, and counting-MAE at conf 0.25 *and* at the best
   threshold, because exposure moves confidences wholesale and a fixed threshold
   would measure calibration.
2. **Does wider HSV jitter at training time buy robustness to it?**
   `train_light.py` trains three YOLO11n variants under `train.py`'s protocol -
   `hsv_v08` (brightness jitter doubled), `hsv_wide` (brightness doubled, hue
   widened), `no_colour` (the floor) - against the stock `laboro_yolo11n_exiffix`
   run. The test perturbation is deliberately not HSV jitter, so a model is not
   scored on its own augmentation.

```bash
python lighting/train_light.py --all
python lighting/run_lighting.py --model n         # also s m l x, light_<config>
bash lighting/run_all.sh                          # both, sequentially
```

Outputs land in `runs/lighting/<model>.json` and example frames in
`runs/lighting/examples/`.

## Results

`python lighting/make_tables.py` writes `lighting/results.md` from
`runs/lighting/*.json`: the four extremes against the clean frame, stage
accuracy, and the full sweep on both axes, for the size ladder and the three
HSV training variants.

`python paper/make_size_lighting_figures.py` writes the figures into
`paper/figures/`: `lighting_sizes.pdf/png` (recall and counting MAE along both
axes for n-x), `lighting_hsv.pdf/png` (the same for the training variants), and
`lighting_examples.jpg` (one validation frame at five exposure and five white
balance levels, captioned with the level's mean luma and the baseline recall).
