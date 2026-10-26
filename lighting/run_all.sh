#!/bin/bash
# Lighting section: train the HSV configs, then sweep every model through the
# exposure and white-balance levels. Sequential on purpose: one GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY="${PY:-$R/.venv/bin/python}"
cd "$R"

echo "[1/2] train hsv_v08, hsv_wide, no_colour (skips completed runs)"
"$PY" lighting/train_light.py --all

echo "[2/2] lighting sweep on every available model"
for m in n s m l x light_hsv_v08 light_hsv_wide light_no_colour; do
  "$PY" lighting/run_lighting.py --model "$m" || echo "skipped $m"
done

echo "DONE"
