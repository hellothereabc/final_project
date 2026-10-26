#!/usr/bin/env bash
# Score every seeded colour-augmentation run over the lighting sweep.
#
# Writes runs/lighting/seeds/<config>_s<seed>.json, which carry the per-image
# series the paired tests need. The first-pass files in runs/lighting/ are the
# earlier one-seed results and are never touched.
#
#   bash lighting/sweep_seeds.sh                 # all four configs, seeds 0 1 2
#   PY=/path/to/python bash lighting/sweep_seeds.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/Users/chel/Projects/master_diploma/reproduce/.venv/bin/python}"
OUT="$ROOT/runs/lighting/seeds"
CONFIGS="${CONFIGS:-stock hsv_v08 hsv_wide no_colour}"
SEEDS="${SEEDS:-0 1 2}"

mkdir -p "$OUT" "$ROOT/runs/logs"

run_dir_for() {  # config seed -> run directory name
  local config="$1" seed="$2"
  if [ "$seed" = "0" ]; then
    if [ "$config" = "stock" ]; then echo "laboro_yolo11n_export"; else echo "light_$config"; fi
  else
    echo "light_${config}_s${seed}"
  fi
}

for seed in $SEEDS; do
  for config in $CONFIGS; do
    run="$(run_dir_for "$config" "$seed")"
    out="$OUT/${config}_s${seed}.json"
    if [ -f "$out" ]; then
      echo "skip $config seed $seed: $out exists"
      continue
    fi
    if [ ! -f "$ROOT/runs/$run/weights/best.pt" ]; then
      echo "skip $config seed $seed: runs/$run/weights/best.pt not trained yet"
      continue
    fi
    echo "=== $config seed $seed  (runs/$run)"
    "$PY" "$ROOT/lighting/run_lighting.py" --model "$run" --out "$out" --no-examples \
      2>&1 | tee "$ROOT/runs/logs/sweep_${config}_s${seed}.log" | tail -3
  done
done

echo "done; now: $PY lighting/seed_stats.py"
