#!/bin/bash
# Whole section end to end: fetch, convert, train, evaluate.
# Sequential on purpose, the runs share one GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY="$R/.venv/bin/python"
cd "$R"

echo "[1/4] fetch (resumes if the archive is already complete)"
"$PY" cross_dataset/fetch.py

echo "[2/4] prepare both datasets"
"$PY" cross_dataset/prepare.py

echo "[3/4] train laboro3, tomatod, and mix"
"$PY" cross_dataset/train_cross.py --all

echo "[4/4] evaluate the 3x2 matrix"
"$PY" cross_dataset/eval_cross.py

echo "DONE"
