#!/bin/bash
# Whole section end to end: fetch, convert, train, evaluate.
# Sequential on purpose, the runs share one GPU.
set -euo pipefail

R=/Users/dvkalandadze/projects/final_project
PY="$R/.venv/bin/python"
cd "$R"

echo "[1/5] fetch (resumes if the archive is already complete)"
"$PY" cross_dataset/fetch.py

echo "[2/5] prepare both datasets"
"$PY" cross_dataset/prepare.py

echo "[3/5] train laboro3"
"$PY" cross_dataset/train_cross.py --dataset laboro3

echo "[4/5] train mix"
"$PY" cross_dataset/train_cross.py --dataset mix

echo "[5/5] evaluate the 3x2 matrix"
"$PY" cross_dataset/eval_cross.py

echo "DONE"
