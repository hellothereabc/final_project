#!/bin/bash
# The whole occlusion section, in the order the notebook reads it.
# Sequential on purpose: every step is inference on the one GPU.
set -euo pipefail

R=/Users/dvkalandadze/projects/final_project
PY="$R/.venv/bin/python"
cd "$R"

echo "[1/6] synthetic sweep, all three modes, matching at IoU 0.5"
"$PY" occlusion/run_occlusion.py --model laboro3 --data laboro3 --occluder leaf

echo "[2/6] same sweep matched at IoU 0.3 - separates 'not found' from 'boxed the visible part'"
"$PY" occlusion/run_occlusion.py --model laboro3 --data laboro3 --occluder leaf \
    --modes random --levels 0.0 0.3 0.4 0.5 0.6 0.7 --iou-thr 0.3

echo "[3/6] paired red-half vs green-half, same fruit twice"
"$PY" occlusion/colour_flip.py --fraction 0.4

echo "[4/6] occlusion already in the data: fruit behind fruit"
"$PY" occlusion/natural_occlusion.py

echo "[5/6] controls: flat black occluder, and the checkpoint the app ships"
"$PY" occlusion/run_occlusion.py --model laboro3 --data laboro3 --occluder black --modes random
"$PY" occlusion/run_occlusion.py --model mix --data laboro3 --occluder leaf --modes random

echo "[6/6] write-up"
cd occlusion
"$PY" build_occlusion_notebook.py
"$PY" -m nbconvert --to notebook --execute --inplace occlusion.ipynb \
    --ExecutePreprocessor.timeout=1800

echo "DONE"
