#!/bin/bash
# Copy the exported size ladder into the app bundle under the names the model
# picker shows. Resources/models/ is git-ignored (220 MB); run this after
# export_coreml.py, then `xcodegen generate` so the project picks them up.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd -- "$HERE/../.." && pwd)"
mkdir -p "$HERE/Resources/models"
for sz in n s m l x; do
  src="$R/ios_model/yolo11${sz}_fp16/best.mlpackage"
  [ "$sz" = n ] && src="$R/ios_model/yolo11n_laboro_fp16/best.mlpackage"
  [ -d "$src" ] || { echo "skip $sz: missing $src"; continue; }
  rm -rf "$HERE/Resources/models/YOLO11${sz}.mlpackage"
  cp -R "$src" "$HERE/Resources/models/YOLO11${sz}.mlpackage"
  echo "YOLO11$sz <- $src"
done
