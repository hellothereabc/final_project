#!/bin/bash
# Build the app around one exported model, install it on the connected iPhone,
# run the Benchmark screen unattended, and pull the JSON back.
#
#   runtime_bench/phone_bench.sh s               # 10 repeats, cpuOnly + cpuAndNeuralEngine
#   runtime_bench/phone_bench.sh x 1500 ane      # sustained run, Neural Engine only
#
# The app bundles the whole size ladder (Resources/models, filled by
# ios/TomatoRipeness/sync_models.sh) and picks one by the `-model` launch
# argument, so nothing in the project is touched. Needs the phone plugged in,
# unlocked, and trusted; the first install per Apple ID may ask Xcode to
# register the device.
set -euo pipefail

SIZE="${1:?model size: n s m l x}"
REPEATS="${2:-10}"
MODE="${3:-both}"                    # both | ane

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd -- "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$R/ios/TomatoRipeness"
PROJECT="$APP_DIR/TomatoRipeness.xcodeproj"
BUNDLE_ID=com.dvk.TomatoRipeness
DERIVED="${DERIVED:-$R/runs/runtime/derived}"

case "$SIZE" in
  n|s|m|l|x) MODEL="YOLO11$SIZE" ;;
  *) echo "unknown size $SIZE"; exit 1 ;;
esac
[ -d "$APP_DIR/Resources/models/$MODEL.mlpackage" ] || {
  echo "missing Resources/models/$MODEL.mlpackage - run ios/TomatoRipeness/sync_models.sh"; exit 1; }

# The phone. Take the only available one, or DEVICE=<udid> to pick.
if [ -z "${DEVICE:-}" ]; then
  DEVICE="$(xcrun devicectl list devices 2>/dev/null \
    | awk '$4 ~ /available|connected/ && $0 ~ /iPhone/ {print $3; exit}')"
fi
[ -n "$DEVICE" ] || { echo "no available iPhone in 'xcrun devicectl list devices' - plug it in and unlock it"; exit 1; }
MODEL_ID="$(xcrun devicectl device info details --device "$DEVICE" 2>/dev/null \
  | awk -F': ' '/productType/ {print $2; exit}')"
echo "device $DEVICE ($MODEL_ID)  model $MODEL  repeats $REPEATS  mode $MODE"

mkdir -p "$R/runs/runtime"

echo "[1/4] build (Release, device)"
xcodebuild -project "$PROJECT" -scheme TomatoRipeness -configuration Release \
  -destination "id=$DEVICE" -derivedDataPath "$DERIVED" -allowProvisioningUpdates \
  build -quiet
APP="$(find "$DERIVED/Build/Products" -name TomatoRipeness.app -path '*iphoneos*' | head -1)"
[ -d "$APP" ] || { echo "no .app under $DERIVED"; exit 1; }

echo "[2/4] install"
xcrun devicectl device install app --device "$DEVICE" "$APP" --quiet

echo "[3/4] launch"
ARGS=(-model "$MODEL" -autorun-benchmark -benchmark-model "$MODEL" -benchmark-repeats "$REPEATS")
[ "$MODE" = ane ] && ARGS+=(-benchmark-ane-only)
# Overwrite any report left by an earlier run, so the poll below cannot pick up
# a stale file that happens to carry the same model label and repeat count.
echo '{"model": "stale"}' > "$R/runs/runtime/stale.json"
xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" --source "$R/runs/runtime/stale.json" \
  --destination Documents/benchmark.json --quiet 2>/dev/null || true
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing \
  --quiet "$BUNDLE_ID" -- "${ARGS[@]}"

echo "[4/4] wait for Documents/benchmark.json"
SUFFIX="laboro"; [ "$MODE" = ane ] && SUFFIX="sustained"
OS_TAG="ios$(xcrun devicectl device info details --device "$DEVICE" 2>/dev/null \
  | awk -F': ' '/osVersionNumber/ {print $2; exit}' | cut -d. -f1)"
OUT="$R/runtime_bench/phone_${MODEL_ID//,/-}_${OS_TAG}_yolo11${SIZE}_${SUFFIX}.json"
# A device measurement cannot be reproduced by re-running a script, so an
# existing report is never overwritten.
if [ -e "$OUT" ]; then OUT="${OUT%.json}_$(date +%Y%m%d-%H%M%S).json"; fi
TMP="$R/runs/runtime/benchmark_pull.json"
STARTED=$(date +%s)
while true; do
  sleep 10
  rm -f "$TMP"
  if xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer \
       --domain-identifier "$BUNDLE_ID" --source Documents/benchmark.json \
       --destination "$TMP" --quiet 2>/dev/null \
     && python3 -c "import json,sys; d=json.load(open('$TMP')); sys.exit(0 if d.get('model')=='$MODEL' and d.get('repeats')==$REPEATS else 1)"; then
    break
  fi
  if [ $(( $(date +%s) - STARTED )) -gt 1800 ]; then echo "timed out after 30 min"; exit 1; fi
done
mv "$TMP" "$OUT"
echo "wrote $OUT"
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d['device']}  {d['os']}  {d['model']}  repeats {d['repeats']}")
for c in d["configs"]:
    print(f"   {c['computeUnits']:<20} median {c['median_ms']:6.2f} ms  p90 {c['p90_ms']:6.2f}"
          f"  fps {1000 / c['median_ms']:6.1f}  load {c['load_ms']:6.0f} ms"
          f"  frames {c['frames']}  thermal {c['thermal_state']}")
PY
