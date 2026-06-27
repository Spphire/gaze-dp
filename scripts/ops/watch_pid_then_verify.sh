#!/usr/bin/env bash
# Watch an existing in-flight conversion / training process by PID. When the
# process dies, automatically run verify_open_zarr.sh on the output and
# preflight_gaze_wam.py with the given config. Result lands in $LOG so it
# can be tail'd without attaching to tmux.
#
# Designed to be launched via run_in_tmux.sh so it survives ssh dropouts.
#
# Usage:
#   scripts/ops/watch_pid_then_verify.sh <pid> <zarr-path> <preflight-config-name>
#
# Example:
#   bash scripts/ops/run_in_tmux.sh \
#     train_zarr_watch data/outputs/logs/train_zarr_watch.log \
#     bash scripts/ops/watch_pid_then_verify.sh 2749461 \
#       data/hot3d_open_train.zarr \
#       train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace

set -euo pipefail

PID=${1:-}
ZARR=${2:-}
CFG=${3:-}

if [[ -z "$PID" || -z "$ZARR" || -z "$CFG" ]]; then
  echo "usage: $0 <pid> <zarr-path> <preflight-config-name>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PYTHON=${PYTHON:-.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi

echo "=== watch_pid_then_verify start $(date) ==="
echo "  watching pid:   $PID"
echo "  zarr:           $ZARR"
echo "  preflight cfg:  $CFG"

# Wait until PID is gone. 10-min polling — process is long-lived video
# decode so frequent polling would waste cycles. Bound to 6 h for safety.
ROUND=0
MAX_ROUNDS=36
while [[ $ROUND -lt $MAX_ROUNDS ]]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date +%T)] pid $PID exited at round $ROUND"
    break
  fi
  if [[ -d "$ZARR" ]]; then
    SIZE=$(du -sh "$ZARR" 2>/dev/null | awk '{print $1}')
  else
    SIZE="(no zarr yet)"
  fi
  echo "[$(date +%T) round $ROUND] pid $PID alive, zarr=$SIZE"
  ROUND=$((ROUND+1))
  sleep 600
done

if kill -0 "$PID" 2>/dev/null; then
  echo "!!! TIMEOUT after $MAX_ROUNDS rounds, pid $PID still alive !!!" >&2
  exit 4
fi

# Filesystem flush before reading
sync
sleep 5

echo ""
echo "=== STEP 1: integrity verify ==="
if ! bash "$(dirname "$0")/verify_open_zarr.sh" "$ZARR"; then
  echo "INTEGRITY FAILED — zarr is broken, delete and re-run prepare_hot3d_zarr.sh" >&2
  exit 2
fi

echo ""
echo "=== STEP 2: preflight ==="
OUT_JSON="/tmp/preflight_${CFG}_$(date +%Y%m%d_%H%M%S).json"
"$PYTHON" scripts/preflight_gaze_wam.py \
  --config-name "$CFG" \
  --override task.open_dataset_path="$ZARR" \
  --override task.robot_dataset_path=null \
  --device cuda:0 \
  --output-json "$OUT_JSON" 2>&1 | tail -10
echo ""
echo "--- preflight errors / warnings ---"
"$PYTHON" - "$OUT_JSON" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
errs = r.get("errors", [])
warns = r.get("warnings", [])
print(f"top-level errors: {errs}")
print(f"top-level warnings: {warns}")
for k, v in r.items():
    if isinstance(v, dict):
        e = v.get("errors")
        if e:
            print(f"  {k}.errors: {e}")
        valid = v.get("valid")
        if valid is False:
            print(f"  {k}.valid: False")
sys.exit(0 if not errs else 3)
PYEOF
PFRC=$?

if [[ $PFRC -eq 0 ]]; then
  echo ""
  echo "=== ALL GREEN — zarr ready, preflight passed ==="
  echo "  zarr:     $ZARR"
  echo "  preflight: $OUT_JSON"
else
  echo "=== preflight reported errors, see above ===" >&2
  exit 3
fi

echo "=== watch_pid_then_verify done $(date) ==="
