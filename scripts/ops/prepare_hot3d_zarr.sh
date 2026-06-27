#!/usr/bin/env bash
# Convert a HOT3D split (a text file of sequence ids, one per line) into the
# Gaze-WAM open-gaze zarr contract.
#
# Idempotent: if the output zarr exists, it is removed (with sync + verify)
# before the conversion is launched, so we never hit the half-deleted /
# half-rewritten race that produced a zarr with missing .zarray files on
# 2026-06-27.
#
# Usage:
#   scripts/ops/prepare_hot3d_zarr.sh <split-file> <output-zarr> [--image-size H W]
#
# Examples:
#   scripts/ops/prepare_hot3d_zarr.sh data/hot3d_val_sequences.txt   data/hot3d_open_val.zarr
#   scripts/ops/prepare_hot3d_zarr.sh data/hot3d_train_sequences.txt data/hot3d_open_train.zarr
#
# Environment:
#   PROCESSED_ROOT (default: /mnt/workspace/shenyibo/datasets/HOT3D/processed)
#   PYTHON         (default: .venv/bin/python; falls back to python3)
#   IMAGE_SIZE     (default: 256 256 — matches Cosmos / mixed_nll config)
#   TOKEN_GRID     (default: 16 16)
#
# Exit codes:
#   0 = success, output zarr present and complete
#   1 = bad arguments / preconditions
#   2 = python conversion script returned non-zero
#   3 = post-conversion integrity verifier failed (zarr is INVALID — delete and retry)

set -euo pipefail

SPLIT_FILE=${1:-}
OUTPUT_ZARR=${2:-}
shift 2 || true

if [[ -z "$SPLIT_FILE" || -z "$OUTPUT_ZARR" ]]; then
  echo "usage: $0 <split-file> <output-zarr> [extra args to converter]" >&2
  exit 1
fi
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "split file not found: $SPLIT_FILE" >&2
  exit 1
fi

PROCESSED_ROOT=${PROCESSED_ROOT:-/mnt/workspace/shenyibo/datasets/HOT3D/processed}
PYTHON=${PYTHON:-.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
IMAGE_SIZE=${IMAGE_SIZE:-256 256}
TOKEN_GRID=${TOKEN_GRID:-16 16}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=== prepare_hot3d_zarr.sh start $(date) ==="
echo "  split:           $SPLIT_FILE ($(wc -l < "$SPLIT_FILE") sequences)"
echo "  output:          $OUTPUT_ZARR"
echo "  processed_root:  $PROCESSED_ROOT"
echo "  image_size:      $IMAGE_SIZE"
echo "  token_grid:      $TOKEN_GRID"
echo "  python:          $PYTHON"

# Pre-clean: explicit two-step delete + verify gone, no race with --overwrite
if [[ -e "$OUTPUT_ZARR" ]]; then
  echo "--- pre-clean: removing existing zarr ---"
  rm -rf "$OUTPUT_ZARR"
  sync
  sleep 2
  if [[ -e "$OUTPUT_ZARR" ]]; then
    echo "ERROR: failed to remove $OUTPUT_ZARR" >&2
    exit 1
  fi
fi
echo "  pre-state: clean"

# Conversion (without --overwrite — we already cleaned)
echo "--- running converter ---"
"$PYTHON" scripts/convert_hot3d_processed_to_open_zarr.py \
  --processed-root "$PROCESSED_ROOT" \
  --output-zarr "$OUTPUT_ZARR" \
  --sequence-file "$SPLIT_FILE" \
  --image-size $IMAGE_SIZE \
  --stride 1 \
  --heatmap-storage token \
  --heatmap-token-grid $TOKEN_GRID \
  "$@"
RC=$?
echo "--- converter exit: $RC ---"
if [[ $RC -ne 0 ]]; then
  exit 2
fi

# Sync filesystem, then verify
sync
sleep 3

echo "--- running integrity verifier ---"
bash "$(dirname "$0")/verify_open_zarr.sh" "$OUTPUT_ZARR"
VRC=$?
if [[ $VRC -ne 0 ]]; then
  echo "ZARR INTEGRITY FAILED — output is NOT usable, delete and re-run" >&2
  exit 3
fi

echo "=== prepare_hot3d_zarr.sh OK $(date) ==="
