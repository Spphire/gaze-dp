#!/usr/bin/env bash
# Estimate Cosmos heatmap latent statistics from an existing open-gaze zarr.
# Writes a JSON file used by the policy to set heatmap latent scale/offset.
#
# Cheap: ~1 minute on a single L20X / H200. Should be re-run any time the
# camera pipeline, image size, or token grid changes.
#
# Usage:
#   scripts/ops/prepare_cosmos_latent_stats.sh <source-zarr> <output-json>
#
# Environment:
#   PYTHON       (default: .venv/bin/python)
#   ENCODER_PATH (default: data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit)
#   DECODER_PATH (default: data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit)
#   IMAGE_SIZE   (default: 256 256)
#   TOKEN_GRID   (default: 16 16)
#   MAX_SAMPLES  (default: 4096)
#   SEED         (default: 42)
#   DEVICE       (default: cuda:0)

set -euo pipefail

SRC_ZARR=${1:-}
OUT_JSON=${2:-}
if [[ -z "$SRC_ZARR" || -z "$OUT_JSON" ]]; then
  echo "usage: $0 <source-zarr> <output-json>" >&2
  exit 1
fi
if [[ ! -d "$SRC_ZARR" ]]; then
  echo "source zarr not found: $SRC_ZARR" >&2
  exit 1
fi

PYTHON=${PYTHON:-.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
ENCODER_PATH=${ENCODER_PATH:-data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit}
DECODER_PATH=${DECODER_PATH:-data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit}
IMAGE_SIZE=${IMAGE_SIZE:-256 256}
TOKEN_GRID=${TOKEN_GRID:-16 16}
MAX_SAMPLES=${MAX_SAMPLES:-4096}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$(dirname "$OUT_JSON")"

echo "=== prepare_cosmos_latent_stats.sh start $(date) ==="
echo "  source: $SRC_ZARR"
echo "  output: $OUT_JSON"
echo "  encoder: $ENCODER_PATH"
echo "  device:  $DEVICE  samples: $MAX_SAMPLES  seed: $SEED"

"$PYTHON" diffusion_policy/scripts/estimate_cosmos_heatmap_latent_stats.py \
  --dataset-path "$SRC_ZARR" \
  --encoder-path "$ENCODER_PATH" \
  --decoder-path "$DECODER_PATH" \
  --output-path  "$OUT_JSON" \
  --image-size  $IMAGE_SIZE \
  --token-grid  $TOKEN_GRID \
  --max-samples $MAX_SAMPLES \
  --seed        $SEED \
  --device      "$DEVICE"

if [[ ! -s "$OUT_JSON" ]]; then
  echo "ERROR: output JSON missing or empty" >&2
  exit 2
fi
echo "=== prepare_cosmos_latent_stats.sh OK $(date) ==="
echo "  $(stat -c '%s bytes' "$OUT_JSON")"
