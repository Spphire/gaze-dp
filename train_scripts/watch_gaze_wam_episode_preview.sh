#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Set OUTPUT_DIR to the training output directory." >&2
  exit 2
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
PREVIEW_ROOT="${PREVIEW_ROOT:-$OUTPUT_DIR/media/episode_heatmap/watched}"
POLL_SECONDS="${POLL_SECONDS:-180}"
STABLE_SECONDS="${STABLE_SECONDS:-30}"
RUN_ON_START="${RUN_ON_START:-1}"
ONCE="${ONCE:-0}"
SOURCE="${SOURCE:-open}"
SPLIT="${SPLIT:-val}"
EPISODE="${EPISODE:-86}"
PREVIEW_DEVICE="${PREVIEW_DEVICE:-cuda:0}"
USE_EMA="${USE_EMA:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FPS="${FPS:-15}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
STILL_INDICES="${STILL_INDICES:-0,60,120,180}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable python at $PYTHON_BIN." >&2
  exit 2
fi

mkdir -p "$PREVIEW_ROOT"
STATE_FILE="${STATE_FILE:-$PREVIEW_ROOT/.last_previewed_ckpt}"
INDEX_FILE="${INDEX_FILE:-$PREVIEW_ROOT/index.tsv}"

find_latest_ckpt() {
  find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

ckpt_signature() {
  local ckpt="$1"
  stat -c '%n	%Y	%s' "$ckpt"
}

preview_once() {
  local ckpt="$1"
  local signature="$2"
  local ckpt_base tag output_dir
  ckpt_base="$(basename "$ckpt" .ckpt)"
  tag="$(date +%Y%m%d_%H%M%S)_${ckpt_base//[^A-Za-z0-9_.=-]/_}_episode${EPISODE}"
  output_dir="$PREVIEW_ROOT/$tag"
  mkdir -p "$output_dir"

  local args=(
    -m diffusion_policy.scripts.preview_gaze_wam_episode
    --checkpoint "$ckpt"
    --trust-checkpoint
    --output-dir "$output_dir"
    --source "$SOURCE"
    --split "$SPLIT"
    --episode "$EPISODE"
    --device "$PREVIEW_DEVICE"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --fps "$FPS"
    --frame-stride "$FRAME_STRIDE"
  )
  if [[ -n "$MAX_FRAMES" ]]; then
    args+=(--max-frames "$MAX_FRAMES")
  fi
  if [[ -n "$STILL_INDICES" ]]; then
    args+=(--still-indices "$STILL_INDICES")
  fi
  if [[ -n "$SAMPLE_SEED" ]]; then
    args+=(--sample-seed "$SAMPLE_SEED")
  fi
  if [[ "$USE_EMA" == "1" || "$USE_EMA" == "true" ]]; then
    args+=(--use-ema)
  else
    args+=(--no-use-ema)
  fi

  echo "[$(date --iso-8601=seconds)] episode previewing $ckpt -> $output_dir"
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$output_dir/preview.log"
  printf '%s\n' "$signature" > "$STATE_FILE"
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$ckpt" "$output_dir" >> "$INDEX_FILE"
}

echo "Watching checkpoints in $CHECKPOINT_DIR"
echo "Episode preview root: $PREVIEW_ROOT"
echo "Device: $PREVIEW_DEVICE, source=$SOURCE, split=$SPLIT, episode=$EPISODE"

while true; do
  ckpt="$(find_latest_ckpt || true)"
  if [[ -z "${ckpt:-}" || ! -f "$ckpt" ]]; then
    echo "[$(date --iso-8601=seconds)] no checkpoint found"
    if [[ "$ONCE" == "1" ]]; then
      exit 1
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  signature="$(ckpt_signature "$ckpt")"
  last_signature=""
  if [[ -f "$STATE_FILE" ]]; then
    last_signature="$(cat "$STATE_FILE")"
  fi

  if [[ "$signature" != "$last_signature" ]]; then
    if [[ -z "$last_signature" && "$RUN_ON_START" != "1" ]]; then
      printf '%s\n' "$signature" > "$STATE_FILE"
      echo "[$(date --iso-8601=seconds)] registered existing checkpoint without preview: $ckpt"
    else
      echo "[$(date --iso-8601=seconds)] detected checkpoint change; waiting ${STABLE_SECONDS}s for file stability"
      sleep "$STABLE_SECONDS"
      stable_signature="$(ckpt_signature "$ckpt")"
      if [[ "$stable_signature" != "$signature" ]]; then
        echo "[$(date --iso-8601=seconds)] checkpoint still changing; will retry on the next poll"
        continue
      fi
      preview_once "$ckpt" "$signature"
    fi
    if [[ "$ONCE" == "1" ]]; then
      exit 0
    fi
  fi

  sleep "$POLL_SECONDS"
done
