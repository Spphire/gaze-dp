#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-gaze_wam_ckpt_preview_watch}"
ROOT="${ROOT:-/mnt/workspace/shenyibo/gaze-wam}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
POLL_SECONDS="${POLL_SECONDS:-180}"
STABLE_SECONDS="${STABLE_SECONDS:-45}"
RUN_ON_START="${RUN_ON_START:-1}"
PREVIEW_DEVICE="${PREVIEW_DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SOURCE="${SOURCE:-open}"
SPLIT="${SPLIT:-val}"
USE_EMA="${USE_EMA:-1}"

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Set OUTPUT_DIR to the training output directory." >&2
  exit 2
fi

cd "$ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  tmux ls
  exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Missing output directory: $ROOT/$OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/media/ckpt_heatmap/watched"

CMD=$(cat <<EOF
cd "$ROOT" && \
OUTPUT_DIR="$OUTPUT_DIR" \
POLL_SECONDS="$POLL_SECONDS" \
STABLE_SECONDS="$STABLE_SECONDS" \
RUN_ON_START="$RUN_ON_START" \
PREVIEW_DEVICE="$PREVIEW_DEVICE" \
MAX_SAMPLES="$MAX_SAMPLES" \
SAMPLE_SEED="$SAMPLE_SEED" \
SOURCE="$SOURCE" \
SPLIT="$SPLIT" \
USE_EMA="$USE_EMA" \
bash train_scripts/watch_gaze_wam_ckpt_preview.sh 2>&1 | tee -a "$OUTPUT_DIR/media/ckpt_heatmap/watched/watch.log"
EOF
)

tmux new-session -d -s "$SESSION" "$CMD"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "output_dir: $ROOT/$OUTPUT_DIR"
echo "preview_root: $ROOT/$OUTPUT_DIR/media/ckpt_heatmap/watched"
