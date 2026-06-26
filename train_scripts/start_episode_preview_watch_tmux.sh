#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-gaze_wam_episode_preview_watch}"
ROOT="${ROOT:-/mnt/workspace/shenyibo/gaze-wam}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
POLL_SECONDS="${POLL_SECONDS:-180}"
STABLE_SECONDS="${STABLE_SECONDS:-45}"
RUN_ON_START="${RUN_ON_START:-1}"
PREVIEW_DEVICE="${PREVIEW_DEVICE:-cuda:0}"
SOURCE="${SOURCE:-open}"
SPLIT="${SPLIT:-val}"
EPISODE="${EPISODE:-86}"
USE_EMA="${USE_EMA:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FPS="${FPS:-15}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
STILL_INDICES="${STILL_INDICES:-0,60,120,180}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

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

mkdir -p "$OUTPUT_DIR/media/episode_heatmap/watched"

CMD=$(cat <<EOF
cd "$ROOT" && \
OUTPUT_DIR="$OUTPUT_DIR" \
POLL_SECONDS="$POLL_SECONDS" \
STABLE_SECONDS="$STABLE_SECONDS" \
RUN_ON_START="$RUN_ON_START" \
PREVIEW_DEVICE="$PREVIEW_DEVICE" \
SOURCE="$SOURCE" \
SPLIT="$SPLIT" \
EPISODE="$EPISODE" \
USE_EMA="$USE_EMA" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
FPS="$FPS" \
FRAME_STRIDE="$FRAME_STRIDE" \
MAX_FRAMES="$MAX_FRAMES" \
STILL_INDICES="$STILL_INDICES" \
SAMPLE_SEED="$SAMPLE_SEED" \
bash train_scripts/watch_gaze_wam_episode_preview.sh 2>&1 | tee -a "$OUTPUT_DIR/media/episode_heatmap/watched/watch.log"
EOF
)

tmux new-session -d -s "$SESSION" "$CMD"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "output_dir: $ROOT/$OUTPUT_DIR"
echo "episode_preview_root: $ROOT/$OUTPUT_DIR/media/episode_heatmap/watched"
