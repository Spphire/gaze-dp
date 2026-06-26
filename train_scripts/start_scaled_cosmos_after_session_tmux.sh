#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/shenyibo/gaze-wam}"
QUEUE_SESSION="${QUEUE_SESSION:-gaze_wam_cosmos_intensity_queue}"
WAIT_SESSION="${WAIT_SESSION:-gaze_wam_open_cosmos_8gpu}"
TRAIN_SESSION="${TRAIN_SESSION:-gaze_wam_open_cosmos_intensity_8gpu}"
WATCH_SESSION="${WATCH_SESSION:-gaze_wam_cosmos_intensity_ckpt_preview_watch}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-data/outputs/hot3d_open_cosmos_intensity_softplus_latent_8gpu_amp}"
CONFIG_NAME="${CONFIG_NAME:-train_gaze_wam_open_only_cosmos_workspace}"
HEATMAP_OBJECTIVE="${HEATMAP_OBJECTIVE:-dsnt_js}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATE_EVERY="${GRADIENT_ACCUMULATE_EVERY:-4}"
PRETRAINED="${PRETRAINED:-true}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
RESUME="${RESUME:-false}"
RESUME_EPOCH="${RESUME_EPOCH:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"

HEATMAP_DIM="${HEATMAP_DIM:-16}"
HEATMAP_SPATIAL_DECODER="${HEATMAP_SPATIAL_DECODER:-cosmos_tokenizer}"
HEATMAP_COSMOS_ENCODER_PATH="${HEATMAP_COSMOS_ENCODER_PATH:-data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit}"
HEATMAP_COSMOS_DECODER_PATH="${HEATMAP_COSMOS_DECODER_PATH:-data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit}"
HEATMAP_COSMOS_INPUT_RANGE="${HEATMAP_COSMOS_INPUT_RANGE:-minus_one_one}"
HEATMAP_COSMOS_OUTPUT_RANGE="${HEATMAP_COSMOS_OUTPUT_RANGE:-minus_one_one}"
HEATMAP_COSMOS_INPUT_NORMALIZATION="${HEATMAP_COSMOS_INPUT_NORMALIZATION:-max}"
HEATMAP_LATENT_SCALE="${HEATMAP_LATENT_SCALE:-0.25}"
HEATMAP_LATENT_OFFSET="${HEATMAP_LATENT_OFFSET:-0.0}"
HEATMAP_LATENT_STATS_PATH="${HEATMAP_LATENT_STATS_PATH:-data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json}"
HEATMAP_SCHEDULER_CLIP_SAMPLE="${HEATMAP_SCHEDULER_CLIP_SAMPLE:-true}"
HEATMAP_DISTRIBUTION_MODE="${HEATMAP_DISTRIBUTION_MODE:-intensity_softplus}"
HEATMAP_DSNT_TEMPERATURE="${HEATMAP_DSNT_TEMPERATURE:-0.1}"

PREVIEW_POLL_SECONDS="${PREVIEW_POLL_SECONDS:-180}"
PREVIEW_STABLE_SECONDS="${PREVIEW_STABLE_SECONDS:-45}"
PREVIEW_RUN_ON_START="${PREVIEW_RUN_ON_START:-1}"
PREVIEW_DEVICE="${PREVIEW_DEVICE:-cuda:7}"
PREVIEW_MAX_SAMPLES="${PREVIEW_MAX_SAMPLES:-4}"
PREVIEW_SAMPLE_SEED="${PREVIEW_SAMPLE_SEED:-42}"
PREVIEW_SOURCE="${PREVIEW_SOURCE:-open}"
PREVIEW_SPLIT="${PREVIEW_SPLIT:-val}"
PREVIEW_USE_EMA="${PREVIEW_USE_EMA:-1}"

cd "$ROOT"

if tmux has-session -t "$QUEUE_SESSION" 2>/dev/null; then
  echo "Queue tmux session already exists: $QUEUE_SESSION" >&2
  tmux ls
  exit 1
fi

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "Training tmux session already exists: $TRAIN_SESSION" >&2
  tmux ls
  exit 1
fi

if tmux has-session -t "$WATCH_SESSION" 2>/dev/null; then
  echo "Watcher tmux session already exists: $WATCH_SESSION" >&2
  tmux ls
  exit 1
fi

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing project venv at $ROOT/.venv" >&2
  exit 2
fi

if [[ ! -d "data/hot3d_open.zarr" ]]; then
  echo "Missing dataset at $ROOT/data/hot3d_open.zarr" >&2
  exit 2
fi

for required_file in \
  "$HEATMAP_COSMOS_ENCODER_PATH" \
  "$HEATMAP_COSMOS_DECODER_PATH" \
  "$HEATMAP_LATENT_STATS_PATH"
do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 2
  fi
done

QUEUE_DIR="${QUEUE_DIR:-data/outputs/queued_runs}"
mkdir -p "$QUEUE_DIR"
RUN_SCRIPT="$QUEUE_DIR/${QUEUE_SESSION}.run.sh"
QUEUE_LOG="$QUEUE_DIR/${QUEUE_SESSION}.log"

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"

echo "[\$(date --iso-8601=seconds)] queue started; waiting for tmux session '$WAIT_SESSION'"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[\$(date --iso-8601=seconds)] still waiting for '$WAIT_SESSION'"
  sleep "$WAIT_POLL_SECONDS"
done

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "[\$(date --iso-8601=seconds)] training session '$TRAIN_SESSION' already exists; not launching duplicate"
  exit 0
fi

RUN_ID="\$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_PREFIX}_\${RUN_ID}"
mkdir -p "\$OUTPUT_DIR"
echo "[\$(date --iso-8601=seconds)] launching Cosmos intensity-softplus run: \$OUTPUT_DIR"

SESSION="$TRAIN_SESSION" \\
OUTPUT_DIR="\$OUTPUT_DIR" \\
CONFIG_NAME="$CONFIG_NAME" \\
HEATMAP_OBJECTIVE="$HEATMAP_OBJECTIVE" \\
NUM_EPOCHS="$NUM_EPOCHS" \\
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \\
MAX_VAL_STEPS="$MAX_VAL_STEPS" \\
BATCH_SIZE="$BATCH_SIZE" \\
GRADIENT_ACCUMULATE_EVERY="$GRADIENT_ACCUMULATE_EVERY" \\
PRETRAINED="$PRETRAINED" \\
LOGGING_MODE="$LOGGING_MODE" \\
RESUME="$RESUME" \\
RESUME_EPOCH="$RESUME_EPOCH" \\
CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \\
SAMPLE_EVERY="$SAMPLE_EVERY" \\
HEATMAP_DIM="$HEATMAP_DIM" \\
HEATMAP_SPATIAL_DECODER="$HEATMAP_SPATIAL_DECODER" \\
HEATMAP_COSMOS_ENCODER_PATH="$HEATMAP_COSMOS_ENCODER_PATH" \\
HEATMAP_COSMOS_DECODER_PATH="$HEATMAP_COSMOS_DECODER_PATH" \\
HEATMAP_COSMOS_INPUT_RANGE="$HEATMAP_COSMOS_INPUT_RANGE" \\
HEATMAP_COSMOS_OUTPUT_RANGE="$HEATMAP_COSMOS_OUTPUT_RANGE" \\
HEATMAP_COSMOS_INPUT_NORMALIZATION="$HEATMAP_COSMOS_INPUT_NORMALIZATION" \\
HEATMAP_LATENT_SCALE="$HEATMAP_LATENT_SCALE" \\
HEATMAP_LATENT_OFFSET="$HEATMAP_LATENT_OFFSET" \\
HEATMAP_LATENT_STATS_PATH="$HEATMAP_LATENT_STATS_PATH" \\
HEATMAP_SCHEDULER_CLIP_SAMPLE="$HEATMAP_SCHEDULER_CLIP_SAMPLE" \\
HEATMAP_DISTRIBUTION_MODE="$HEATMAP_DISTRIBUTION_MODE" \\
HEATMAP_DSNT_TEMPERATURE="$HEATMAP_DSNT_TEMPERATURE" \\
bash train_scripts/start_open_only_8gpu_tmux.sh

sleep 20

SESSION="$WATCH_SESSION" \\
OUTPUT_DIR="\$OUTPUT_DIR" \\
POLL_SECONDS="$PREVIEW_POLL_SECONDS" \\
STABLE_SECONDS="$PREVIEW_STABLE_SECONDS" \\
RUN_ON_START="$PREVIEW_RUN_ON_START" \\
PREVIEW_DEVICE="$PREVIEW_DEVICE" \\
MAX_SAMPLES="$PREVIEW_MAX_SAMPLES" \\
SAMPLE_SEED="$PREVIEW_SAMPLE_SEED" \\
SOURCE="$PREVIEW_SOURCE" \\
SPLIT="$PREVIEW_SPLIT" \\
USE_EMA="$PREVIEW_USE_EMA" \\
bash train_scripts/start_ckpt_preview_watch_tmux.sh

echo "[\$(date --iso-8601=seconds)] launched training='$TRAIN_SESSION' watcher='$WATCH_SESSION' output='\$OUTPUT_DIR'"
EOF

chmod +x "$RUN_SCRIPT"
tmux new-session -d -s "$QUEUE_SESSION" "bash '$RUN_SCRIPT' 2>&1 | tee -a '$QUEUE_LOG'"

echo "started queue tmux session: $QUEUE_SESSION"
echo "queue_script: $ROOT/$RUN_SCRIPT"
echo "queue_log: $ROOT/$QUEUE_LOG"
echo "waiting_for: $WAIT_SESSION"
echo "future_training_session: $TRAIN_SESSION"
echo "future_watcher_session: $WATCH_SESSION"
