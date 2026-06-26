#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/hot3d_open_cosmos_intensity_softplus_latent_8gpu_amp}"
CONFIG_NAME="${CONFIG_NAME:-train_gaze_wam_open_only_cosmos_workspace}"
HEATMAP_OBJECTIVE="${HEATMAP_OBJECTIVE:-}"
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
HEATMAP_DIFFUSION_FINAL_LOSS_ENABLED="${HEATMAP_DIFFUSION_FINAL_LOSS_ENABLED:-}"
HEATMAP_FINAL_LOSS_TIMESTEP_WEIGHTING="${HEATMAP_FINAL_LOSS_TIMESTEP_WEIGHTING:-}"
HEATMAP_XY_LOSS_WEIGHT="${HEATMAP_XY_LOSS_WEIGHT:-}"
HEATMAP_POINT_NLL_LOSS_WEIGHT="${HEATMAP_POINT_NLL_LOSS_WEIGHT:-}"
HEATMAP_JS_LOSS_WEIGHT="${HEATMAP_JS_LOSS_WEIGHT:-}"
TEMPORAL_HEATMAP_MODE="${TEMPORAL_HEATMAP_MODE:-}"
TEMPORAL_HEATMAP_WINDOW_RADIUS="${TEMPORAL_HEATMAP_WINDOW_RADIUS:-}"
TEMPORAL_HEATMAP_BETA="${TEMPORAL_HEATMAP_BETA:-}"
TEMPORAL_HEATMAP_SIGMA_PX="${TEMPORAL_HEATMAP_SIGMA_PX:-}"
TEMPORAL_HEATMAP_CURRENT_WEIGHT="${TEMPORAL_HEATMAP_CURRENT_WEIGHT:-}"

ACCELERATE_BIN="$ROOT/.venv/bin/accelerate"
if [[ ! -x "$ACCELERATE_BIN" ]]; then
  echo "Missing $ACCELERATE_BIN. Create the project venv first." >&2
  exit 1
fi

if [[ "$HEATMAP_SPATIAL_DECODER" != "cosmos_tokenizer" ]]; then
  echo "Only HEATMAP_SPATIAL_DECODER=cosmos_tokenizer is supported." >&2
  exit 1
fi

if [[ ! -f "$HEATMAP_COSMOS_ENCODER_PATH" ]]; then
  echo "Missing Cosmos encoder JIT: $HEATMAP_COSMOS_ENCODER_PATH" >&2
  exit 1
fi

if [[ ! -f "$HEATMAP_COSMOS_DECODER_PATH" ]]; then
  echo "Missing Cosmos decoder JIT: $HEATMAP_COSMOS_DECODER_PATH" >&2
  exit 1
fi

ARGS=(
  launch
  --config_file accelerate/8gpu-amp.yaml
  train.py
  "--config-name=${CONFIG_NAME}"
  "policy.model_architecture=cached_dual_stream"
  "task.heatmap_key=null"
  "task.heatmap_dim=${HEATMAP_DIM}"
  "policy.heatmap_dim=${HEATMAP_DIM}"
  "policy.heatmap_spatial_decoder=${HEATMAP_SPATIAL_DECODER}"
  "training.require_amp=true"
  "policy.obs_encoder.pretrained=${PRETRAINED}"
  "open_dataloader.batch_size=${BATCH_SIZE}"
  "val_open_dataloader.batch_size=${BATCH_SIZE}"
  "training.gradient_accumulate_every=${GRADIENT_ACCUMULATE_EVERY}"
  "training.num_epochs=${NUM_EPOCHS}"
  "training.checkpoint_every=${CHECKPOINT_EVERY}"
  "training.sample_every=${SAMPLE_EVERY}"
  "training.val_every=1"
  "training.save_val_heatmap_preview=true"
  "training.max_val_steps=${MAX_VAL_STEPS}"
  "training.resume=${RESUME}"
  "logging.mode=${LOGGING_MODE}"
  "hydra.run.dir=${OUTPUT_DIR}"
)

if [[ -n "$HEATMAP_OBJECTIVE" ]]; then
  ARGS+=("policy.heatmap_objective=${HEATMAP_OBJECTIVE}")
fi

ARGS+=(
  "policy.heatmap_cosmos_encoder_path=${HEATMAP_COSMOS_ENCODER_PATH}"
  "policy.heatmap_cosmos_decoder_path=${HEATMAP_COSMOS_DECODER_PATH}"
  "policy.heatmap_cosmos_input_range=${HEATMAP_COSMOS_INPUT_RANGE}"
  "policy.heatmap_cosmos_output_range=${HEATMAP_COSMOS_OUTPUT_RANGE}"
  "policy.heatmap_cosmos_input_normalization=${HEATMAP_COSMOS_INPUT_NORMALIZATION}"
  "policy.heatmap_latent_scale=${HEATMAP_LATENT_SCALE}"
  "policy.heatmap_latent_offset=${HEATMAP_LATENT_OFFSET}"
  "policy.heatmap_latent_stats_path=${HEATMAP_LATENT_STATS_PATH}"
  "policy.heatmap_scheduler_clip_sample=${HEATMAP_SCHEDULER_CLIP_SAMPLE}"
  "policy.heatmap_distribution_mode=${HEATMAP_DISTRIBUTION_MODE}"
  "policy.heatmap_dsnt_temperature=${HEATMAP_DSNT_TEMPERATURE}"
)

if [[ -n "$HEATMAP_DIFFUSION_FINAL_LOSS_ENABLED" ]]; then
  ARGS+=("policy.heatmap_diffusion_final_loss_enabled=${HEATMAP_DIFFUSION_FINAL_LOSS_ENABLED}")
fi

if [[ -n "$HEATMAP_FINAL_LOSS_TIMESTEP_WEIGHTING" ]]; then
  ARGS+=("policy.heatmap_final_loss_timestep_weighting=${HEATMAP_FINAL_LOSS_TIMESTEP_WEIGHTING}")
fi

if [[ -n "$HEATMAP_XY_LOSS_WEIGHT" ]]; then
  ARGS+=("policy.heatmap_xy_loss_weight=${HEATMAP_XY_LOSS_WEIGHT}")
fi

if [[ -n "$HEATMAP_POINT_NLL_LOSS_WEIGHT" ]]; then
  ARGS+=("policy.heatmap_point_nll_loss_weight=${HEATMAP_POINT_NLL_LOSS_WEIGHT}")
fi

if [[ -n "$HEATMAP_JS_LOSS_WEIGHT" ]]; then
  ARGS+=("policy.heatmap_js_loss_weight=${HEATMAP_JS_LOSS_WEIGHT}")
fi

if [[ "$MAX_TRAIN_STEPS" != "0" ]]; then
  ARGS+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
fi

if [[ -n "$RESUME_EPOCH" ]]; then
  ARGS+=("training.resume_epoch=${RESUME_EPOCH}")
fi

if [[ -n "$TEMPORAL_HEATMAP_MODE" ]]; then
  ARGS+=("task.temporal_heatmap_mode=${TEMPORAL_HEATMAP_MODE}")
fi

if [[ -n "$TEMPORAL_HEATMAP_WINDOW_RADIUS" ]]; then
  ARGS+=("task.temporal_heatmap_window_radius=${TEMPORAL_HEATMAP_WINDOW_RADIUS}")
fi

if [[ -n "$TEMPORAL_HEATMAP_BETA" ]]; then
  ARGS+=("task.temporal_heatmap_beta=${TEMPORAL_HEATMAP_BETA}")
fi

if [[ -n "$TEMPORAL_HEATMAP_SIGMA_PX" ]]; then
  ARGS+=("task.temporal_heatmap_sigma_px=${TEMPORAL_HEATMAP_SIGMA_PX}")
fi

if [[ -n "$TEMPORAL_HEATMAP_CURRENT_WEIGHT" ]]; then
  ARGS+=("task.temporal_heatmap_current_weight=${TEMPORAL_HEATMAP_CURRENT_WEIGHT}")
fi

HYDRA_FULL_ERROR=1 "$ACCELERATE_BIN" "${ARGS[@]}"
