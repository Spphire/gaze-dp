#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:?Set MODE to single or multinode.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new benchmark output directory.}"
CONFIG_NAME="${CONFIG_NAME:-train_gaze_wam_robot_a_image_only_workspace}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-12}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-512}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"

case "$MODE" in
  single)
    NUM_PROCESSES=8
    ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"
    ;;
  multinode)
    NUM_PROCESSES=16
    ACCELERATE_CONFIG="accelerate/2node-16gpu-deepspeed-bf16.yaml"
    MACHINE_RANK="${MACHINE_RANK:?Set MACHINE_RANK to 0 or 1.}"
    MAIN_PROCESS_IP="${MAIN_PROCESS_IP:?Set MAIN_PROCESS_IP for multinode mode.}"
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
    case "$MACHINE_RANK" in
      0|1) ;;
      *) echo "MACHINE_RANK must be 0 or 1, got: $MACHINE_RANK" >&2; exit 2 ;;
    esac
    ;;
  *) echo "MODE must be single or multinode, got: $MODE" >&2; exit 2 ;;
esac

if (( BENCHMARK_STEPS <= 0 )); then
  echo "BENCHMARK_STEPS must be positive." >&2
  exit 2
fi
if (( WARMUP_STEPS < 0 || WARMUP_STEPS >= BENCHMARK_STEPS )); then
  echo "WARMUP_STEPS must be nonnegative and smaller than BENCHMARK_STEPS." >&2
  exit 2
fi
if (( EFFECTIVE_BATCH_SIZE % NUM_PROCESSES != 0 )); then
  echo "EFFECTIVE_BATCH_SIZE must be divisible by NUM_PROCESSES." >&2
  exit 2
fi
PER_PROCESS_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE / NUM_PROCESSES))

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Missing a Python launcher. Install the research environment first." >&2
  exit 1
fi

ARGS=(
  launch
  --config_file "$ACCELERATE_CONFIG"
)
if [[ "$MODE" == "multinode" ]]; then
  ARGS+=(
    --machine_rank "$MACHINE_RANK"
    --main_process_ip "$MAIN_PROCESS_IP"
    --main_process_port "$MAIN_PROCESS_PORT"
  )
fi
ARGS+=(
  train.py
  "--config-name=${CONFIG_NAME}"
  "hydra.run.dir=${OUTPUT_DIR}"
  "robot_dataloader.batch_size=${PER_PROCESS_BATCH_SIZE}"
  "val_robot_dataloader.batch_size=${PER_PROCESS_BATCH_SIZE}"
  "data_mixing.total_batch_size_per_process=${PER_PROCESS_BATCH_SIZE}"
  "training.num_epochs=1"
  "training.max_train_steps=${BENCHMARK_STEPS}"
  "training.max_val_steps=1"
  "training.val_every=0"
  "training.sample_every=0"
  "training.gdr_every=0"
  "training.require_amp=true"
  "training.resume=false"
  "training.measure_step_performance=true"
  "training.performance_warmup_steps=${WARMUP_STEPS}"
  "checkpoint.save_deepspeed_state=false"
  "checkpoint.save_last_ckpt=false"
  "checkpoint.save_last_snapshot=false"
  "checkpoint.topk.k=0"
  "logging.mode=${LOGGING_MODE}"
)

HF_HUB_OFFLINE=1 WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
  "$PYTHON_BIN" -m accelerate.commands.accelerate_cli "${ARGS[@]}"

if [[ "$MODE" == "single" || "${MACHINE_RANK:-}" == "0" ]]; then
  "$PYTHON_BIN" diffusion_policy/scripts/summarize_gaze_wam_benchmark.py \
    --logs "$OUTPUT_DIR/logs.json.txt"
fi
