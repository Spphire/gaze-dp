#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Launch this script once on each host. The two processes must share the same
# checkout, dataset, and output directory; only MACHINE_RANK differs.
MACHINE_RANK="${MACHINE_RANK:?Set MACHINE_RANK to 0 on the rendezvous host or 1 on the second host.}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:?Set MAIN_PROCESS_IP to the internal address of the rank-0 host.}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
CONFIG_NAME="${CONFIG_NAME:-train_gaze_wam_robot_a_image_only_workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/deepspeed_multinode_smoke}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-2}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
RESUME="${RESUME:-false}"

case "$MACHINE_RANK" in
  0|1) ;;
  *) echo "MACHINE_RANK must be 0 or 1, got: $MACHINE_RANK" >&2; exit 2 ;;
esac

case "$RESUME" in
  true|false) ;;
  *) echo "RESUME must be true or false, got: $RESUME" >&2; exit 2 ;;
esac

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
  --config_file accelerate/2node-16gpu-deepspeed-bf16.yaml
  --machine_rank "$MACHINE_RANK"
  --main_process_ip "$MAIN_PROCESS_IP"
  --main_process_port "$MAIN_PROCESS_PORT"
  train.py
  "--config-name=${CONFIG_NAME}"
  "hydra.run.dir=${OUTPUT_DIR}"
  "training.num_epochs=${NUM_EPOCHS}"
  "training.max_train_steps=${MAX_TRAIN_STEPS}"
  "training.max_val_steps=${MAX_VAL_STEPS}"
  "training.require_amp=true"
  "training.resume=${RESUME}"
  "logging.mode=${LOGGING_MODE}"
)

# Keep the smoke/benchmark independent of online model downloads and W&B.
HF_HUB_OFFLINE=1 WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
  "$PYTHON_BIN" -m accelerate.commands.accelerate_cli "${ARGS[@]}"
