#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${CONFIGURE_NCCL_TRANSPORT:-false}" == "true" ]]; then
  source "$ROOT/train_scripts/configure_nccl_transport.sh"
  configure_gaze_wam_nccl_transport
fi

# Launch this script once on each host. All ranks must share the same checkout,
# dataset, output directory, and rendezvous values; only MACHINE_RANK differs.
MACHINE_RANK="${MACHINE_RANK:?Set the zero-based host rank.}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:?Set MAIN_PROCESS_IP to the internal address of the rank-0 host.}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
CONFIG_NAME="${CONFIG_NAME:-train_gaze_wam_robot_a_image_only_workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/deepspeed_multinode_smoke}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate/4node-32gpu-deepspeed-bf16.yaml}"
NUM_MACHINES="${NUM_MACHINES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
RESUME="${RESUME:-false}"

if ! [[ "$MACHINE_RANK" =~ ^[0-9]+$ ]] || (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "MACHINE_RANK must be an integer in [0, $((NUM_MACHINES - 1))], got: $MACHINE_RANK" >&2
  exit 2
fi
if ! [[ "$NUM_MACHINES" =~ ^[1-9][0-9]*$ ]] || ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_MACHINES and GPUS_PER_NODE must be positive integers." >&2
  exit 2
fi
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "Accelerate config not found: $ACCELERATE_CONFIG" >&2
  exit 2
fi

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
  --config_file "$ACCELERATE_CONFIG"
  --machine_rank "$MACHINE_RANK"
  --main_process_ip "$MAIN_PROCESS_IP"
  --main_process_port "$MAIN_PROCESS_PORT"
  train.py
  "--config-name=${CONFIG_NAME}"
  "hydra.run.dir=${OUTPUT_DIR}"
  "training.num_epochs=${NUM_EPOCHS}"
  "training.require_amp=true"
  "training.resume=${RESUME}"
  "logging.mode=${LOGGING_MODE}"
)

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  ARGS+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
fi
if [[ -n "$MAX_VAL_STEPS" ]]; then
  ARGS+=("training.max_val_steps=${MAX_VAL_STEPS}")
fi

# Keep the smoke/benchmark independent of online model downloads and W&B.
HF_HUB_OFFLINE=1 WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
  "$PYTHON_BIN" -m accelerate.commands.accelerate_cli "${ARGS[@]}"
