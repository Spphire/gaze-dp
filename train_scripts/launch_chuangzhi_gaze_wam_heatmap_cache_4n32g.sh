#!/usr/bin/env bash
set -euo pipefail

# Chuangzhi one-click launcher for the offline temporal heatmap cache.
# The platform starts this same script on every node and supplies PET_*.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK_NAME="gaze_wam_heatmap_cache_4n32g"
EXPECTED_NNODES=4
EXPECTED_NPROC_PER_NODE=8

PET_MASTER_PORT="${PET_MASTER_PORT:?The platform must provide PET_MASTER_PORT.}"
PET_MASTER_ADDR="${PET_MASTER_ADDR:?The platform must provide PET_MASTER_ADDR.}"
PET_NPROC_PER_NODE="${PET_NPROC_PER_NODE:?The platform must provide PET_NPROC_PER_NODE.}"
PET_NNODES="${PET_NNODES:?The platform must provide PET_NNODES.}"
PET_NODE_RANK="${PET_NODE_RANK:?The platform must provide PET_NODE_RANK.}"

is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

if ! is_positive_int "$PET_MASTER_PORT" || (( PET_MASTER_PORT > 65535 )); then
  echo "PET_MASTER_PORT must be in [1, 65535], got $PET_MASTER_PORT" >&2
  exit 2
fi
if ! is_positive_int "$PET_NPROC_PER_NODE" || ! is_positive_int "$PET_NNODES"; then
  echo "PET_NPROC_PER_NODE and PET_NNODES must be positive integers." >&2
  exit 2
fi
if ! is_nonnegative_int "$PET_NODE_RANK" || (( PET_NODE_RANK >= PET_NNODES )); then
  echo "PET_NODE_RANK must be in [0, $((PET_NNODES - 1))], got $PET_NODE_RANK" >&2
  exit 2
fi
if (( PET_NNODES != EXPECTED_NNODES )); then
  echo "$TASK_NAME requires $EXPECTED_NNODES nodes, got $PET_NNODES" >&2
  exit 2
fi
if (( PET_NPROC_PER_NODE != EXPECTED_NPROC_PER_NODE )); then
  echo "$TASK_NAME requires $EXPECTED_NPROC_PER_NODE GPUs/node, got $PET_NPROC_PER_NODE" >&2
  exit 2
fi

raw_run_id="${CHUANGZHI_RUN_ID:-${PET_MASTER_ADDR}_${PET_MASTER_PORT}}"
run_id="$(printf '%s' "$raw_run_id" | tr -c 'A-Za-z0-9._-' '_')"
OUTPUT_ROOT="${HEATMAP_CACHE_OUTPUT_ROOT:-$ROOT/data/heatmap_cache/$run_id}"
RUNTIME_ROOT="$ROOT/data/runtime/heatmap_cache/$run_id/node${PET_NODE_RANK}"
# Multiprocessing uses AF_UNIX sockets; keep this path below Linux's 108-byte limit.
SHORT_TMP_ROOT="$ROOT/../.tmp"
mkdir -p \
  "$OUTPUT_ROOT" \
  "$RUNTIME_ROOT/home" \
  "$RUNTIME_ROOT/xdg-cache" \
  "$RUNTIME_ROOT/torch" \
  "$RUNTIME_ROOT/triton" \
  "$RUNTIME_ROOT/huggingface" \
  "$SHORT_TMP_ROOT"
export HOME="$RUNTIME_ROOT/home"
export TMPDIR="$SHORT_TMP_ROOT"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg-cache"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torch"
export TRITON_HOME="$RUNTIME_ROOT/triton"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export CUDA_CACHE_PATH="$RUNTIME_ROOT/torch/cuda-cache"
export HF_HOME="$RUNTIME_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$RUNTIME_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/huggingface/transformers"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"
export NCCL_DEBUG="${CHUANGZHI_NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${CHUANGZHI_NCCL_DEBUG_SUBSYS:-INIT,NET}"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-$OUTPUT_ROOT/nccl_node${PET_NODE_RANK}_host%h_pid%p.log}"

NODE_LOG="$OUTPUT_ROOT/launcher_node${PET_NODE_RANK}.log"
ENV_LOG="$OUTPUT_ROOT/environment_node${PET_NODE_RANK}.txt"
EXIT_LOG="$OUTPUT_ROOT/exit_code_node${PET_NODE_RANK}.txt"

{
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "task_name=$TASK_NAME"
  echo "pwd=$PWD"
  echo "PET_MASTER_ADDR=$PET_MASTER_ADDR"
  echo "PET_MASTER_PORT=$PET_MASTER_PORT"
  echo "PET_NPROC_PER_NODE=$PET_NPROC_PER_NODE"
  echo "PET_NNODES=$PET_NNODES"
  echo "PET_NODE_RANK=$PET_NODE_RANK"
  echo "output_root=$OUTPUT_ROOT"
  echo "runtime_root=$RUNTIME_ROOT"
  echo "HOME=$HOME"
  echo "TMPDIR=$TMPDIR"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  git status -sb 2>&1 || true
  getent hosts "$PET_MASTER_ADDR" 2>&1 || true
  df -h "$ROOT" "$OUTPUT_ROOT" "$TMPDIR" /root 2>&1 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 || true
  fi
} > "$ENV_LOG"

AUTO_SETUP_VENV="${CHUANGZHI_AUTO_SETUP_VENV:-true}"
if [[ "$AUTO_SETUP_VENV" == "true" ]]; then
  set +e
  PET_NODE_RANK="$PET_NODE_RANK" PET_NPROC_PER_NODE="$PET_NPROC_PER_NODE" \
    "$ROOT/train_scripts/setup_chuangzhi_uv_env.sh" 2>&1 | tee -a "$OUTPUT_ROOT/environment_setup_node${PET_NODE_RANK}.log"
  setup_status=${PIPESTATUS[0]}
  set -e
  if (( setup_status != 0 )); then
    echo "uv environment setup failed with exit code $setup_status" >&2
    exit "$setup_status"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing project Python: $PYTHON_BIN" >&2
  exit 1
fi

OPEN_TRAIN_ZARR="${OPEN_TRAIN_ZARR:-$ROOT/data/hot3d_open_train.zarr}"
OPEN_VAL_ZARR="${OPEN_VAL_ZARR:-$ROOT/data/hot3d_open_val.zarr}"
ROBOT_ZARR="${ROBOT_ZARR:-$ROOT/data/gaze_wam_robot_20260814_from_162120.zarr}"
ENCODER_PATH="${HEATMAP_COSMOS_ENCODER:-$ROOT/data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit}"
DECODER_PATH="${HEATMAP_COSMOS_DECODER:-$ROOT/data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit}"

DATASET_ARGS=(
  --dataset "open_train=$OPEN_TRAIN_ZARR"
  --dataset "open_val=$OPEN_VAL_ZARR"
  --dataset "robot=$ROBOT_ZARR"
)
if [[ "${HEATMAP_CACHE_SKIP_ROBOT:-false}" == "true" ]]; then
  DATASET_ARGS=(
    --dataset "open_train=$OPEN_TRAIN_ZARR"
    --dataset "open_val=$OPEN_VAL_ZARR"
  )
fi

SAVE_DENSE_ARGS=()
if [[ "${HEATMAP_CACHE_SAVE_DENSE:-false}" == "true" ]]; then
  SAVE_DENSE_ARGS+=(--save-dense)
fi
RESUME_ARGS=()
if [[ "${HEATMAP_CACHE_RESUME:-false}" == "true" ]]; then
  RESUME_ARGS+=(--resume)
fi

world_size=$((PET_NNODES * PET_NPROC_PER_NODE))
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching $TASK_NAME rank=$PET_NODE_RANK/$PET_NNODES world_size=$world_size" \
  | tee -a "$NODE_LOG"
echo "output_root=$OUTPUT_ROOT" | tee -a "$NODE_LOG"
echo "datasets=${DATASET_ARGS[*]}" | tee -a "$NODE_LOG"

set +e
"$PYTHON_BIN" -m accelerate.commands.accelerate_cli launch \
  --config_file accelerate/4node-32gpu-amp.yaml \
  --machine_rank "$PET_NODE_RANK" \
  --main_process_ip "$PET_MASTER_ADDR" \
  --main_process_port "$PET_MASTER_PORT" \
  --num_machines "$PET_NNODES" \
  --num_processes "$world_size" \
  diffusion_policy/scripts/precompute_gaze_wam_heatmap_cache.py \
  --output-root "$OUTPUT_ROOT" \
  --encoder-path "$ENCODER_PATH" \
  --decoder-path "$DECODER_PATH" \
  --batch-size "${HEATMAP_CACHE_BATCH_SIZE:-64}" \
  --latent-scale "${HEATMAP_LATENT_SCALE:-0.25}" \
  --latent-offset "${HEATMAP_LATENT_OFFSET:-0.0}" \
  "${DATASET_ARGS[@]}" \
  "${SAVE_DENSE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$NODE_LOG"
launcher_status=${PIPESTATUS[0]}
set -e

{
  echo "exit_code=$launcher_status"
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$EXIT_LOG"
exit "$launcher_status"
