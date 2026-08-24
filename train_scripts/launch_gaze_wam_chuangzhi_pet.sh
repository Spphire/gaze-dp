#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK_NAME="${TASK_NAME:?Set TASK_NAME in the task-specific wrapper.}"
CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME in the task-specific wrapper.}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:?Set ACCELERATE_CONFIG in the task-specific wrapper.}"
EXPECTED_NNODES="${EXPECTED_NNODES:?Set EXPECTED_NNODES in the task-specific wrapper.}"
EXPECTED_NPROC_PER_NODE="${EXPECTED_NPROC_PER_NODE:?Set EXPECTED_NPROC_PER_NODE in the task-specific wrapper.}"

PET_MASTER_PORT="${PET_MASTER_PORT:?The platform must provide PET_MASTER_PORT.}"
PET_MASTER_ADDR="${PET_MASTER_ADDR:?The platform must provide PET_MASTER_ADDR.}"
PET_NPROC_PER_NODE="${PET_NPROC_PER_NODE:?The platform must provide PET_NPROC_PER_NODE.}"
PET_NNODES="${PET_NNODES:?The platform must provide PET_NNODES.}"
PET_NODE_RANK="${PET_NODE_RANK:?The platform must provide PET_NODE_RANK.}"

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_positive_int "$PET_MASTER_PORT" || (( PET_MASTER_PORT > 65535 )); then
  echo "PET_MASTER_PORT must be an integer in [1, 65535], got: $PET_MASTER_PORT" >&2
  exit 2
fi
if ! is_positive_int "$PET_NPROC_PER_NODE" || ! is_positive_int "$PET_NNODES"; then
  echo "PET_NPROC_PER_NODE and PET_NNODES must be positive integers." >&2
  exit 2
fi
if ! is_nonnegative_int "$PET_NODE_RANK" || (( PET_NODE_RANK >= PET_NNODES )); then
  echo "PET_NODE_RANK must be an integer in [0, $((PET_NNODES - 1))], got: $PET_NODE_RANK" >&2
  exit 2
fi
if [[ "$PET_NNODES" != "$EXPECTED_NNODES" ]]; then
  echo "Task $TASK_NAME requires $EXPECTED_NNODES nodes, but PET_NNODES=$PET_NNODES." >&2
  exit 2
fi
if [[ "$PET_NPROC_PER_NODE" != "$EXPECTED_NPROC_PER_NODE" ]]; then
  echo "Task $TASK_NAME requires $EXPECTED_NPROC_PER_NODE processes per node, but PET_NPROC_PER_NODE=$PET_NPROC_PER_NODE." >&2
  exit 2
fi
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "Accelerate config not found: $ACCELERATE_CONFIG" >&2
  exit 2
fi
if [[ ! -f "diffusion_policy/config/${CONFIG_NAME}.yaml" ]]; then
  echo "Hydra config not found: diffusion_policy/config/${CONFIG_NAME}.yaml" >&2
  exit 2
fi

RESUME="${RESUME:-false}"
case "$RESUME" in
  true|false) ;;
  *) echo "RESUME must be true or false, got: $RESUME" >&2; exit 2 ;;
esac

raw_run_id="${CHUANGZHI_RUN_ID:-${PET_MASTER_ADDR}_${PET_MASTER_PORT}}"
run_id="$(printf '%s' "$raw_run_id" | tr -c 'A-Za-z0-9._-' '_')"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/outputs/chuangzhi/$TASK_NAME/$run_id}"
mkdir -p "$OUTPUT_DIR"

# Keep every runtime-generated file on the project GPFS volume.  The training
# pods have a separate, small /root overlay; leaving these defaults untouched
# can fill that overlay with Triton, TorchInductor, DeepSpeed, HF, or W&B data.
RUNTIME_ROOT="$ROOT/data/runtime/chuangzhi/$TASK_NAME/$run_id/node${PET_NODE_RANK}"
# Keep the base itself short: multiprocessing appends a ~38-character
# ``pymp-*`` socket name, and Linux limits AF_UNIX paths to 108 bytes.
SHORT_TMP_ROOT="$ROOT/../.tmp"
mkdir -p \
  "$RUNTIME_ROOT/home" \
  "$RUNTIME_ROOT/tmp" \
  "$RUNTIME_ROOT/xdg-cache" \
  "$RUNTIME_ROOT/torch-extensions" \
  "$RUNTIME_ROOT/triton" \
  "$RUNTIME_ROOT/torch" \
  "$RUNTIME_ROOT/huggingface" \
  "$RUNTIME_ROOT/wandb" \
  "$RUNTIME_ROOT/mplconfig" \
  "$SHORT_TMP_ROOT"
export HOME="$RUNTIME_ROOT/home"
# Python multiprocessing uses AF_UNIX sockets below the 108-byte path limit;
# keep TMPDIR short while still placing all temporary files on the project disk.
export TMPDIR="$SHORT_TMP_ROOT"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg-cache"
export TORCH_EXTENSIONS_DIR="$RUNTIME_ROOT/torch-extensions"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torch"
export TRITON_HOME="$RUNTIME_ROOT/triton"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export CUDA_CACHE_PATH="$RUNTIME_ROOT/torch/cuda-cache"
export HF_HOME="$RUNTIME_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$RUNTIME_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/huggingface/transformers"
export WANDB_DIR="$RUNTIME_ROOT/wandb"
export WANDB_CACHE_DIR="$RUNTIME_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUNTIME_ROOT/wandb/config"
export PIP_CACHE_DIR="$RUNTIME_ROOT/xdg-cache/pip"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export MPLCONFIGDIR="$RUNTIME_ROOT/mplconfig"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"

NODE_LOG="$OUTPUT_DIR/launcher_node${PET_NODE_RANK}.log"
ENV_LOG="$OUTPUT_DIR/environment_node${PET_NODE_RANK}.txt"
GPU_LOG="$OUTPUT_DIR/nvidia_smi_node${PET_NODE_RANK}.txt"
GPU_MONITOR_LOG="$OUTPUT_DIR/gpu_monitor_node${PET_NODE_RANK}.csv"
GPU_MONITOR_PID_FILE="$OUTPUT_DIR/gpu_monitor_node${PET_NODE_RANK}.pid"
PIP_LOG="$OUTPUT_DIR/pip_freeze_node${PET_NODE_RANK}.txt"
EXIT_LOG="$OUTPUT_DIR/exit_code_node${PET_NODE_RANK}.txt"
CONFIG_LOG="$OUTPUT_DIR/resolved_config_node${PET_NODE_RANK}.yaml"
ENV_SETUP_LOG="$OUTPUT_DIR/environment_setup_node${PET_NODE_RANK}.log"
TIMING_FILE="$OUTPUT_DIR/run_timing.json"
MANIFEST_FILE="$OUTPUT_DIR/launch_manifest.json"

AUTO_SETUP_VENV="${CHUANGZHI_AUTO_SETUP_VENV:-true}"
case "$AUTO_SETUP_VENV" in
  true|false) ;;
  *) echo "CHUANGZHI_AUTO_SETUP_VENV must be true or false." >&2; exit 2 ;;
esac

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
  fi
elif [[ "$AUTO_SETUP_VENV" == "true" ]]; then
  set +e
  "$ROOT/train_scripts/setup_chuangzhi_uv_env.sh" 2>&1 | tee -a "$ENV_SETUP_LOG"
  setup_status=${PIPESTATUS[0]}
  set -e
  if (( setup_status != 0 )); then
    echo "Chuangzhi uv environment setup failed with exit code $setup_status." >&2
    exit "$setup_status"
  fi
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="$ROOT/.venv/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing project venv at $PYTHON_BIN and automatic setup is disabled." >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Offline heatmap token cache generated by the 4-node precompute job.  Keep it
# overridable because a later cache job can be selected without changing code.
export GAZE_WAM_HEATMAP_CACHE_ROOT="${GAZE_WAM_HEATMAP_CACHE_ROOT:-$ROOT/data/heatmap_cache/job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456}"
# Keep production logs small enough to be useful for post-mortem debugging.
# Full distributed/NCCL traces remain available through explicit overrides.
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${CHUANGZHI_NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${CHUANGZHI_NCCL_DEBUG_SUBSYS:-INIT,NET}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-$OUTPUT_DIR/nccl_node${PET_NODE_RANK}_host%h_pid%p.log}"

{
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=$PWD"
  echo "task_name=$TASK_NAME"
  echo "config_name=$CONFIG_NAME"
  echo "accelerate_config=$ACCELERATE_CONFIG"
  echo "python_bin=$PYTHON_BIN"
  echo "python_version=$("$PYTHON_BIN" --version 2>&1)"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_begin"
  git status -sb 2>&1 || true
  echo "git_status_end"
  echo "PET_MASTER_ADDR=$PET_MASTER_ADDR"
  echo "PET_MASTER_PORT=$PET_MASTER_PORT"
  echo "PET_NPROC_PER_NODE=$PET_NPROC_PER_NODE"
  echo "PET_NNODES=$PET_NNODES"
  echo "PET_NODE_RANK=$PET_NODE_RANK"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
  echo "NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>}"
  echo "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-<unset>}"
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<unset>}"
  echo "NCCL_DEBUG=$NCCL_DEBUG"
  echo "NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
  echo "TORCH_DISTRIBUTED_DEBUG=$TORCH_DISTRIBUTED_DEBUG"
  echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING"
  getent hosts "$PET_MASTER_ADDR" 2>&1 || true
  ip -brief address 2>&1 || true
  echo "output_dir=$OUTPUT_DIR"
  echo "runtime_root=$RUNTIME_ROOT"
  echo "HOME=$HOME"
  echo "TMPDIR=$TMPDIR"
  echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "TRITON_HOME=$TRITON_HOME"
  echo "HF_HOME=$HF_HOME"
  echo "WANDB_DIR=$WANDB_DIR"
  echo "GAZE_WAM_HEATMAP_CACHE_ROOT=$GAZE_WAM_HEATMAP_CACHE_ROOT"
  echo "runtime_filesystems"
  df -h "$ROOT" "$RUNTIME_ROOT" "$HOME" "$TMPDIR" /root 2>&1 || true
  echo "runtime_filesystems_end"
} > "$ENV_LOG"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$GPU_LOG" 2>&1 || true
  nvidia-smi topo -m >> "$GPU_LOG" 2>&1 || true
else
  echo "nvidia-smi is unavailable on $(hostname)." > "$GPU_LOG"
fi

# Capture runtime utilization independently of the training logger.  This is
# deliberately a small CSV sampled every few seconds so it survives a sudden
# worker/platform termination without producing another multi-GB debug log.
GPU_MONITOR_INTERVAL_SEC="${CHUANGZHI_GPU_MONITOR_INTERVAL_SEC:-5}"
if ! is_positive_int "$GPU_MONITOR_INTERVAL_SEC"; then
  echo "CHUANGZHI_GPU_MONITOR_INTERVAL_SEC must be a positive integer." >&2
  exit 2
fi
GPU_MONITOR_PID=""
stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
}
trap stop_gpu_monitor EXIT
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    printf '%s\n' 'timestamp_utc,gpu_index,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w'
    while :; do
      timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      sample="$(nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null || true)"
      if [[ -n "$sample" ]]; then
        while IFS= read -r sample_line; do
          [[ -z "$sample_line" ]] && continue
          sample_line="${sample_line//, /,}"
          printf '%s,%s\n' "$timestamp_utc" "$sample_line"
        done <<< "$sample"
      fi
      sleep "$GPU_MONITOR_INTERVAL_SEC"
    done
  ) > "$GPU_MONITOR_LOG" 2>> "$NODE_LOG" &
  GPU_MONITOR_PID=$!
  printf '%s\n' "$GPU_MONITOR_PID" > "$GPU_MONITOR_PID_FILE"
else
  printf '%s\n' 'timestamp_utc,gpu_index,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w' > "$GPU_MONITOR_LOG"
  printf '%s\n' 'nvidia-smi unavailable' >> "$GPU_MONITOR_LOG"
fi
snapshot_uv="${UV_BIN:-$ROOT/.tools/uv}"
if [[ -x "$snapshot_uv" ]]; then
  "$snapshot_uv" pip freeze --python "$PYTHON_BIN" > "$PIP_LOG" 2>&1 || true
else
  "$PYTHON_BIN" -m pip freeze > "$PIP_LOG" 2>&1 || true
fi

# Hydra's config-only mode records the exact inherited values without loading data.
"$PYTHON_BIN" train.py \
  "--config-name=${CONFIG_NAME}" \
  --cfg=job \
  --resolve \
  > "$CONFIG_LOG" 2>&1

if [[ "$PET_NODE_RANK" == "0" ]]; then
  cp "$CONFIG_LOG" "$OUTPUT_DIR/resolved_config.yaml"
  export TASK_NAME CONFIG_NAME ACCELERATE_CONFIG OUTPUT_DIR RESUME
  export PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK
  "$PYTHON_BIN" - "$MANIFEST_FILE" "$TIMING_FILE" <<'PY'
import json
import os
import pathlib
import sys
import time

manifest_path, timing_path = map(pathlib.Path, sys.argv[1:])
manifest = {
    "task_name": os.environ["TASK_NAME"],
    "config_name": os.environ["CONFIG_NAME"],
    "accelerate_config": os.environ["ACCELERATE_CONFIG"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "resume": os.environ["RESUME"],
    "pet": {
        "master_addr": os.environ["PET_MASTER_ADDR"],
        "master_port": int(os.environ["PET_MASTER_PORT"]),
        "nnodes": int(os.environ["PET_NNODES"]),
        "nproc_per_node": int(os.environ["PET_NPROC_PER_NODE"]),
        "world_size": int(os.environ["PET_NNODES"]) * int(os.environ["PET_NPROC_PER_NODE"]),
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
now = int(time.time())
timing = {
    "status": "running",
    "started_at_epoch_seconds": now,
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
}
timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

export TASK_NAME CONFIG_NAME ACCELERATE_CONFIG OUTPUT_DIR RESUME
world_size=$((PET_NNODES * PET_NPROC_PER_NODE))
launch_args=(
  launch
  --config_file "$ACCELERATE_CONFIG"
  --machine_rank "$PET_NODE_RANK"
  --main_process_ip "$PET_MASTER_ADDR"
  --main_process_port "$PET_MASTER_PORT"
  --num_machines "$PET_NNODES"
  --num_processes "$world_size"
  train.py
  "--config-name=${CONFIG_NAME}"
  "hydra.run.dir=${OUTPUT_DIR}"
  "training.resume=${RESUME}"
  "logging.mode=disabled"
)
if (( $# > 0 )); then
  launch_args+=("$@")
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching $TASK_NAME on node rank $PET_NODE_RANK/$PET_NNODES" | tee -a "$NODE_LOG"
set +e
"$PYTHON_BIN" -m accelerate.commands.accelerate_cli "${launch_args[@]}" 2>&1 | tee -a "$NODE_LOG"
launcher_status=${PIPESTATUS[0]}
set -e

{
  echo "exit_code=$launcher_status"
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$EXIT_LOG"

if [[ "$PET_NODE_RANK" == "0" ]]; then
  "$PYTHON_BIN" - "$TIMING_FILE" "$launcher_status" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
exit_code = int(sys.argv[2])
timing = json.loads(path.read_text(encoding="utf-8"))
now = int(time.time())
timing.update({
    "status": "finished" if exit_code == 0 else "failed",
    "finished_at_epoch_seconds": now,
    "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    "elapsed_seconds": now - int(timing["started_at_epoch_seconds"]),
    "launcher_exit_code": exit_code,
})
path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

exit "$launcher_status"
