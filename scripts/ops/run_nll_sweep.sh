#!/usr/bin/env bash
# Launch the point-NLL weight sweep: 8 single-GPU runs in parallel, one config
# per physical GPU, each in its own detached tmux session.
#
# Grid (gpu nll js): a 1-D NLL main axis at js=0.1 (gpu 0-5) plus two JS-interaction
# probes at the likely-good nll=0.05 (gpu 6-7).
#
# All runs share: diffusion=1.0, xy=0.0 (multimodal), temporal=bidirectional,
# grad_accum=1, batch 16, STEPS optimizer steps (default 2000 ≈ 35-50 min/run).
#
# Usage:
#   bash scripts/ops/run_nll_sweep.sh            # 2000 steps, train zarr
#   STEPS=5000 bash scripts/ops/run_nll_sweep.sh
#   GRID_ROWS=0-5 bash scripts/ops/run_nll_sweep.sh   # only NLL main axis
#
# Inspect:
#   tmux ls | grep sweep
#   tail -f <sweep-dir>/logs/<name>.log
#   (sweep dir is printed at launch and saved to data/outputs/.last_nll_sweep)
#
# NOTE on GPU pinning: each run uses accelerate/1gpu-amp-inherit.yaml, which has
# NO hard-coded gpu_ids. accelerate otherwise reads gpu_ids:'0' from the config
# and resets CUDA_VISIBLE_DEVICES to physical GPU 0 for every run, stacking all
# 8 jobs onto one GPU. The inherit config respects the per-run
# CUDA_VISIBLE_DEVICES=$gpu set below.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

STEPS=${STEPS:-2000}
ZARR=${ZARR:-data/hot3d_open_train.zarr}
CONFIG=${CONFIG:-train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace}
PYTHON=${PYTHON:-.venv/bin/python}

if [[ ! -d "$ZARR" ]]; then echo "train zarr not found: $ZARR" >&2; exit 1; fi

# gpu  nll    js
ALL_GRID=(
  "0 0.0   0.1"
  "1 0.001 0.1"
  "2 0.01  0.1"
  "3 0.05  0.1"
  "4 0.2   0.1"
  "5 1.0   0.1"
  "6 0.05  0.3"
  "7 0.05  0.0"
)

# Optional subset, e.g. GRID_ROWS=0-5 or GRID_ROWS=0,3,5
SELECT=${GRID_ROWS:-all}

TS=$(date +%Y%m%d_%H%M)
SWEEP_DIR="data/outputs/sweep_nll_$TS"
mkdir -p "$SWEEP_DIR/logs"
echo "$SWEEP_DIR" > data/outputs/.last_nll_sweep

echo "=== NLL sweep $TS ==="
echo "  config: $CONFIG"
echo "  zarr:   $ZARR"
echo "  steps:  $STEPS  (grad_accum=1, batch 16)"
echo "  dir:    $SWEEP_DIR"
echo ""

selected() {
  local idx=$1
  [[ "$SELECT" == "all" ]] && return 0
  if [[ "$SELECT" == *-* ]]; then
    local lo=${SELECT%-*} hi=${SELECT#*-}
    (( idx >= lo && idx <= hi )) && return 0 || return 1
  fi
  [[ ",$SELECT," == *",$idx,"* ]] && return 0 || return 1
}

i=0
for row in "${ALL_GRID[@]}"; do
  read -r gpu nll js <<< "$row"
  if ! selected "$i"; then i=$((i+1)); continue; fi
  i=$((i+1))

  name="nll${nll}_js${js}"
  rundir="$SWEEP_DIR/$name"
  log="$SWEEP_DIR/logs/${name}.log"
  sess="sweep_${name}"
  port=$((29500 + gpu))

  RUN_CMD="CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 PYTHONPATH=$REPO_ROOT \
$PYTHON -m accelerate.commands.launch --config_file accelerate/1gpu-amp-inherit.yaml \
--main_process_port $port train.py --config-name=$CONFIG \
task.open_dataset_path=$ZARR task.robot_dataset_path=null \
policy.heatmap_point_nll_loss_weight=$nll policy.heatmap_js_loss_weight=$js \
training.gradient_accumulate_every=1 training.max_train_steps=$STEPS \
training.num_epochs=1 training.lr_warmup_steps=100 \
training.checkpoint_every=1000 training.val_every=500 training.max_val_steps=8 \
training.sample_every=999999 training.save_val_heatmap_preview=false \
name=sweep_${name} hydra.run.dir=$rundir"

  echo "[gpu $gpu] $name  -> tmux:$sess  log:$log"
  bash "$(dirname "$0")/run_in_tmux.sh" "$sess" "$log" bash -lc "$RUN_CMD"
done

echo ""
echo "=== launched. monitor with: tmux ls | grep sweep ==="
echo "=== sweep dir: $SWEEP_DIR ==="
