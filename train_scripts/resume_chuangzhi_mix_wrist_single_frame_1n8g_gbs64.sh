#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="mix_wrist_single_frame_temporal_mixed_nll_1n8g_gbs64"
export CONFIG_NAME="train_gaze_wam_chuangzhi_mix_wrist_single_frame_1n8g_gbs64"
export ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"
export EXPECTED_NNODES=1
export EXPECTED_NPROC_PER_NODE=8
export RESUME=true
# This single-node resume is the cache-read smoke test.  Override the path
# when the cache was staged under a different project directory.
export GAZE_WAM_HEATMAP_CACHE_ROOT="${GAZE_WAM_HEATMAP_CACHE_ROOT:-$ROOT/data/heatmap_cache/job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456}"

CONTRACT_REPORT="$ROOT/data/runtime/resume_contract/$TASK_NAME.json"
mkdir -p "$(dirname "$CONTRACT_REPORT")"
"$ROOT/.venv/bin/python" \
  "$ROOT/diffusion_policy/scripts/verify_gaze_condition_contract.py" \
  --dataset-path "$ROOT/data/gaze_wam_robot_20260814_from_162120.zarr" \
  --require-out-of-frame \
  > "$CONTRACT_REPORT"

# Reuse the stopped run so the workspace can find checkpoints/latest.ckpt.
DEFAULT_RESUME_OUTPUT_DIR="$ROOT/data/outputs/chuangzhi/$TASK_NAME/job-0ee2b0f1-13e4-4b28-837f-726f4afef1dc-worker-0_23456"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_RESUME_OUTPUT_DIR}"
if [[ ! -s "$OUTPUT_DIR/checkpoints/latest.ckpt" ]]; then
  echo "Resume checkpoint is missing or empty: $OUTPUT_DIR/checkpoints/latest.ckpt" >&2
  exit 2
fi

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
