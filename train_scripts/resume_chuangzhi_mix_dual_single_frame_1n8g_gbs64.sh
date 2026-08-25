#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="mix_dual_shared_vit_temporal_mixed_nll_1n8g_gbs64"
export CONFIG_NAME="train_gaze_wam_chuangzhi_mix_dual_single_frame_1n8g_gbs64"
export ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"
export EXPECTED_NNODES=1
export EXPECTED_NPROC_PER_NODE=8
export RESUME=true
export GAZE_WAM_HEATMAP_CACHE_ROOT="${GAZE_WAM_HEATMAP_CACHE_ROOT:-$ROOT/data/heatmap_cache/job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456}"

for dataset_name in hot3d_open_train gaze_wam_robot_20260814_from_162120_dual_view; do
  manifest="$GAZE_WAM_HEATMAP_CACHE_ROOT/$dataset_name/manifest.json"
  if [[ ! -s "$manifest" ]]; then
    echo "Required heatmap latent cache manifest is missing or empty: $manifest" >&2
    exit 2
  fi
done

# Reuse the stopped run so the workspace can find checkpoints/latest.ckpt.
DEFAULT_RESUME_OUTPUT_DIR="$ROOT/data/outputs/chuangzhi/$TASK_NAME/job-0bc4dab9-08f0-4540-aa4f-8b9855be740d-worker-0_23456"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_RESUME_OUTPUT_DIR}"
if [[ ! -s "$OUTPUT_DIR/checkpoints/latest.ckpt" ]]; then
  echo "Resume checkpoint is missing or empty: $OUTPUT_DIR/checkpoints/latest.ckpt" >&2
  exit 2
fi

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
