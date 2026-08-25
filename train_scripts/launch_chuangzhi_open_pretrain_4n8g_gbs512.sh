#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="open_pretrain_temporal_mixed_nll_4n8g_gbs512"
export CONFIG_NAME="train_gaze_wam_chuangzhi_open_pretrain_4n8g_gbs512"
export ACCELERATE_CONFIG="accelerate/4node-32gpu-deepspeed-bf16.yaml"
export EXPECTED_NNODES=4
export EXPECTED_NPROC_PER_NODE=8
export GAZE_WAM_HEATMAP_CACHE_ROOT="${GAZE_WAM_HEATMAP_CACHE_ROOT:-$ROOT/data/heatmap_cache/job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456}"

manifest="$GAZE_WAM_HEATMAP_CACHE_ROOT/hot3d_open_train/manifest.json"
if [[ ! -s "$manifest" ]]; then
  echo "Required heatmap latent cache manifest is missing or empty: $manifest" >&2
  exit 2
fi

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
