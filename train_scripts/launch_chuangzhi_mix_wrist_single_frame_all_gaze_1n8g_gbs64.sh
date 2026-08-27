#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="chuangzhi_mix_wrist_single_frame_all_gaze_latent_1n8g_gbs64"
export CONFIG_NAME="train_gaze_wam_chuangzhi_mix_wrist_single_frame_all_gaze_1n8g_gbs64"
export ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"
export EXPECTED_NNODES=1
export EXPECTED_NPROC_PER_NODE=8
export RESUME=false
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
export GAZE_WAM_HEATMAP_CACHE_ROOT="${GAZE_WAM_HEATMAP_CACHE_ROOT:-$ROOT/data/heatmap_cache/job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456}"

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
