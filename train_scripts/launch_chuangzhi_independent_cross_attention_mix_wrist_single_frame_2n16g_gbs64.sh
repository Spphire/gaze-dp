#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="independent_cross_attention_mix_wrist_single_frame_temporal_mixed_nll_2n16g_gbs64"
export CONFIG_NAME="train_gaze_wam_chuangzhi_independent_cross_attention_mix_wrist_single_frame_2n16g_gbs64"
export ACCELERATE_CONFIG="accelerate/2node-16gpu-deepspeed-bf16.yaml"
export EXPECTED_NNODES=2
export EXPECTED_NPROC_PER_NODE=8
export RESUME=false
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-data/outputs/chuangzhi/mix_wrist_single_frame_temporal_mixed_nll_1n8g_gbs64/job-0ee2b0f1-13e4-4b28-837f-726f4afef1dc-worker-0_23456/checkpoints/latest.ckpt}"
export GAZE_WAM_HEATMAP_CACHE_ROOT="${GAZE_WAM_HEATMAP_CACHE_ROOT:-$ROOT/data/heatmap_cache/job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456}"

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
