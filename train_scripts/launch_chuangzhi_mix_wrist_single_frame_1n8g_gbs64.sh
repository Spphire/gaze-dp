#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="mix_wrist_single_frame_temporal_mixed_nll_1n8g_gbs64"
export CONFIG_NAME="train_gaze_wam_chuangzhi_mix_wrist_single_frame_1n8g_gbs64"
export ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"
export EXPECTED_NNODES=1
export EXPECTED_NPROC_PER_NODE=8

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
