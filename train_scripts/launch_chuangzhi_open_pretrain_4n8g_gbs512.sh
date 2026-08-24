#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_NAME="open_pretrain_temporal_mixed_nll_4n8g_gbs512"
export CONFIG_NAME="train_gaze_wam_chuangzhi_open_pretrain_4n8g_gbs512"
export ACCELERATE_CONFIG="accelerate/4node-32gpu-deepspeed-bf16.yaml"
export EXPECTED_NNODES=4
export EXPECTED_NPROC_PER_NODE=8

exec "$ROOT/train_scripts/launch_gaze_wam_chuangzhi_pet.sh" "$@"
