# DeepSpeed Multi-Node Research

This document describes the experimental launcher on
`research/deepspeed-multinode`. It is deliberately separate from the
production `gaze-wam-cleanup` branch and from existing training output
directories.

## Candidate hosts

The currently verified resume-test pair is:

| machine | internal address | `MACHINE_RANK` |
| --- | --- | ---: |
| H200-4102 | `10.0.8.112` | 0 |
| H200-4103 | `10.0.8.133` | 1 |

The rendezvous host is rank 0. Both hosts must be reachable over TCP port
`29500`, see the same repository and dataset paths, and have the same Python
environment. Access these nodes with `ssh -p 4102 root@106.14.2.243` and
`ssh -p 4103 root@106.14.2.243`. Do not use nodes reserved for other
experiments or any existing Gaze-DP training process.

On 2026-08-19, NCCL's automatically selected IB/GDRDMA transport connected all
rings but stalled during ZeRO optimizer initialization on this pair. The same
run completed over the `net0` socket transport. Until the IB path is diagnosed,
set both `NCCL_IB_DISABLE=1` and `NCCL_SOCKET_IFNAME=net0` on both hosts.

## Configuration

`accelerate/2node-16gpu-deepspeed-bf16.yaml` selects Accelerate's DeepSpeed
backend with 2 machines and 16 total processes. It uses the `standard`
multi-node launcher so both nodes start their own local 8-process group; this
does not depend on a DeepSpeed `pdsh` hostfile. ZeRO stage 2 and its offload
settings are inline in the Accelerate YAML. The top-level `mixed_precision:
bf16` is intentional because the training workspace validates the accelerator
AMP mode; using a separate `deepspeed_config_file` with Accelerate 1.12 would
reject that field as a duplicate.

The training workspace saves two complementary checkpoint forms:

- `checkpoints/latest.ckpt` is the custom `BaseWorkspace` checkpoint used for
  inference, model/EMA restoration, configuration, epoch, and global step. In
  DeepSpeed runs it deliberately excludes the partitioned optimizer state.
- `checkpoints/accelerate_state_step_<step>/` is the native Accelerate and
  DeepSpeed checkpoint used for exact training resume. It contains the model,
  one ZeRO optimizer shard per rank, scheduler, dataloader sampler, and RNG
  state.

Resume first loads the workspace checkpoint to recover the run metadata, then
loads the matching native state directory. The runtime normalizer is rebound
from `normalizer_state.pt` after the native model load. ZeRO-3 remains disabled
until its checkpoint and export semantics receive the same audit.

## Two-node smoke test

Use a new output directory on the shared workspace. Start both commands within
the same rendezvous window:

```bash
# H200-4102
cd /mnt/workspace/shenyibo/gaze-proj-deepspeed
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=net0 \
MACHINE_RANK=0 MAIN_PROCESS_IP=10.0.8.112 \
  OUTPUT_DIR=data/outputs/deepspeed_smoke_$(date +%Y%m%d_%H%M%S) \
  ./train_scripts/train_gaze_wam_deepspeed_multinode.sh

# H200-4103: use the exact same OUTPUT_DIR printed/selected above
cd /mnt/workspace/shenyibo/gaze-proj-deepspeed
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=net0 \
MACHINE_RANK=1 MAIN_PROCESS_IP=10.0.8.112 \
  OUTPUT_DIR=data/outputs/deepspeed_smoke_<run_id> \
  ./train_scripts/train_gaze_wam_deepspeed_multinode.sh
```

For a smoke run, keep `MAX_TRAIN_STEPS=2`, `NUM_EPOCHS=1`, and use a small
debug-compatible config if the full robot dataset is not available. Verify:

1. all 16 ranks initialize and complete the first optimizer step;
2. no NCCL, DeepSpeed, or dataloader deadlock occurs;
3. rank 0 writes `training_contract.json`, logs, and a checkpoint;
4. a second run can load the produced checkpoint with the repository's normal
   trusted-checkpoint path; and
5. the output contains finite losses and matching global step counts.

## Verified native resume

Commit `eb18e2d` completed two consecutive 16-GPU resume runs on H200-4102 and
H200-4103 using:

```text
data/outputs/deepspeed_resume_native_b2080e2_4102_4103
step 4 -> step 5 -> step 6
```

Both runs restored model, ZeRO optimizer, scheduler, dataloader sampler, and
RNG state before completing a forward/backward/optimizer step. Each new native
checkpoint contains 16 optimizer shards. The second run restored
`accelerate_state_step_000005` and wrote `accelerate_state_step_000006` without
the previous missing `camera0_rgb` normalizer error.

This validates checkpoint continuity on the tested socket path. The launcher
remains a research artifact until a single-node versus two-node throughput
benchmark is recorded and the current IB initialization stall is understood.

## Benchmark boundary

Compare a short single-node Accelerate DDP run and this two-node DeepSpeed run
with the same config, effective global batch, number of optimizer steps, and
data-loading settings. Record initialization time, warm-up step time,
steady-state step time, samples/sec, peak memory, and checkpoint time. A lower
step time alone is not sufficient if checkpoint restore, validation, or data
loading regresses.
