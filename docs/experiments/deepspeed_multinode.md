# DeepSpeed Multi-Node Research

This document describes the experimental launcher on
`research/deepspeed-multinode`. It is deliberately separate from the
production `gaze-wam-cleanup` branch and from existing training output
directories.

## Candidate hosts

The first planned pair is:

| machine | internal address | `MACHINE_RANK` |
| --- | --- | ---: |
| H200-4065 | `10.0.8.64` | 0 |
| H200-4066 | `10.0.8.78` | 1 |

The rendezvous host is rank 0. Both hosts must be reachable over TCP port
`29500`, see the same repository and dataset paths, and have the same Python
environment. Do not use nodes reserved for other experiments or any existing
Gaze-DP training process.

## Configuration

`accelerate/2node-16gpu-deepspeed-bf16.yaml` selects Accelerate's DeepSpeed
backend with 2 machines and 16 total processes. It uses the `standard`
multi-node launcher so both nodes start their own local 8-process group; this
does not depend on a DeepSpeed `pdsh` hostfile. ZeRO stage 2 and its offload
settings are inline in the Accelerate YAML. The top-level `mixed_precision:
bf16` is intentional because the training workspace validates the accelerator
AMP mode; using a separate `deepspeed_config_file` with Accelerate 1.12 would
reject that field as a duplicate.

The current training workspace saves a custom `BaseWorkspace` checkpoint, not a
native DeepSpeed checkpoint directory. The smoke test must therefore verify
both optimizer/model stepping and this checkpoint path before a long run is
considered valid. ZeRO-3 is intentionally not enabled until checkpoint restore
and export semantics are audited.

## Two-node smoke test

Use a new output directory on the shared workspace. Start both commands within
the same rendezvous window:

```bash
# H200-4065
cd /mnt/workspace/shenyibo/gaze-proj-deepspeed
MACHINE_RANK=0 MAIN_PROCESS_IP=10.0.8.64 \
  OUTPUT_DIR=data/outputs/deepspeed_smoke_$(date +%Y%m%d_%H%M%S) \
  ./train_scripts/train_gaze_wam_deepspeed_multinode.sh

# H200-4066: use the exact same OUTPUT_DIR printed/selected above
cd /mnt/workspace/shenyibo/gaze-proj-deepspeed
MACHINE_RANK=1 MAIN_PROCESS_IP=10.0.8.64 \
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

Until all five checks pass, this launcher is a research artifact only and must
not replace the existing single-node training launcher.

## Benchmark boundary

Compare a short single-node Accelerate DDP run and this two-node DeepSpeed run
with the same config, effective global batch, number of optimizer steps, and
data-loading settings. Record initialization time, warm-up step time,
steady-state step time, samples/sec, peak memory, and checkpoint time. A lower
step time alone is not sufficient if checkpoint restore, validation, or data
loading regresses.
