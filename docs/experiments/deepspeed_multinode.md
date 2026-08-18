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

## Verified throughput benchmark

The benchmark launcher was run from commit `f65d0790db66ab079b59437d137e5884d3f959e0`
with `train_gaze_wam_robot_a_image_only_workspace`, bf16, 12 optimizer steps,
3 warm-up steps, validation/checkpointing disabled, and the same robot-A data
pipeline. The two-node runs used `NCCL_IB_DISABLE=1` and
`NCCL_SOCKET_IFNAME=net0`. The IB/GDRDMA path was not used because it stalls
during ZeRO initialization on this host pair.

The exact launcher shape was:

```bash
cd /mnt/workspace/shenyibo/gaze-proj-deepspeed

# Single node, H200-4102, 8 GPUs, effective batch 512.
MODE=single OUTPUT_DIR=data/outputs/benchmark_single8_f65d079_20260819 \
  EFFECTIVE_BATCH_SIZE=512 BENCHMARK_STEPS=12 WARMUP_STEPS=3 \
  ./train_scripts/benchmark_gaze_wam_distributed.sh

# Two nodes, H200-4102 and H200-4103, 16 GPUs, effective batch 512.
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=net0 MODE=multinode MACHINE_RANK=0 \
MAIN_PROCESS_IP=10.0.8.112 MAIN_PROCESS_PORT=29630 \
OUTPUT_DIR=data/outputs/benchmark_multi16_f65d079_20260819 \
EFFECTIVE_BATCH_SIZE=512 BENCHMARK_STEPS=12 WARMUP_STEPS=3 \
./train_scripts/benchmark_gaze_wam_distributed.sh

# Rank 1 uses the same command and output directory with MACHINE_RANK=1.
# Weak-scaling run: two nodes, 16 GPUs, effective batch 1024, port 29631.
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=net0 MODE=multinode MACHINE_RANK=0 \
MAIN_PROCESS_IP=10.0.8.112 MAIN_PROCESS_PORT=29631 \
OUTPUT_DIR=data/outputs/benchmark_multi16_b1024_f65d079_20260819 \
EFFECTIVE_BATCH_SIZE=1024 BENCHMARK_STEPS=12 WARMUP_STEPS=3 \
./train_scripts/benchmark_gaze_wam_distributed.sh
```

Rank 1 used the same output directory, with `MACHINE_RANK=1`. The measured
steady-state values are:

| run | GPUs | effective batch | transport | steady samples/s | p50 step (s) | p95 step (s) | peak allocated GiB/GPU | peak reserved GiB/GPU |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| single-node DDP | 8 | 512 | local GPU | 1590.98 | 0.3205 | 0.3326 | 40.85 | 42.48 |
| two-node DeepSpeed ZeRO-2 | 16 | 512 | `net0` TCP sockets | 986.996 | 0.5199 | 0.5407 | 20.59 | 21.75 |
| two-node DeepSpeed ZeRO-2 | 16 | 1024 | `net0` TCP sockets | 1615.584 | 0.6351 | 0.6630 | 38.37 | 39.93 |

The raw summaries are:

```text
/mnt/workspace/shenyibo/gaze-proj-deepspeed/data/outputs/benchmark_single8_f65d079_20260819/benchmark_summary.json
/mnt/workspace/shenyibo/gaze-proj-deepspeed/data/outputs/benchmark_multi16_f65d079_20260819/benchmark_summary.json
/mnt/workspace/shenyibo/gaze-proj-deepspeed/data/outputs/benchmark_multi16_b1024_f65d079_20260819/benchmark_summary.json
```

### Interpretation

- At the same effective global batch of 512, the 16-GPU two-node run reaches
  `986.996 / 1590.980 = 0.620` of the single-node throughput. Its step time is
  `1.61x` slower, so this configuration is not a fixed-batch speedup.
- ZeRO-2 does reduce peak per-GPU memory at batch 512 from `40.85` to
  `20.59 GiB` allocated, approximately half, which is the useful result of
  this configuration.
- The batch-1024 run is a weak-scaling probe, not an apples-to-apples speed
  comparison. It reaches `1615.584 samples/s`, only `1.015x` the single-node
  batch-512 throughput while processing twice the global batch. Its memory
  returns close to the single-node level, as expected for the larger local
  batch.
- The current socket transport and ZeRO-2 configuration therefore provides
  memory scaling and checkpoint-correct multi-node execution, but no measured
  training acceleration. Do not move the production branch to this launcher
  based on these results. Diagnose the IB/GDRDMA initialization stall and
  reduce inter-node communication overhead before retesting for speed.

After the benchmark completed, both nodes were checked for residual training
processes and GPU allocations; no benchmark process remained and the GPUs were
free.
