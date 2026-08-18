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
rings but stalled during ZeRO optimizer initialization on this pair. The cause
was narrowed to unreliable automatic RoCE GID/HCA selection. Both hosts expose
eight 200-Gbit/s RoCE v2 HCAs, with the usable IPv4 RoCE v2 address at GID
index 3. Explicitly selecting `mlx5_0` through `mlx5_7` and GID 3 removes the
stall and accelerates training. `train_scripts/configure_nccl_transport.sh`
now validates this profile before launch. Use `NCCL_TRANSPORT=socket` only as
the measured fallback, or `NCCL_TRANSPORT=inherit` for an explicitly managed
environment.

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
NCCL_TRANSPORT=roce MACHINE_RANK=0 MAIN_PROCESS_IP=10.0.8.112 \
  OUTPUT_DIR=data/outputs/deepspeed_smoke_$(date +%Y%m%d_%H%M%S) \
  ./train_scripts/train_gaze_wam_deepspeed_multinode.sh

# H200-4103: use the exact same OUTPUT_DIR printed/selected above
cd /mnt/workspace/shenyibo/gaze-proj-deepspeed
NCCL_TRANSPORT=roce MACHINE_RANK=1 MAIN_PROCESS_IP=10.0.8.112 \
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

## Initial socket throughput benchmark

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
- The socket transport and ZeRO-2 configuration therefore provides
  memory scaling and checkpoint-correct multi-node execution, but no measured
  training acceleration. These results motivated the explicit RoCE diagnosis
  and retest below.

After the benchmark completed, both nodes were checked for residual training
processes and GPU allocations; no benchmark process remained and the GPUs were
free.

## Explicit RoCE v2 diagnosis

Commit `3c7922fa584c9ce01c433948bb121d369cee7235` added a small torch-distributed
all-reduce benchmark. It operates only on synthetic GPU tensors and does not
read the dataset or start training. Both hosts reported eight active 200-Gbit/s
Ethernet HCAs. Their stable cross-host RTT was approximately `0.07 ms`, and
GID index 3 was confirmed as the IPv4 `RoCE v2` entry on every HCA.

For the tested host pair, the required environment is:

```bash
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=net0
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
NCCL_IB_GID_INDEX=3
```

The launcher applies and validates these values when
`NCCL_TRANSPORT=roce`, which is the research-branch default. The 256-MiB,
20-step all-reduce results were:

| world size | transport | mean collective (s) | algorithm GB/s | ring bus GB/s |
| ---: | --- | ---: | ---: | ---: |
| 2 | `net0` TCP | 0.11030 | 2.434 | 2.434 |
| 2 | one RoCE rail, GID 3 | 0.01053 | 25.490 | 25.490 |
| 16 | `net0` TCP | 0.09893 | 2.713 | 5.088 |
| 16 | eight RoCE rails, GID 3 | 0.001419 | 189.212 | 354.772 |

Raw outputs:

```text
data/outputs/nccl_tcp2_3c7922f.json
data/outputs/nccl_roce2_gid3_3c7922f.json
data/outputs/nccl_tcp16_3c7922f.json
data/outputs/nccl_roce16_gid3_3c7922f.json
```

The 16-rank RoCE bus bandwidth is about `69.7x` the socket result. This proves
that the high-speed fabric and NCCL/GDRDMA path are functional when the RoCE
address and rails are selected explicitly.

## Accelerated training benchmark

The same model, data pipeline, bf16 mode, effective global batch 512, and
optimizer-step count were then compared with the corrected transport. The
short 12-step run produced `2361.608 samples/s` on 16 GPUs versus
`1590.980 samples/s` on 8 GPUs, a `1.484x` throughput gain while retaining the
ZeRO-2 per-GPU memory reduction from about `40.85` to `20.59 GiB`.

A longer paired run used 100 optimizer steps and excluded 10 warm-up steps:

| run | GPUs | steady steps | total steady time (s) | steady samples/s | p50 step (s) | p95 step (s) | peak allocated GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-node DDP | 8 | 90 | 53.3597 | 863.573 | 0.3362 | 1.4693 | 41.24 |
| two-node ZeRO-2, RoCE v2 | 16 | 90 | 26.1082 | 1764.960 | 0.2271 | 0.6549 | 20.59 |

Artifacts:

```text
data/outputs/benchmark_multi16_roce_gid3_3c7922f_20260819/benchmark_summary.json
data/outputs/benchmark_single8_long_3c7922f_20260819/benchmark_summary.json
data/outputs/benchmark_multi16_roce_long_3c7922f_20260819/benchmark_summary.json
```

The long run completes the same 90 measured optimizer steps `2.044x` faster
end to end. Median step time improves by `1.480x`; the larger end-to-end gain
also reflects the second host sharing data decoding and input work. This is a
real fixed-global-batch training acceleration, not a weak-scaling comparison.

The production `gaze-wam-cleanup` branch remains unchanged. Before adopting
this research launcher for long production jobs, run a checkpoint-enabled
RoCE resume cycle and a multi-epoch stability test; the existing exact-resume
test was performed on the socket fallback before the GID fix.
