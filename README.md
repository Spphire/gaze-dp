# Gaze-WAM

Gaze-conditioned manipulation policy that co-trains on **action-free egocentric
gaze data** (open datasets such as HOT3D) and **recorded robot demonstrations**.
The two sources are merged into one mixed batch:

- **open rows** (no robot action) supervise a **gaze / heatmap loss** — where
  the agent should look;
- **robot rows** supervise an **action loss** — what the arm should do.

The backbone is a FastWAM-style **shared-KV dual-stream transformer**: a world
(image + gaze) tower is prefilled once into a per-layer K/V cache, which is
then consumed by two separate target decoders — an **action** decoder and a
**gaze-heatmap** decoder. Per-sample masks route each loss to the rows that
actually carry that supervision, so gaze that is fed as a condition is never
also used as a target (no label leakage).

> **Status (2026-06-27):** the **open-only** path (HOT3D → zarr → train) is
> runnable end-to-end and has been smoke-trained on val data. Robot data
> collection is not started yet, so the robot/action side is wired but
> untrained.

## Layout

```
diffusion_policy/
  model/gaze_wam/        # dual-stream transformer, gaze encoder, heatmap codec, losses
  policy/gaze_wam_policy.py
  dataset/               # gaze_wam_dataset.py, gaze_wam_mixing.py (mixed-batch builder)
  workspace/train_gaze_wam_workspace.py
  config/                # train_gaze_wam_*_workspace.yaml, task/gaze_wam.yaml
  common/                # gaze/action utils, pose_util, training-config validation
  real_world/            # gaze_wam_* deployment/inference bindings
scripts/                 # data prep, eval, ablation comparison, preflight (all gaze-wam)
train_scripts/           # 8-GPU / preview launch scripts
accelerate/              # 1gpu-amp.yaml, 8gpu-amp.yaml
docs/                    # see docs/README.md for the index
```

## Setup

Either path works; both produce the same dependency set.

**Conda (canonical):**

```shell
conda env create -f conda_environment.yaml
conda activate umi
pip install -e .
```

**venv (server, reuses system PyTorch):**

```shell
python3 -m venv --system-site-packages .venv
.venv/bin/pip install numpy scipy numba pandas opencv-python pyyaml \
    hydra-core omegaconf einops 'zarr<3' diffusers accelerate av pillow tqdm \
    timm huggingface_hub
export PYTHONPATH=$(pwd):$PYTHONPATH
```

`zarr<3` is required — the data conversion scripts use the zarr v2
`DirectoryStore` API.

## Pretrained weights

Two checkpoints are pulled from Hugging Face on first use:

1. **Cosmos tokenizer** (heatmap latent codec, ~317 MB):

   ```shell
   python diffusion_policy/scripts/download_cosmos_tokenizer.py
   ```

   Writes to `data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/`.

2. **DINOv3 ViT-B/16** (vision backbone, ~85 M params): downloaded
   automatically by `timm` on the first training run. To avoid network calls
   later, set `HF_HUB_OFFLINE=1` after the first successful download.

## Data — open side (HOT3D)

The end-to-end pipeline from raw HOT3D Aria recordings to the open-gaze zarr:

```shell
# 1. (one-off) preprocess HOT3D Aria packages → per-sequence processed/ dir
python scripts/preprocess_hot3d_aria.py ...

# 2. choose train/val split (sorted by sequence id is fine for a first cut)
ls /path/to/HOT3D/processed | sort > data/hot3d_train_sequences.txt
# move a held-out tail to data/hot3d_val_sequences.txt

# 3. convert to zarr (one file per split)
python scripts/convert_hot3d_processed_to_open_zarr.py \
  --processed-root /path/to/HOT3D/processed \
  --output-zarr data/hot3d_open_train.zarr \
  --sequence-file data/hot3d_train_sequences.txt \
  --image-size 224 224 --stride 1 \
  --heatmap-storage token --heatmap-token-grid 16 16 --overwrite

# 4. inspect + validate
python scripts/inspect_gaze_wam_zarr.py --dataset-path data/hot3d_open_train.zarr --dataset-type open
python scripts/validate_gaze_wam_zarr.py --zarr data/hot3d_open_train.zarr
```

Reference sizes (HOT3D 198 sequences, 224×224 RGB, all frames kept):
val (40 seqs, 137 k frames) ≈ 19 GB zarr; train (158 seqs) ≈ 75 GB zarr.

## Preflight

`scripts/preflight_gaze_wam.py` resolves a Hydra config end-to-end, validates
the zarr, and optionally runs a 2-step loss smoke. Always run it on a fresh
machine before launching a real job.

```shell
python scripts/preflight_gaze_wam.py \
  --config-name train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace \
  --override task.open_dataset_path=data/hot3d_open_train.zarr \
  --override task.robot_dataset_path=null \
  --device cuda:0
```

## Train

The training loop **requires AMP** — launch via `accelerate`, not bare
`python`.

**Single GPU smoke (300 steps, mixed_nll):**

```shell
python -m accelerate.commands.launch \
  --config_file accelerate/1gpu-amp.yaml \
  train.py \
  --config-name=train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace \
  task.open_dataset_path=data/hot3d_open_train.zarr \
  task.robot_dataset_path=null \
  training.max_train_steps=300
```

**8-GPU full run:** see `train_scripts/train_gaze_wam_open_only_8gpu_amp.sh`,
which uses `accelerate/8gpu-amp.yaml`.

Gaze-loss ablation variants live in
`diffusion_policy/config/train_gaze_wam_open_only_cosmos_temporal_*_workspace.yaml`
— `mixed_loss`, `mixed_nll`, delta-noise, … see
[docs/experiments/smoke_mixed_nll_diagnostic.md](docs/experiments/smoke_mixed_nll_diagnostic.md)
for the empirical decomposition of which loss components actually contribute.

## Evaluate / compare

```shell
python scripts/eval_gaze_wam_metrics.py            # gaze-prediction metrics
python scripts/compare_gaze_wam_ablation_metrics.py
```

Gaze L2 baselines on HOT3D val (center, random, prev-frame, etc.) are
tabulated in
[docs/experiments/gaze_baseline_hot3d_val.md](docs/experiments/gaze_baseline_hot3d_val.md)
— use those as the lower bound any trained model must beat.

## Documentation

See [docs/README.md](docs/README.md) for the full index. Highlights:

- **Guides** (中文) — local dev, test, training review walk-through
- **Design** — full design plan, single-point-to-multimodal heatmap research
- **Reviews** — expert review (2026-06-07), full-resolution FastWAM review
- **Experiments** — temporal-window report, HOT3D baselines, mixed_nll
  smoke-training diagnostic

## Branch / rollback notes

- `gaze-wam-cleanup` — current working branch, post UMI/SLAM removal.
- `gaze-wam-nll-abtest-sync` — pre-cleanup snapshot (tag
  `rollback/pre-cleanup-20260627` points to the same commit).
- Remote: `gaze-dp` (`git@github.com:Spphire/gaze-dp.git`). `origin` still
  points at the upstream `Data-Scaling-Laws` fork and should not be pushed to.

## Acknowledgement

Forked from the Data Scaling Laws / UMI diffusion-policy codebase; the
simulation, UMI teleop, and SLAM data-collection tooling have been removed to
focus the repo on the Gaze-WAM training core. The pose-math utility module
`diffusion_policy/common/pose_util.py` is the one piece that survived from the
UMI side because it is still used by core training code.
