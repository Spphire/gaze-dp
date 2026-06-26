# Gaze-WAM

Gaze-conditioned manipulation policy that co-trains on **action-free egocentric
gaze data** (open datasets such as HOT3D) and **recorded robot demonstrations**.
The two sources are merged into one mixed batch:

- **open rows** (no robot action) supervise a **gaze / heatmap loss** — where the
  agent should look;
- **robot rows** supervise an **action loss** — what the arm should do.

The backbone is a FastWAM-style **shared-KV dual-stream transformer**: a world
(image + gaze) tower is prefilled once into a per-layer K/V cache, which is then
consumed by two separate target decoders — an **action** decoder and a
**gaze-heatmap** decoder. Per-sample masks route each loss to the rows that
actually carry that supervision, so gaze that is fed as a condition is never also
used as a target (no label leakage).

> Status: the **open-only** path (HOT3D → zarr → train/eval) is runnable today.
> Robot data collection is not started yet, so the robot/action side is wired but
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
docs/                    # gaze_wam_test_guide_zh.md, experiment reports
```

## Install

```shell
conda env create -f conda_environment.yaml
```

## Data (open side)

HOT3D is converted to the open-gaze zarr contract with the scripts under
`scripts/`, e.g. `preprocess_hot3d_aria.py`,
`convert_hot3d_processed_to_open_zarr.py`, `prepare_open_gaze_wam_zarr.py`.
Inspect / validate a built zarr with `scripts/inspect_gaze_wam_zarr.py` and
`scripts/validate_gaze_wam_zarr.py`.

## Train

Single GPU (Hydra workspace via `train.py`):

```shell
python train.py --config-name train_gaze_wam_open_only_workspace
```

Multi-GPU (8×): configure `accelerate` (see `accelerate/8gpu-amp.yaml`) and use the
launch scripts in `train_scripts/`, e.g. `train_gaze_wam_open_only_8gpu_amp.sh`.

Gaze-loss ablation variants live in
`diffusion_policy/config/train_gaze_wam_open_only_cosmos_temporal_*_workspace.yaml`
(`mixed_loss`, `mixed_nll`, delta-noise, …).

## Evaluate / compare

```shell
python scripts/eval_gaze_wam_metrics.py        # gaze-prediction metrics on a zarr
python scripts/compare_gaze_wam_ablation_metrics.py
```

## Tests & preflight

See [docs/gaze_wam_test_guide_zh.md](docs/gaze_wam_test_guide_zh.md) for the
shortest path to confirm the code compiles, key unit tests pass, and
`scripts/preflight_gaze_wam.py` runs.

## Acknowledgement

Forked from the Data Scaling Laws / UMI diffusion-policy codebase; the
simulation, UMI teleop, and SLAM data-collection tooling have been removed to
focus the repo on the Gaze-WAM training core.
