# Gaze-WAM 本地开发使用指南

这份文档只负责“本地怎么用”。如果你要看当前服务器上跑到哪一步，读
[`gaze_wam_server_run_status.md`](./gaze_wam_server_run_status.md)。
如果你要做代码链路核对和问题排查，读
[`gaze_wam_training_review_guide_zh.md`](./gaze_wam_training_review_guide_zh.md)。

## 先看什么

1. 先确认仓库根目录是 `W:\实验室项目\gaze-wam`。
2. 再确认虚拟环境 `.venv` 可用。
3. 然后根据目标选入口：
   - 只想看数据是否能进训练链路，跑 `review_gaze_wam_data_onboarding.py`。
   - 只想做本地预检，跑 `preflight_gaze_wam.py`。
   - 想做轻量 smoke，跑 `gaze_wam_smoke_pipeline.py`。
   - 想拉起正式训练，跑 `launch_gaze_wam_training.py` 或 `train_scripts/*`。

## 常用入口

### 数据和预检

```powershell
.\.venv\Scripts\python.exe scripts\review_gaze_wam_data_onboarding.py --help
.\.venv\Scripts\python.exe scripts\review_gaze_wam_training_readiness.py --help
.\.venv\Scripts\python.exe scripts\preflight_gaze_wam.py --help
.\.venv\Scripts\python.exe scripts\gaze_wam_smoke_pipeline.py --help
```

### 训练

```powershell
.\.venv\Scripts\python.exe scripts\launch_gaze_wam_training.py --help
bash train_scripts/start_open_only_8gpu_tmux.sh
bash train_scripts/train_gaze_wam_open_only_8gpu_amp.sh
```

### 预览

```powershell
.\.venv\Scripts\python.exe scripts\preview_gaze_wam_checkpoint.py --help
.\.venv\Scripts\python.exe -m diffusion_policy.scripts.preview_gaze_wam_episode --help
bash train_scripts/start_ckpt_preview_watch_tmux.sh
bash train_scripts/start_episode_preview_watch_tmux.sh
```

## 训练产物

正式 run 通常会产出这些东西：

```text
<output_dir>/.hydra/config.yaml
<output_dir>/.hydra/overrides.yaml
<output_dir>/training_contract.json
<output_dir>/logs.json.txt
<output_dir>/console.log
<output_dir>/normalizer.pkl
<output_dir>/checkpoints/
<output_dir>/media/val_heatmap/
<output_dir>/media/ckpt_heatmap/watched/
<output_dir>/media/episode_heatmap/watched/
```

检查时先看：

- `training_contract.json` 是否存在且 `canonical_main_config_ok=true`
- `logs.json.txt` 是否持续增长
- `checkpoints/` 是否在按预期保存
- `media/ckpt_heatmap/watched/` 和 `media/episode_heatmap/watched/` 是否有新预览

## 当前建议

- 不要把根目录 `README.md` 当成 Gaze-WAM 的操作手册；那是上游仓库说明。
- 如果服务器连不上，先看本地这份文档和最近一次同步过的状态文档。
- 如果你只想判断“能不能上线跑”，先跑测试指南里的预检和 pytest。
