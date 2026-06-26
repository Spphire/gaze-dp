# Gaze-WAM 测试指南

这份文档只讲测试，不讲训练背景。
目标是用最短路径确认：代码能编译、关键单测能过、预检能跑、训练阅读性检查能输出。

## 推荐顺序

1. `py_compile`
2. 针对性 `pytest`
3. `preflight_gaze_wam.py`
4. `gaze_wam_smoke_pipeline.py`
5. `review_gaze_wam_training_readiness.py`

## 1. 语法检查

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  diffusion_policy\workspace\train_gaze_wam_workspace.py `
  diffusion_policy\policy\gaze_wam_policy.py `
  diffusion_policy\model\gaze_wam\heatmap_decoder.py `
  diffusion_policy\model\gaze_wam\cached_dual_stream_transformer.py `
  diffusion_policy\dataset\gaze_wam_dataset.py `
  diffusion_policy\dataset\gaze_wam_mixing.py `
  diffusion_policy\scripts\preflight_gaze_wam.py `
  diffusion_policy\scripts\preview_gaze_wam_checkpoint.py `
  diffusion_policy\scripts\preview_gaze_wam_episode.py `
  diffusion_policy\scripts\launch_gaze_wam_training.py `
  diffusion_policy\scripts\review_gaze_wam_training_readiness.py
```

## 2. 针对性 pytest

优先跑训练 contract 相关测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_gaze_wam_utils.py -k `
  "open_only or dsnt_js or cosmos_heatmap_codec or cached_dual_stream or validation_mixing or training_config" -q
```

如果只想快速确认核心工具链，再加上这些：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pose_util.py tests\test_uvc_camera.py tests\test_multi_uvc_camera.py -q
```

## 3. 预检

```powershell
.\.venv\Scripts\python.exe scripts\preflight_gaze_wam.py `
  --config-name train_gaze_wam_open_only_cosmos_workspace `
  --device cpu `
  --output-json data\outputs\preflight_open_only_cosmos.json
```

如果要连带看数据链路，可以先做 onboarding 和 readiness：

```powershell
.\.venv\Scripts\python.exe scripts\review_gaze_wam_data_onboarding.py --help
.\.venv\Scripts\python.exe scripts\review_gaze_wam_training_readiness.py --help
```

## 4. Smoke

```powershell
.\.venv\Scripts\python.exe scripts\gaze_wam_smoke_pipeline.py --help
```

默认 smoke 路线应该能串起 debug 数据、验证、preflight 和预览。
如果 smoke 失败，优先看：

- 数据源是否缺失
- `training_contract.json` 是否没写出
- `preflight` 是否已经报错
- 预览目录里是否没有 `summary.json`

## 5. 常见失败信号

- `canonical_main_config_ok=false`
- `train_action_loss` 在 open-only 里不是 0
- `build_gaze_wam_mixed_batch()` 的 mask 语义被破坏
- checkpoint preview 只有空白图或明显退化图
- `pytest` 只过了部分，但关键 contract 测试失败

## 最小验收标准

本地改动至少要满足：

- `py_compile` 通过
- 相关 `pytest` 通过
- `preflight_gaze_wam.py` 能输出 JSON
- `review_gaze_wam_training_readiness.py` 至少能完整跑通帮助或 dry-run
- 预览目录里能找到 `summary.json` 和对比图
