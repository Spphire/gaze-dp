# Gaze-WAM 服务器作业 Runbook

实战手册：服务器上从 0 到能开训的所有动作。命令均假设登录 `H200-4042`，
在仓库根目录 `/mnt/workspace/shenyibo/gaze-wam` 下执行。

> 所有 `scripts/ops/*.sh` 都已纳入版本控制，重启服务器 / 重 clone 仓库后
> 仍可立即使用——不要再把它们写在 `/tmp` 里。

## 工具一览

| 脚本 | 作用 | 典型耗时 |
|---|---|---|
| `scripts/ops/prepare_hot3d_zarr.sh` | 把一个 HOT3D split 转成 open-gaze zarr | val 10 min / train ~3 h |
| `scripts/ops/verify_open_zarr.sh` | 完整性校验（.zarray + chunk 数） | < 5 s |
| `scripts/ops/prepare_cosmos_latent_stats.sh` | 估 Cosmos heatmap latent 统计 → JSON | ~1 min（GPU） |
| `scripts/ops/run_in_tmux.sh` | 把任意命令丢到 tmux session detached | — |

## 0. 环境前置（一次）

```bash
# 在仓库根目录
python3 -m venv --system-site-packages .venv
.venv/bin/pip install 'zarr<3' hydra-core omegaconf einops diffusers \
    accelerate av timm huggingface_hub opencv-python tqdm
python diffusion_policy/scripts/download_cosmos_tokenizer.py   # ~317 MB
# DINOv3 ViT-B/16 会在第一次训练时自动从 HF 拉
```

`zarr<3` 必须固定，转换脚本依赖 v2 API。

## 1. 准备 zarr（一键 + 自动校验）

```bash
# val 先跑，10 min；流程没问题再开 train
bash scripts/ops/prepare_hot3d_zarr.sh \
  data/hot3d_val_sequences.txt \
  data/hot3d_open_val.zarr

bash scripts/ops/prepare_hot3d_zarr.sh \
  data/hot3d_train_sequences.txt \
  data/hot3d_open_train.zarr
```

脚本会：
1. 预删旧 zarr 并 `sync` 等文件系统真删完（防止 `--overwrite` race）
2. 跑 `convert_hot3d_processed_to_open_zarr.py`
3. **完整性校验**：检查 7 个字段都有 `.zarray`，chunk 数等于推导值

任何一步失败立刻 exit 非零并打印问题字段。

### 2026-06-27 复盘

那天 train 跑完后 `du -sh` 显示 99G、log 写 "complete"、但 zarr 实际：
- 4 个字段（`gaze_xy` / `camera0_rgb` / `source_sequence_index` / `timestamp_ns`）**缺 `.zarray`**
- `camera0_rgb` 缺 192 个 chunk

根因：脚本启动时上一次的部分 zarr 残留 + `--overwrite` 边删边写产生 race，
Python 进程内部 catch 了 `OSError` 然后继续推进，但已经把目录搞成"半坏"。
`prepare_hot3d_zarr.sh` 通过**显式预删 + sync + verify** 闭环消除这个隐患。

## 2. 估 Cosmos latent stats（一次，val zarr 完成后跑）

```bash
bash scripts/ops/prepare_cosmos_latent_stats.sh \
  data/hot3d_open_val.zarr \
  data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json
```

输出 JSON 里 `scale_recommendations.recommended_default` 应该和
`policy.heatmap_latent_scale`（默认 0.25）一致；不一致时把对应配置文件里
`heatmap_latent_scale` 改成推荐值即可。

HOT3D 224→256 升级后实测 `0.2549 → 0.25 rounded`，正好匹配默认值，**不需要改配置**。

## 3. 用 tmux 跑大任务（推荐做法）

```bash
bash scripts/ops/run_in_tmux.sh \
  train_zarr  data/outputs/logs/train_zarr_$(date +%Y%m%d_%H%M).log \
  bash scripts/ops/prepare_hot3d_zarr.sh \
       data/hot3d_train_sequences.txt \
       data/hot3d_open_train.zarr
```

之后随时：

```bash
tmux ls                                   # 看 session 列表
tmux attach -t train_zarr                 # 进 session，Ctrl-b d 退出
tail -f data/outputs/logs/train_zarr_*.log   # 不进 tmux 也能跟进度

tmux kill-session -t train_zarr           # 强制结束
```

## 4. Preflight（动训前最后一步）

```bash
.venv/bin/python scripts/preflight_gaze_wam.py \
  --config-name train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace \
  --override task.open_dataset_path=data/hot3d_open_train.zarr \
  --override task.robot_dataset_path=null \
  --device cuda:0 \
  --output-json /tmp/preflight.json

.venv/bin/python -c "
import json
r = json.load(open('/tmp/preflight.json'))
print('errors:', r.get('errors'))
print('warnings:', r.get('warnings'))
"
```

期望 `errors: []`、`warnings: []`。如果还报错，按错误信息逐条排查：

- **`Zarr metadata image_size=[H,W] does not match validation image_size=[256,256]`** —
  zarr 转换时 `--image-size` 不是 256×256。重跑 `prepare_hot3d_zarr.sh`，
  默认就是 256×256。
- **`Configured heatmap latent stats file is missing`** — 跑 §2。
- **`Open zarr validation failed ... missing required ... key 'gaze_xy'`** —
  zarr 不完整，跑 `scripts/ops/verify_open_zarr.sh data/hot3d_open_train.zarr`
  看哪个字段坏，重转。

## 5. 启训

单卡 smoke：

```bash
bash scripts/ops/run_in_tmux.sh \
  smoke_mixed_nll  data/outputs/logs/smoke_$(date +%Y%m%d_%H%M).log \
  .venv/bin/python -m accelerate.commands.launch \
    --config_file accelerate/1gpu-amp.yaml \
    train.py \
    --config-name=train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace \
    task.open_dataset_path=data/hot3d_open_train.zarr \
    task.robot_dataset_path=null \
    training.max_train_steps=300
```

8 卡正式训：

```bash
bash scripts/ops/run_in_tmux.sh \
  full_8gpu  data/outputs/logs/full_$(date +%Y%m%d_%H%M).log \
  bash train_scripts/train_gaze_wam_open_only_8gpu_amp.sh
```

## 备注

- 服务器路径：`/mnt/workspace/shenyibo/gaze-wam`
- HOT3D 原始数据：`/mnt/workspace/shenyibo/datasets/HOT3D/processed`
- 8 卡 L20X，每卡 140 GB 显存
- wandb 项目：`cwen/gaze_wam`
