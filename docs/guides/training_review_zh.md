# Gaze-WAM 训练流程代码 Review 指南

本文档给合作者用于 review Gaze-WAM 的训练流程代码。目标不是解释整个项目背景，而是帮助 reviewer 沿着真实训练链路检查：配置是否正确进入模型、数据是否按预期混合、loss routing 是否无泄漏、Cosmos heatmap latent 是否按当前 contract 训练、运行产物是否能证明训练健康。

如果你要的是“本地怎么用”和“本地怎么测”，先看：

- [`gaze_wam_local_usage_zh.md`](./gaze_wam_local_usage_zh.md)
- [`gaze_wam_test_guide_zh.md`](./gaze_wam_test_guide_zh.md)

这份文档保留给深度代码 review 和 contract 核对。当前主线是 policy 训练，不包含 runner/部署侧 review。

## 一句话训练链路

当前 open-only 预训练路径：

```text
HOT3D open zarr
-> GazeWamOpenDataset
-> build_gaze_wam_mixed_batch(open only)
-> image tokens + learned [MASK] gaze token
-> cached-dual-stream DiT
-> heatmap latent denoising
-> frozen Cosmos decoder
-> full-resolution DSNT + JS heatmap loss
```

混合/机器人主线在同一套代码中支持：

```text
robot zarr absolute TCP action
-> GazeWamRobotDataset converts to relative 10D action chunk
-> mixed robot/open batch
-> robot rows train action, open rows train heatmap
```

## 建议 Review 顺序

1. 先看配置入口，确认本次训练到底是 open-only、robot-only 还是 robot/open mixed。
2. 再看 dataset 输出的 tensor contract，尤其是 action/gaze/heatmap 的 mask。
3. 再看 batch mixing，确认 open rows 不可能产生 action loss，robot rows 不会错误训练 heatmap。
4. 再看 policy loss，确认 loss mask 与 dataset/mixing 的 contract 一致。
5. 再看 heatmap codec 和 latent normalizer，确认 Cosmos encoder/decoder 冻结，且 scale 不裁剪 clean label。
6. 最后看 workspace/Accelerate/checkpoint/preview，确认实际运行证据支持上述 contract。

## 核心文件索引

| 模块 | 文件 | Reviewer 重点 |
| --- | --- | --- |
| 主配置 | `diffusion_policy/config/train_gaze_wam_workspace.yaml` | 默认模型、loss、Cosmos codec、latent scale、AMP、dataloader |
| open-only 配置 | `diffusion_policy/config/train_gaze_wam_open_only_workspace.yaml` | `robot_dataloader.batch_size=0`、`action_loss_weight=0` |
| Cosmos open-only 配置 | `diffusion_policy/config/train_gaze_wam_open_only_cosmos_workspace.yaml` | `heatmap_latent_scale=0.25`、Cosmos JIT 路径 |
| task 配置 | `diffusion_policy/config/task/gaze_wam.yaml` | zarr 字段、动作维度、obs/action horizon、heatmap token 形状 |
| 8 卡启动脚本 | `train_scripts/train_gaze_wam_open_only_8gpu_amp.sh` | Accelerate 配置、Hydra override、AMP 强制开启 |
| tmux 启动脚本 | `train_scripts/start_open_only_8gpu_tmux.sh` | 输出目录、环境变量、console log |
| dataset | `diffusion_policy/dataset/gaze_wam_dataset.py` | open/robot rows 的原始样本 contract |
| batch mixing | `diffusion_policy/dataset/gaze_wam_mixing.py` | `is_open/has_action/has_heatmap/use_gaze_condition` |
| policy | `diffusion_policy/policy/gaze_wam_policy.py` | forward、loss routing、DSNT/JS、Cosmos encode/decode |
| backbone | `diffusion_policy/model/gaze_wam/cached_dual_stream_transformer.py` | token 顺序、attention/cache contract、推理删 heatmap |
| codec | `diffusion_policy/model/gaze_wam/heatmap_decoder.py` | frozen Cosmos CI16x16 wrapper |
| workspace | `diffusion_policy/workspace/train_gaze_wam_workspace.py` | dataloader、Accelerate、normalizer、train/val loop、contract |
| preflight | `diffusion_policy/scripts/preflight_gaze_wam.py` | 本地训练前检查和 loss smoke |
| checkpoint preview | `diffusion_policy/scripts/preview_gaze_wam_checkpoint.py` | ckpt 推理可视化、summary.json |
| watcher | `train_scripts/watch_gaze_wam_ckpt_preview.sh` | 新 ckpt 自动生成 heatmap 对比图 |

## 当前主线 Contract

### 图像与 token

- 输入图像尺寸：`256 x 256`
- 观察帧数：`n_obs_steps=2`
- 每帧图像 token 数：`256`
- 总 image token：`2 * 256 = 512`
- gaze token：`1`
- action horizon：`16`
- action dim：`10 = 3 pos + 6 rot + 1 gripper`
- heatmap token grid：`16 x 16 = 256`
- heatmap latent dim：`16`
- heatmap latent map：`[C=16, H=16, W=16]`

Review 时应确认这些值在 config、policy、training contract 三处一致。

### 当前 open-only 训练

当前服务器参考 run 见 `docs/gaze_wam_server_run_status.md`。截至本文档撰写，active run 是：

```text
/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_scaled_latent_8gpu_amp_20260608_040418
```

这个 run 的关键 contract：

- `canonical_main_config_ok=true`
- `robot_ratio=0.0`
- `open_ratio=1.0`
- `action_loss_weight=0.0`
- `heatmap_loss_weight=1.0`
- `train_action_loss=0.0`
- `train_routing_open_rows=512`
- `train_routing_robot_rows=0`
- `training_contract_num_processes=8`
- effective train batch size per optimizer step：`512`

如果 reviewer 看到 open-only run 中 action loss 非零，或 open rows 被标记为 `has_action=True`，应视为 blocker。

### 混合训练预期

混合训练主线预期是 robot/open 两个 zarr 分开读，然后每步在线拼 batch：

- robot dataloader batch：默认 `48`
- open dataloader batch：默认 `16`
- 比例：`75% robot : 25% open`
- robot rows：训练 action；真实 gaze 输入时默认不训练 heatmap
- robot gaze-dropout rows：可在 `[MASK]` gaze token 下训练 action + heatmap
- open rows：只训练 heatmap，不训练 action

不要把 robot 和 open 预先合并成一个 zarr。review 时应重点确认 `build_gaze_wam_mixed_batch()` 的输出 mask 是否仍表达这个语义。

## 数据 Contract

### Open zarr

open data 只需要训练 heatmap，不需要 action label。关键字段：

- `data/camera0_rgb`
- `data/gaze_xy`
- 可选 `data/gaze_heatmap`，当前主线不依赖
- 可选 `data/has_gaze_label`

要求：

- `gaze_xy` 已在 zarr 阶段归一化到 `[0, 1]`
- open dataset 生成 zero action placeholder
- open row 的 `has_action=False`
- open row 的 `has_heatmap=True`
- open row 的 `use_gaze_condition=False`
- open row 使用 learned `[MASK]` gaze token，而不是把 label gaze_xy 作为输入条件

对应代码：

- `GazeWamOpenDataset.__getitem__`
- `build_gaze_wam_mixed_batch(robot_batch=None, open_batch=...)`

### Robot zarr

robot zarr 预期存完整绝对 TCP 轨迹，读取时才切成 action chunk 并转 relative action。关键字段：

- `data/camera0_rgb`
- `data/gaze_xy`
- `data/action_abs_tcp`
- `data/tcp_pose_abs`
- `data/gripper_width`

要求：

- zarr 中 action 是绝对 TCP 轨迹
- dataset 内部使用当前 obs/action base 计算 relative transform
- 输出 action shape 为 `[action_horizon=48, action_dim=10]`
- robot row 的 `has_action=True`
- robot row 默认 `has_heatmap=False`
- robot row 若 `use_gaze_condition=True`，不应同时训练 heatmap

对应代码：

- `GazeWamRobotDataset._compose_action_abs`
- `absolute_actions_to_relative_actions`
- `GazeWamRobotDataset.__getitem__`

## Loss Routing Review

核心实现位于 `GazeWamPolicy.compute_loss_components()`。

必须保持：

```text
action_loss_mask = (~is_open) & has_action
heatmap_loss_mask = has_heatmap & has_gaze_label   # dsnt_js 主线
```

open-only 训练中：

- `action_loss_mask_count=0`
- `heatmap_loss_mask_count=batch_size`
- `use_gaze_condition=False`
- `is_gaze_condition_dropped=True`
- action head 可参与 forward 的结构约束，但 action loss 必须为 0

混合训练中：

- open rows 的 action placeholder 必须全零
- open rows 不允许 `has_action=True`
- robot rows 不允许 `has_action=False`
- robot rows 若输入真实 gaze，不应有 heatmap loss
- robot gaze dropout rows 才允许 heatmap supervision

Policy 已有 `_validate_loss_batch_contract()` 做硬检查。review 时应确认新增代码没有绕开这个函数，也没有重新引入旧的 `valid_mask` 逻辑。

## Heatmap 训练路径 Review

当前主线是 `heatmap_objective=dsnt_js`。不要把它误读成 token MSE 或 latent MSE。

训练目标生成：

```text
gaze_xy
-> online 256x256 Gaussian target
-> frozen Cosmos CI16x16 encoder
-> raw latent tokens [256, 16]
-> (raw - offset) * scale
-> scheduler-space clean heatmap latent
```

预测监督：

```text
noisy heatmap latent
-> DiT predicts noise/clean depending scheduler target
-> pred clean heatmap latent
-> inverse scale: tokens / scale + offset
-> frozen Cosmos decoder
-> 256x256 heatmap logits/image
-> DSNT xy loss + spatial JS loss
```

必须保持：

- `latent_mse_loss=false`
- `heatmap_token_kl_loss_weight=0.0`
- DSNT/JS 在 full-resolution decoded heatmap 上计算
- Cosmos encoder 用于 label 生成时可以 `no_grad`
- Cosmos decoder 在预测路径不能包 `no_grad`，否则 heatmap DiT 收不到 DSNT/JS 梯度
- Cosmos encoder/decoder 参数本身必须 frozen

对应代码：

- `_target_heatmap_image_from_xy`
- `_heatmap_image_to_training_tokens`
- `_heatmap_tokens_to_spatial_image`
- `CosmosHeatmapCodec.encode_image`
- `CosmosHeatmapCodec.decode_tokens`

## Cosmos Latent Normalizer

当前不要套用 diffusers `AutoencoderKLCosmos` 的 `latents_mean/latents_std`。本项目当前接的是 standalone JIT tokenizer：

```text
nvidia/Cosmos-0.1-Tokenizer-CI16x16
encoder.jit / decoder.jit
```

该包没有可直接用于当前 heatmap chain 的官方 shift/scale。项目使用自己的 HOT3D open heatmap latent 统计：

```text
data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json
```

当前 contract：

- raw min/max：`[-3.921875, 3.375]`
- `abs_max=3.921875`
- `abs_p99.5=3.5`
- `heatmap_latent_scale=0.25`
- `heatmap_latent_offset=0.0`
- scaled observed range：`[-0.98046875, 0.84375]`
- `heatmap_scheduler_clip_sample=true`

Review 时应确认：

- config、Hydra overrides、policy summary、training contract 都是 `0.25/0.0/clip=true`
- scheduler clipping 不会裁掉 clean label
- decoder 接收的是 denormalized raw Cosmos latent
- stats 文件只作为当前链路的统计依据，不是外部通用 normalizer

## Transformer / Attention Contract

当前主线 backbone 是 `cached_dual_stream`。

必须保持：

- `use_block_attention_mask=True`
- condition token 不读 noisy target token
- action 不读 noisy heatmap target token
- heatmap 不读 action token
- fast action inference 可以删除 heatmap target token
- action 可以消费由 image/gaze condition 预填充的 world K/V cache

对应 contract 位于：

- `CachedDualStreamGazeWamTransformer.attention_contract_summary()`

关键字段：

```text
architecture=cached_dual_stream
shared_world_kv_cache=True
world_cache_consumed_by_action=True
action_reads_heatmap_world_cache=True
condition_reads_targets=False
action_reads_heatmap=False
heatmap_reads_action=False
action_inference_drops_heatmap=True
```

Review 风险点：

- 不要让 action 在训练时读取 noisy heatmap target，否则推理删除 heatmap token 会 train-test mismatch
- 不要让 heatmap 读取 action，否则 open-only heatmap 训练会被 dummy action 污染
- 不要让 condition token 读取 target token，否则信息泄漏

## Workspace / 训练循环 Review

核心文件：`diffusion_policy/workspace/train_gaze_wam_workspace.py`

重点检查：

1. `Accelerator(...)` 是否启用 DDP，并且 `training.require_amp=true` 时要求 bf16/fp16。
2. dataloader 是否根据 batch size 决定启用 robot/open。
3. open-only 时 normalizer 是否为 heatmap-only identity action normalizer。
4. robot 训练时 normalizer 是否只从 robot relative actions 拟合。
5. 每个 step 是否调用 `build_gaze_wam_mixed_batch()`，而不是直接把 source batch 喂给 policy。
6. `training_contract.json` 是否在 run 开始写出。
7. `logs.json.txt` 是否记录 routing count、mask count、contract fields。
8. validation 是否使用同一套 batch mixing 和 loss routing。
9. checkpoint 保存频率是否由 `training.checkpoint_every` 控制。
10. preview 是否只在主进程生成。

当前 8 卡 open-only 训练启动脚本：

```bash
bash train_scripts/start_open_only_8gpu_tmux.sh
```

底层训练命令来自：

```bash
bash train_scripts/train_gaze_wam_open_only_8gpu_amp.sh
```

Accelerate 配置：

```text
accelerate/8gpu-amp.yaml
mixed_precision: bf16
num_processes: 8
```

## 运行产物 Review

每个正式 run 至少应检查这些文件：

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
```

`training_contract.json` 应证明：

- `canonical_main_config_ok=true`
- 数据比例符合当前模式
- `heatmap_latent_scale=0.25`
- `heatmap_latent_offset=0.0`
- `heatmap_scheduler_clip_sample=true`
- `heatmap_latent_scaled_range.within_clip=true`
- `heatmap_spatial_decoder=cosmos_tokenizer`
- `heatmap_supervision=full_resolution_dsnt_plus_js_after_frozen_decoder`
- `latent_mse_loss=false`

`logs.json.txt` 应证明：

- open-only：`train_action_loss=0.0`
- open-only：`train_routing_open_rows > 0`
- open-only：`train_routing_robot_rows=0`
- heatmap loss、xy loss、JS loss 都是 finite
- `training_contract_canonical_main_config_ok=1`

checkpoint preview 应检查：

- `summary.json`
- `sample_*/comparison.png`
- `sample_*/pred_heatmap.png`
- `sample_*/target_heatmap.png`
- `sample_*/pred_overlay.png`
- `sample_*/target_overlay.png`

视觉上重点看：

- 是否仍有硬棋盘格/块状边界
- pred heatmap 是否过度贴图像纹理
- pred 与 gaze target 是否至少粗略同区域
- 是否出现边界高亮、整图泛红、全零/全白等退化
- 是否每个 checkpoint 都稳定改善，而不是只有单个样本好看

## 建议 Review 命令

Windows 本地语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  diffusion_policy\workspace\train_gaze_wam_workspace.py `
  diffusion_policy\policy\gaze_wam_policy.py `
  diffusion_policy\model\gaze_wam\heatmap_decoder.py `
  diffusion_policy\model\gaze_wam\cached_dual_stream_transformer.py `
  diffusion_policy\dataset\gaze_wam_dataset.py `
  diffusion_policy\dataset\gaze_wam_mixing.py `
  diffusion_policy\scripts\preflight_gaze_wam.py `
  diffusion_policy\scripts\preview_gaze_wam_checkpoint.py
```

Linux/server 语法检查：

```bash
.venv/bin/python -m py_compile \
  diffusion_policy/workspace/train_gaze_wam_workspace.py \
  diffusion_policy/policy/gaze_wam_policy.py \
  diffusion_policy/model/gaze_wam/heatmap_decoder.py \
  diffusion_policy/model/gaze_wam/cached_dual_stream_transformer.py \
  diffusion_policy/dataset/gaze_wam_dataset.py \
  diffusion_policy/dataset/gaze_wam_mixing.py \
  diffusion_policy/scripts/preflight_gaze_wam.py \
  diffusion_policy/scripts/preview_gaze_wam_checkpoint.py
```

针对性 pytest，优先跑与训练 contract 相关的测试：

```bash
.venv/bin/python -m pytest tests/test_gaze_wam_utils.py -k \
"open_only or dsnt_js or cosmos_heatmap_codec or cached_dual_stream or validation_mixing or training_config" -q
```

open-only preflight 示例：

```bash
.venv/bin/python -m diffusion_policy.scripts.preflight_gaze_wam \
  --config-name train_gaze_wam_open_only_cosmos_workspace \
  --device cuda:0 \
  --output-json data/outputs/preflight_open_only_cosmos.json
```

服务器查看当前训练：

```bash
ssh -p 1024 root@106.14.2.243
cd /mnt/workspace/shenyibo/gaze-wam
tmux capture-pane -pt gaze_wam_open_cosmos_scaled_8gpu -S -80 | tail -80
tail -5 data/outputs/hot3d_open_cosmos_scaled_latent_8gpu_amp_20260608_040418/logs.json.txt
```

查看 training contract：

```bash
python - <<'PY'
import json
p = "data/outputs/hot3d_open_cosmos_scaled_latent_8gpu_amp_20260608_040418/training_contract.json"
d = json.load(open(p))
print("ok:", d["canonical_main_config_ok"])
print("scale:", d["tokens"]["heatmap_latent_scale"])
print("scaled_range:", d["tokens"]["heatmap_latent_scaled_range"])
print("failed checks:", {k: v for k, v in d["checks"].items() if v is not True})
PY
```

## Reviewer Checklist

### 配置

- [ ] 本次训练模式明确：open-only / robot-only / mixed。
- [ ] Hydra config 与 launch script 的 override 一致。
- [ ] `training.require_amp=true`，正式训练使用 bf16/fp16。
- [ ] `heatmap_objective=dsnt_js`。
- [ ] `heatmap_spatial_decoder=cosmos_tokenizer`。
- [ ] `heatmap_latent_scale=0.25`、`offset=0.0`、`clip_sample=true`。
- [ ] open-only 时 `action_loss_weight=0.0`。
- [ ] mixed/robot 训练时 action loss 权重符合实验设计。

### 数据

- [ ] open zarr 不依赖 action label。
- [ ] open `gaze_xy` 已归一化。
- [ ] robot zarr 存绝对 TCP action，dataset 内转 relative action。
- [ ] robot/open 的 obs shape、heatmap placeholder shape 一致。
- [ ] optional presence mask 不会把无效数据误当有效监督。

### Batch / Routing

- [ ] open rows：`is_open=True`、`has_action=False`、`has_heatmap=True`、`use_gaze_condition=False`。
- [ ] robot rows：`is_open=False`、`has_action=True`。
- [ ] open action placeholder 为全零。
- [ ] action loss mask 只覆盖 robot rows。
- [ ] DSNT/JS heatmap loss mask 只覆盖有 gaze label 的 heatmap rows。
- [ ] 没有重新引入 `valid_mask`。

### 模型

- [ ] DINOv3 obs encoder 输出 token 数与 image/token contract 对齐。
- [ ] learned `[MASK]` gaze token 用于 open rows 和 gaze dropout rows。
- [ ] cached-dual-stream attention contract 没有 target leakage。
- [ ] action inference 可以删除 heatmap target token。

### Heatmap

- [ ] target heatmap 在线由 `gaze_xy` 生成。
- [ ] clean heatmap label 经同一个 frozen Cosmos encoder 得到 latent。
- [ ] pred latent 经 inverse scale 后进同一个 frozen Cosmos decoder。
- [ ] decoder 不训练，但预测 decode 路径保留对输入 latent 的梯度。
- [ ] 不使用 latent MSE 代替 full-res DSNT/JS 主损失。

### 运行产物

- [ ] `training_contract.json` 存在且 `canonical_main_config_ok=true`。
- [ ] `logs.json.txt` 中 loss/routing count finite 且符合训练模式。
- [ ] checkpoint watcher 正常运行。
- [ ] 新 checkpoint 有 `summary.json` 和 `sample_*/comparison.png`。
- [ ] 预览图没有明显硬棋盘格或空白退化。

## 常见 Blocker

以下问题建议直接要求修改后再继续训练：

- open rows 参与 action loss。
- robot rows 在真实 gaze 输入时参与 heatmap loss。
- `heatmap_latent_scale` 回退到 `0.2857142857142857` 或 `1.0`，且没有新统计依据。
- `clip_sample=false`，但没有替代的 latent 范围约束。
- Cosmos decoder 被 `torch.no_grad()` 包住导致 DSNT/JS 梯度不能回到 heatmap DiT。
- 自研 heatmap autoencoder 路径重新进入主线。
- `data/gaze_heatmap` 被改成主线必需字段。
- action 从 zarr 读出后没有相对当前 obs/action base 转换。
- attention mask 允许 condition/action/heatmap target 互相泄漏。
- `training_contract.json` 缺失或 `canonical_main_config_ok=false` 且未解释。

## Review 结论模板

建议 reviewer 最后用下面模板给结论：

```text
Review 结论：
- 训练模式：
- 检查的 config：
- 检查的 run/output_dir：
- 数据 contract 是否通过：
- loss routing 是否通过：
- heatmap Cosmos/normalizer 是否通过：
- attention/cache contract 是否通过：
- preflight/pytest/py_compile 结果：
- checkpoint preview 视觉结论：
- blocking issues：
- non-blocking risks：
- 建议下一步：
```

如果没有实际运行证据，只能说“代码静态 review 通过/不通过”，不能声称训练流程已经被端到端验证。
