# Gaze-WAM 单点监督到多峰热力图研究备忘

Date: 2026-06-12

## 结论先行

一图一个注意点的数据不能直接证明“这张图还有哪些其它峰”。如果把标注点固定变成单峰
Gaussian，再用 MSE/JS/latent MSE 训练，模型学到的目标天然就是单峰或单峰的平滑版本。要让模型
从单点数据学出复杂形状热力图，需要把任务从“回归一个 Gaussian 标签”改成“学习条件密度
`p(gaze | image)` 或一个带先验的 heatmap 分布”。

可落地路线是：

1. 短期：把监督换成 point-process NLL，不再构造固定 Gaussian 标签。
2. 中期：用 episode 时序邻域、光流/跟踪、对象分割或 affordance prior，把一帧一个点扩展成弱监督
   pseudo density。
3. 长期：用 latent diffusion / DiT 生成 heatmap 样本，但训练目标仍然要通过点似然、分布匹配和先验
   约束来锚定，否则多峰不可辨识。

当前代码里的 `CachedDualStreamGazeWamTransformer` 已经接近理想架构：先把 image/gaze condition
预填成 world K/V cache，再让 action stream 和 heatmap stream 分别查询同一份 per-layer cache。
因此主要问题不在“能不能共享 KV cache”，而在 heatmap 的学习目标仍然把单点压成单峰 Gaussian 或
单一 latent label。

## 相关文献脉络

### 点标注到密度图

- Bayesian Loss for Crowd Count Estimation with Point Supervision, ICCV 2019:
  https://arxiv.org/abs/1908.03684
  该文指出 crowd counting 中常见的“点标注 -> Gaussian density map”会因为遮挡、尺度、形状变化而
  产生不可靠的逐像素监督；它改用 point annotation 下的概率贡献/期望约束。对 Gaze-WAM 的启发是：
  单点不必被硬编码成固定 sigma 的 Gaussian。

- DM-Count: Distribution Matching for Crowd Counting, NeurIPS 2020:
  https://arxiv.org/abs/2009.13077
  该文用 optimal transport 比较预测 density 和点标注形成的分布，并明确认为强行 Gaussian smoothing
  会伤害泛化。对 gaze heatmap 的启发是：可以让预测热力图保持复杂形状，只用 OT/TV/总质量约束把
  分布和弱标注对齐。

### 点标注到区域/形状

- What's the Point: Semantic Segmentation with Point Supervision, 2015:
  https://arxiv.org/abs/1506.02106
  单点监督配合 objectness potential 能显著强于 image-level 标签。启发是：点本身不足以给出形状，
  但图像中的对象性、边界、语义一致性可以作为形状先验。

- Pointly-Supervised Instance Segmentation, 2021:
  https://arxiv.org/abs/2104.06404
  少量点监督可以训练实例分割，但它通常还使用 box 或多点。启发是：如果要从 gaze point 扩到复杂
  shape，最好引入对象 proposal、SAM/Mask2Former 伪 mask、或手/物体接触区域等结构先验。

### 注视/显著性作为条件密度

- DeepGaze II, 2016:
  https://arxiv.org/abs/1610.01563
  saliency prediction 本质是预测 fixations 的空间概率分布，并使用物体识别网络特征。对当前任务的
  启发是：把每个 gaze point 看成从某个分布采到的样本，而不是把它当成完整热力图本身。

### 多解/多峰输出

- Probabilistic U-Net, 2018:
  https://arxiv.org/abs/1806.05034
  针对 ambiguous segmentation 学习条件分布而不是单一 mask。启发是：复杂 heatmap 应当是
  `p(heatmap | image)`，可以采样多个 plausible heatmaps。

- Diffusion Models for Implicit Image Segmentation Ensembles, 2021:
  https://arxiv.org/abs/2112.03145
  用 diffusion 在给定图像条件下产生 segmentation distribution 和 uncertainty map。启发是：
  heatmap diffusion 适合表达不确定性和多峰，但必须有点似然/先验来防止任意多样性。

- TNT / PECNet / multimodal trajectory prediction:
  https://arxiv.org/abs/2008.08294
  https://arxiv.org/abs/2004.02025
  这些工作把多模态未来表示为目标点/endpoint 分布，再条件生成轨迹。启发是：gaze heatmap 可先预测
  多个候选 attention targets 或 target density，再生成连续形状。

### Diffusion / DiT / cache

- DDPM:
  https://arxiv.org/abs/2006.11239
- DDIM:
  https://arxiv.org/abs/2010.02502
- Latent Diffusion:
  https://arxiv.org/abs/2112.10752
- DiT:
  https://arxiv.org/abs/2212.09748

这些文献支持当前“在 compact latent/token space 里 denoise，而不是直接像素回归”的方向。

- Fast-WAM, 2026:
  https://arxiv.org/abs/2603.16666
  该文的核心问题是 world/video co-training 的收益是否需要 test-time future imagination。它支持
  “训练时用 world branch 建表示，推理时跳过未来生成”的设计方向。

- AHA-WAM, 2026:
  https://arxiv.org/abs/2606.09811
  该文使用双 DiT 和 rolling key-value memory，让 action DiT 读取可复用的 world context。它和
  Gaze-WAM 当前 cached-dual-stream 设计高度同构。

## 为什么单点 Gaussian 训练不够

当前常规路径类似：

```text
gaze_xy -> fixed Gaussian heatmap -> Cosmos latent label -> latent denoising/MSE or decoded DSNT/JS
```

这会带来三个问题：

1. 标签把不确定性抹掉了：一个点被解释成完整真值，而不是一次注视采样。
2. sigma 人工决定形状：模型没有机会学习对象边界、可操作区域或多目标候选。
3. 单样本条件密度不可辨识：同一张图只有一个点时，所有未标出的峰都没有直接证据。

因此，想要多峰，不能只把 backbone 换成 diffusion。必须改变监督语义。

## 推荐训练目标

### 1. Point-process NLL 基线

模型输出 full-resolution logits `z`，转成归一化空间分布：

```text
p_theta(u | I) = softplus(z_u) / sum_v softplus(z_v)
L_point = -log p_theta(y | I)
```

其中 `y` 是标注 gaze point 所在像素或双线性采样位置。这个 loss 只要求标注点概率高，不要求其它位置
为 0，因此允许多峰。

建议加：

```text
L = L_point
  + lambda_tv * TV(p_theta)
  + lambda_prior * KL(p_theta || p_prior)
  + lambda_mass * calibration_regularizer
```

`p_prior` 可以来自 saliency model、SAM/object mask、hand/object proximity、episode temporal prior。
不要简单最大化 entropy，否则会变成到处亮；应控制 entropy 在合理区间，或者用 prior 约束。

### 2. Episode temporal pseudo density

HOT3D/open 数据是 video episode，不是一堆完全独立图片。这是最有价值的弱监督来源。

对每个 frame `t`：

1. 取邻域 `t-k ... t+k` 的 gaze points。
2. 用 optical flow / homography / object tracker 把邻域点 warp 到当前帧。
3. 聚合成 pseudo density `q_t`，允许多峰或拉长形状。
4. 用 `JS(p_theta, q_t)`、OT/Sinkhorn、或 point NLL + prior 训练。

这条路线能把“一帧一个点”变成“同一局部场景的多次注视样本”，比凭空生成多峰更可辨识。

### 3. Object/affordance prior

如果 gaze 常落在可操作对象或接触区域，单点可以扩散到对象/部件层级：

```text
q_t(u) ∝ exp(-d(u, point)^2 / sigma^2) * object_mask(u) * hand_object_affordance(u)
```

对象 mask 可以来自 SAM/grounded segmentation；affordance prior 可以来自手部位置、工具/杯子/按钮等
类别、深度边界、或动作成功片段。这样可学出“围绕物体的一片区域”而不是单点 Gaussian。

### 4. Generative heatmap diffusion

让 heatmap branch 生成多个样本 `H_1...H_K`：

```text
L_point_sample = -log mean_k p_theta_k(y | I)
L_div = repulsion(H_i, H_j) or diversity-on-mean
L_prior = distribution/shape prior
```

注意：仅有 `-log mean_k p_k(y)` 会让某个样本覆盖点，其它样本可能漂移，所以必须加入 saliency/object/
temporal prior 或 group-level distribution matching。这里 diffusion 的角色是表达多样性，不是替代监督。

### 5. 多候选 target heads + heatmap renderer

先预测 `M` 个 candidate gaze/region anchors 和权重：

```text
{(mu_m, Sigma_m, weight_m)}_{m=1..M}
```

再用 differentiable renderer 合成 heatmap。训练用 point NLL 或 mixture NLL。优点是可解释、稳定；
缺点是复杂形状依赖 renderer 表达力。可以作为 diffusion heatmap 前的强基线。

## 推荐架构

当前代码已经实现核心 cache 形态：

- `diffusion_policy/model/gaze_wam/cached_dual_stream_transformer.py`
  - `prefill_world_cache(image_tokens, gaze_token)` 生成 per-layer image/gaze world K/V。
  - `action_blocks` 查询同一份 `world_cache.key_values`。
  - `heatmap_blocks` 也查询同一份 `world_cache.key_values`。
  - `attention_contract_summary()` 标明 `shared_world_kv_cache=True`。

建议保留这个结构，把 supervision 改为：

```text
image frames + optional gaze/mask token
  -> DINO / visual encoder
  -> world/context DiT prefill
  -> world K/V cache

heatmap DiT:
  noisy heatmap latent tokens + timestep
  -> mixed attention over world K/V + heatmap self K/V
  -> heatmap latent / logits
  -> point-process NLL + weak distribution priors

action DiT:
  noisy action tokens + timestep
  -> mixed attention over same world K/V + action self K/V
  -> action chunk
```

Action inference 仍然可以只跑：

```text
world cache once -> action denoising steps
```

不需要 decode heatmap，也不需要 action 读取 noisy heatmap target。

## 实验阶梯

1. **Point NLL direct head**
   - 不用 Gaussian target。
   - 输出 `256 x 256` density 或 Cosmos-decoded density。
   - 指标：point NLL、AUC/NSS、top-k hit、entropy、calibration。

2. **Temporal pseudo density**
   - 先不用光流，只用相邻帧 gaze 的小窗口 KDE 做弱标签。
   - 再加 flow/track warp。
   - 指标：是否出现合理多峰，是否减少固定偏移。

3. **Object-mask prior**
   - 用 SAM 或已有对象/手部先验，把点扩展到对象区域。
   - 指标：热图是否贴合物体，不再出现 scene-texture/checkerboard 伪影。

4. **Latent diffusion + point likelihood**
   - 保留当前 cached-dual-stream DiT。
   - 将 loss 从 latent MSE/固定 Gaussian JS 改成 decoded density 的 point NLL + temporal/object prior。
   - 多采样评估 diversity/fidelity。

5. **Action co-training**
   - Heatmap branch 作为 world representation auxiliary。
   - 验证 `action_reads_heatmap=False` 但 `shared_world_kv_cache=True`。
   - 比较 action success、latency、heatmap quality。

## 需要避免的坑

- 不要指望单点 Gaussian + latent MSE 自然长出多峰。
- 不要只加 diffusion sampling 而不改监督；那通常只是从单峰标签分布里采样。
- 不要用 entropy 奖励无约束扩散；会得到大面积发亮但无语义的热图。
- 如果继续使用 RGB Cosmos tokenizer，要警惕 heatmap 被解码成纹理/棋盘伪影。长期更适合训练一个
  heatmap/mask 专用 frozen autoencoder，或者使用更适合二值/概率图的 codec。

## 最小实现建议

第一版可以只改 loss，不动 architecture：

1. 在 `GazeWamPolicy` 增加 `heatmap_objective=point_nll`。
2. `predict_heatmap` 解码出 `heatmap_image_logits` 后，用 `softplus/logsumexp` 转成 density。
3. 从 `gaze_xy` 双线性采样 `log p(y)`。
4. 加 `TV` 和可选 entropy band regularizer。
5. 预览保存 entropy、top-k、point probability、argmax distance。

如果这个 baseline 都比 Gaussian/latent MSE 更能贴合复杂区域，再上 temporal pseudo density 和 diffusion
multi-sample objective。
