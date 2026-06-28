# Gaze 多峰热力图与 point-NLL 权重研究

Date: 2026-06-28
关联：[single_point_to_multimodal_research.md](single_point_to_multimodal_research.md)（2026-06-12 文献备忘）、
[../experiments/smoke_mixed_nll_diagnostic.md](../experiments/smoke_mixed_nll_diagnostic.md)（130 步实测）

## 设计目标（明确）

**即使数据集只有单个 gaze_xy，也要让模型预测出多样化、可多峰的热力图** ——
表达「这张图里有哪些 plausible 的注视区域」的条件密度 `p(gaze | image)`，而不是塌成
一个质心 Gaussian。多峰是**期望行为，不是噪声**。

## 关键洞察：各 loss 分量对"多峰"的态度相反

| Loss | 对多峰 | 结论 |
|---|---|---|
| **DSNT-xy（期望坐标回归）** | ❌ 摧毁多峰。期望=概率加权质心，双峰质心落在两峰**中间空白**，逼模型塌成单峰 | **weight 必须保持 0** |
| **point-NLL（真值点负对数似然）** | ✅ 多峰友好。只要求真值像素概率高，**不惩罚其它峰** | **核心锚定项**（当前 0.001 形同虚设） |
| **JS 散度（分布匹配）** | ✅ 目标多峰则奖励多峰 | 主力塑形项 |
| **diffusion MSE（Cosmos latent）** | ✅ 重建整张 heatmap，支持任意多峰 | backbone |

> **纠正一条早先的建议**：`smoke_mixed_nll_diagnostic.md` 曾建议「加回 xy_loss_weight=0.05」。
> 在多峰目标下这是**错的**——DSNT 坐标回归会主动塌峰。`xy=0` 是正确设计，不是疏忽。

## 当前管线已支持多峰的两个来源

1. **时序融合目标**（dataset `_sample_temporal_heatmap_image`，bidirectional，
   exp(-dt/beta) 衰减，±30 帧 @30fps=±1s）：扫视帧天然产生双峰/多峰目标。
   渲染验证：稳定注视帧单峰干净，扫视帧出现双峰——多峰目标真实存在。
2. **Cosmos latent diffusion**：目标是构造出的多峰 heatmap 图像，diffusion 如实匹配，
   不强制单峰。

所以「能否多峰」不是瓶颈；瓶颈是 **如何在保持多峰的同时，保证真值 gaze 点总被某个峰覆盖**
—— 这正是 point-NLL 权重要回答的问题。

## 中心研究问题

> `heatmap_point_nll_loss_weight` 该设多大？
> 太小（现状 0.001）→ NLL 失效，真值点可能不被覆盖，定位漂；
> 太大 → NLL 主导，可能把分布拉成围绕真值点的单峰，**牺牲多峰**。
> 存在一个 sweet spot：真值覆盖好（argmax_l2 低）且多峰保留（峰数/熵不塌）。

point-NLL 与 JS 的分工：
- JS：把整个多峰分布对上构造的时序融合目标
- point-NLL：保证当前真值 gaze 被锚定为一个峰（不破坏其它峰）

## 对比实验设计（见下方 experiment plan）

主轴：`heatmap_point_nll_loss_weight ∈ {0, 0.001, 0.01, 0.05, 0.2, 1.0}`（固定 js=0.1）
副轴：在候选最优 nll 上探 `js ∈ {0, 0.1, 0.3}` 的交互。

固定：diffusion=1.0、xy=0.0、temporal=bidirectional(radius30/beta10)。

### 评测指标（多峰友好）

- `heatmap_argmax_l2`：最强峰到真值的 L2（**主定位指标**，多峰不惩罚——只看 dominant peak）
- point-NLL on val：真值点似然（覆盖度）
- **多峰度指标（需补）**：预测分布的有效峰数（局部极大值 > 阈值）/ 空间熵 ——
  用来检测「NLL 过大导致塌峰」
- `heatmap_mse` / `heatmap_kl`：辅助

### 判读

- argmax_l2 ↓ 且 峰数/熵 不塌 → 好
- argmax_l2 ↓ 但 峰数→1 / 熵骤降 → NLL 过大，塌峰
- argmax_l2 不降 → NLL 太小或无效

## 后续路线（承接 2026-06-12 备忘）

本实验是「短期：换成 point-process NLL 监督」的定量落地。若 NLL 权重确认有效，
中期可接入 affordance / 手物接触 / SAM 伪 mask 作为形状先验（备忘中期路线），
进一步把弱单点扩成结构化多峰密度。
