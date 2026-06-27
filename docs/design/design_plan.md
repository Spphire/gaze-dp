# Gaze-WAM Design And Implementation Plan

> **Note (2026-06-27):** parts of this document — particularly UMI alignment
> notes (pose conventions, action utilities, SLAM dataset conversion) — refer
> to the original fork's tooling that has since been removed. The core design
> (mixed-batch routing, dual-stream KV cache, gaze/heatmap loss split) is
> still current. Treat UMI/SLAM mentions as historical rationale, not as a
> blueprint for current code. See the top-level [README.md](../../README.md)
> for the current architecture summary.

## 0. Current Scope

This document is the implementation contract for `gaze-wam`, built on top of
`Fanqi-Lin/Data-Scaling-Laws`.

Current status:

- The original diffusion-policy baseline entrypoints/configs are archived under
  `_archive_datascalinglaws_original_20260606` and are not part of the active training surface.
- Gaze-WAM now owns the active policy-training path through dedicated `gaze_wam` modules,
  dataset adapters, workspace, configs, launch scripts, and tests.
- This document fixes the model IO, data contract, loss routing, heatmap tokenization,
  action representation, training acceleration plan, policy-level action inference path, and
  experiment matrix.
- Any change that affects the agreed contract should be reflected here and in
  `docs/implementation_log.md`.

Current single-contract snapshot:

- Geometry: `256 x 256` RGB input, DINOv3 ViT/16 visual encoder, `16 x 16 = 256`
  visual tokens per frame, and default `T_obs=2`.
- Heatmap: canonical zarr stores normalized `gaze_xy`; dense `gaze_heatmap` is optional
  compatibility/visualization metadata, not required training storage. During training the policy
  generates a normalized `256 x 256` target distribution online from `gaze_xy`, encodes it with the
  frozen NVIDIA Cosmos `Cosmos-Tokenizer-CI16x16` encoder into raw `16 x 16 x 16` latents, maps
  those latents into scheduler space with the project-estimated affine scale, denoises them as
  `16 x 16 = 256` tokens with `heatmap_dim=16`, then denormalizes and decodes predicted clean
  latents through the frozen Cosmos decoder back to `256 x 256` logits for DSNT coordinate loss,
  spatial JS regularization, preview, and evaluation. This follows the FastWAM/Wan convention that
  the generative side denoises pretrained-tokenizer latents, not raw full-resolution pixels:
  `full-resolution heatmap` in Gaze-WAM means the decoded heatmap used for loss, preview, and
  evaluation.
- Action: `T_action=16`, `D_action=10 = 3 pos + 6 rot6d + 1 gripper`; robot zarr stores
  absolute TCP trajectories and the dataset emits relative action targets from the latest
  observed TCP base.
- Routing: mixed batches target `75%` robot and `25%` open-source gaze; robot real-gaze rows
  train action only, robot `[MASK]` dropout rows train action + heatmap, and open-source rows
  train heatmap only. There is no dynamic head freezing in the main method. The same mixed loop
  also supports boundary ratios such as robot-only `100/0` and open-only `0/100` for ablation and
  heatmap-pretraining runs.
- Masking/inference: condition/world tokens do not read target tokens, action and heatmap target
  tokens do not read each other, action does not read noisy heatmap targets, and fast action
  inference drops heatmap target tokens while reusing the heatmap/world K/V cache. Prediction heads
  are separate `LayerNorm + Linear` heads.
- Remaining open items are real data, checkpoint, training-data synchronization, ablations, and
  paper-presentation choices summarized in Section 15, not core model-shape decisions.
  Runner/hardware-specific deployment work is deferred for the current policy-training-only scope.
  Existing deployment adapter, runner, and rehearsal notes are reference utilities only until the
  project moves from policy training to live deployment.
- Training acceleration: AMP is mandatory for non-debug Gaze-WAM training. The default config sets
  `training.require_amp=true`, and the workspace rejects launches whose Accelerate
  `mixed_precision` is not `bf16` or `fp16`; use bf16 first.

Policy-only closeout boundary for this session:

- In scope: canonical robot/open zarr preparation, zarr validation, DINOv3 source verification,
  mixed-batch dataloader checks, policy instantiation, normalizer fitting, one-batch loss smoke,
  `training_contract.json`, validation loss/mask logging, heatmap previews, and offline ablation
  metrics.
- Out of scope: live runner behavior, hardware bindings, command scheduling, deployment rehearsal
  tuning, and real-time control-loop latency. Existing `real_world` and rehearsal utilities may
  remain as reference scaffolding without blocking policy training.
- Closeout criterion for the current milestone: the policy-training contract, data/preflight path,
  and remaining real-data tasks are documented; no runner or deployment code change is required
  before the first real mixed 75/25 policy-training run.
- A first real policy-training launch should be gated by preflight/report artifacts that record
  `policy_training_scope=true`, `deployment_runner_scope="deferred"`, valid robot/open dataloader
  batches, valid attention/routing contracts, and a verified local DINOv3 source.

## 1. Base Repository Understanding

The cloned base repository is a UMI-style fork of `diffusion_policy`.

Important existing pieces:

- `train.py`
  - Hydra entrypoint.
  - Instantiates `cfg._target_` workspace and calls `workspace.run()`.
- `diffusion_policy/workspace/*`
  - Training loops live here.
  - The closest reusable loop is `train_diffusion_transformer_timm_workspace.py`.
- `diffusion_policy/policy/diffusion_transformer_timm_policy.py`
  - Image policy using a transformer diffusion model.
  - Uses `TransformerObsEncoder` to turn observations into visual/low-dim tokens.
  - Uses action-only diffusion loss.
- `diffusion_policy/model/vision/transformer_obs_encoder.py`
  - Can produce ViT token sequences from RGB observations.
  - This is directly useful for the visual condition path.
- `diffusion_policy/model/diffusion/transformer_for_action_diffusion.py`
  - Current transformer diffusion backbone.
  - It uses action tokens as decoder target and observation tokens as memory.
  - Gaze-WAM needs a different joint sequence formulation:
    `[image tokens, gaze token, noisy action tokens, noisy heatmap tokens]`.
- `diffusion_policy/dataset/umi_dataset.py`
  - Existing UMI dataset logic includes pose conversion and action normalization patterns.
  - It does not currently expose gaze coordinates or heatmap targets as first-class outputs.

Reference from `W:/umi_base`:

- `W:/umi_base/diffusion_policy/config/task/q3_place_cup_no_tcp.yaml`
  - Uses image policy task metadata, `shape_meta`, `relative_action: True`,
    `use_absolute_action: True`, and `action_representation: 'relative'`.
- `W:/umi_base/diffusion_policy/common/action_utils.py`
  - Provides `absolute_actions_to_relative_actions` and
    `relative_actions_to_absolute_actions`.
- `W:/umi_base/Makefile`
  - Provides `train_acc8_amp` as the target for 8-GPU bf16 Accelerate training.
- `W:/umi_base/accelerate/8gpu-amp.yaml`
  - Uses `distributed_type: MULTI_GPU`, `num_processes: 8`, and
    `mixed_precision: 'bf16'`.

Architectural conclusion:

- Keep the original policy/workspace untouched for baselines.
- Add a new Gaze-WAM policy/backbone/workspace/config path.
- Port or mirror the UMI relative-action and Accelerate training conventions where appropriate.

## 2. Research Objective

Gaze-WAM is a gaze-conditioned conditional diffusion policy with joint visual representation
learning.

Primary control task:

- Predict a 16-step robot action trajectory.
- Robot data is the only source of action supervision in the main method.

Auxiliary gaze task:

- Predict gaze heatmap tokens from open-source gaze data and selected robot rows.
- Open-source gaze data is always a heatmap-supervised source.
- A configured subset of robot rows also contributes heatmap loss when their gaze condition is
  replaced by the trainable `[MASK]` token.

Key condition:

- Robot branch receives a real human gaze point `(x, y)` as a condition.
- A random subset of robot rows applies gaze-condition dropout: the label gaze point is still
  available for heatmap supervision, but the model receives the trainable `[MASK]` token instead
  of the real gaze condition.
- Open-source branch uses a learned trainable `[MASK]` gaze token as the condition in the main
  method, even if the raw open data includes a gaze point label.
- The open-data gaze point is used to create the heatmap target, not to condition the model in
  the main method.

Key data mixture:

- One optimizer step uses one mixed batch.
- Target composition: `75%` robot samples and `25%` open-source gaze samples.
- The action head is not dynamically frozen.
- Loss masks decide which samples contribute to which loss:
  - Robot samples with real gaze condition: action loss only.
  - Robot samples with gaze-condition dropout: action loss + heatmap loss.
  - Open-source gaze samples: heatmap loss only.

Why robot heatmap loss is gated by gaze-condition dropout:

- If a robot sample inputs the real gaze point as a condition, training it to reconstruct a heatmap
  derived from that same point risks a trivial "copy the gaze coordinate" shortcut.
- If the robot sample receives `[MASK]` instead, heatmap prediction must come from image context,
  while action prediction still receives valid robot supervision.
- This also trains the `[MASK]` token on robot action prediction, making missing-gaze inference
  and classifier-free guidance more meaningful.

## 3. Tensor Contract

### 3.1 Main Batch Contract

The current Gaze-WAM policy accepts mixed batches in this normalized format:

```python
batch = {
    "obs": {
        "camera0_rgb": FloatTensor[B, T_obs, 3, 256, 256],
        # optional low-dim robot state keys can remain here
    },

    # Model target after dataset-side conversion from absolute TCP to relative action.
    "action": FloatTensor[B, T_action, D_action],

    # Optional debug/evaluation/future-deployment metadata for converting relative predictions
    # back to absolute TCP.
    "action_abs": FloatTensor[B, T_action, D_action],       # optional, robot rows only
    "action_base_abs": FloatTensor[B, D_action],            # optional but recommended

    # Raw gaze label point in normalized image coordinates.
    # Corresponds to the latest observation frame by default.
    # If only a dense gaze heatmap exists, fill this with zeros and set has_gaze_label=False.
    "gaze_xy": FloatTensor[B, 2],

    # Compact latent heatmap placeholder/target.
    # Canonical DSNT+JS training overwrites eligible rows online from gaze_xy.
    # With the main 16x16 token grid and heatmap_dim=4 this packs a 32x32
    # latent map as [256, 4] tokens.
    "heatmap": FloatTensor[B, T_heatmap, 256, 4],

    # Optional full-res heatmap for visualization/evaluation/alternate labels.
    "heatmap_image": FloatTensor[B, T_heatmap, 256, 256],

    # Optional presence masks emitted when optional metadata is mixed across sources.
    "has_action_abs": BoolTensor[B],          # optional
    "has_action_base_abs": BoolTensor[B],     # optional
    "has_heatmap_image": BoolTensor[B],       # optional

    # Source and supervision masks.
    "is_open": BoolTensor[B],
    "has_action": BoolTensor[B],
    "has_heatmap": BoolTensor[B],
    "has_gaze_label": BoolTensor[B],
    "use_gaze_condition": BoolTensor[B],
    "is_gaze_condition_dropped": BoolTensor[B],
}
```

Main-method mask convention:

```text
robot row with real gaze condition:
  is_open=False
  has_action=True
  has_heatmap=False
  has_gaze_label=True
  use_gaze_condition=True
  is_gaze_condition_dropped=False

robot row with gaze-condition dropout:
  is_open=False
  has_action=True
  has_heatmap=True
  has_gaze_label=True
  use_gaze_condition=False
  is_gaze_condition_dropped=True

robot row without a usable point-gaze condition but with a heatmap target:
  is_open=False
  has_action=True
  has_heatmap=True when robot_heatmap_on_gaze_dropout=True
  has_gaze_label=False
  use_gaze_condition=False
  is_gaze_condition_dropped=True

open-source gaze row:
  is_open=True
  has_action=False
  has_heatmap=True
  has_gaze_label=True for the canonical xy-supervised DSNT+JS path
  use_gaze_condition=False
  is_gaze_condition_dropped=True
```

Notes:

- Open-source rows can keep the raw gaze point internally for label generation, but
  `use_gaze_condition=False` makes the model consume the learned `[MASK]` token.
- Dense-heatmap-only rows are compatibility/ablation data. They are not part of the current
  canonical DSNT+JS path because the main heatmap objective is supervised from `gaze_xy`.
- If a robot row lacks a usable point-gaze label, it uses `[MASK]` for the gaze condition and does
  not contribute DSNT/JS heatmap loss unless an explicit alternate dense-label objective is enabled.
- Open-source rows use zero-filled placeholder actions for shape consistency, but these rows never
  contribute action loss. The placeholder width must match the configured positive `action_dim`
  (`10` in the main method).
- Robot rows with gaze-condition dropout contribute both action and heatmap losses.
- Robot rows without gaze-condition dropout may still carry optional heatmap visualizations for
  logging, but they do not contribute heatmap loss.
- Optional metadata such as `action_abs`, `action_base_abs`, and `heatmap_image` may exist for only
  one source. The mixed-batch builder zero-fills the missing side and emits `has_*` masks so
  evaluators and future deployment utilities do not treat placeholders as real metadata.
- A source zarr may provide row-level `has_gaze_label`; when it is false, the dataset must not use
  the same row's `gaze_xy` as a condition or DSNT/JS target. The row routes the gaze condition
  through the learned `[MASK]` token.
- If a source batch already provides per-row presence masks such as `has_action_abs`,
  `has_action_base_abs`, or `has_heatmap_image`, the mixed-batch builder preserves those masks
  instead of assuming every row with the optional tensor key is valid. This supports real-data
  adapters that keep optional tensors dense but mark unavailable rows explicitly.
- Rows where these optional `has_*` masks are false must carry zero metadata placeholders after
  mixed-batch construction; source-provided stale values are not allowed to survive into the policy
  loss/evaluation boundary.
- The mixed-batch builder also validates source-provided `has_gaze_label` and optional `has_*`
  masks for manual or custom batches: they must be vector-shaped `BoolTensor` values. Raw zarr
  validation may accept boolean/0-1 arrays for compatibility, but dataset adapters must convert
  those arrays to bool tensors before they enter the mixed-batch training boundary.
- The policy loss-batch boundary validates every observation tensor before normalization/backbone
  execution: observation values must be floating tensors, finite, and batch-aligned with
  `action`. This catches bad real-data image/state tensors before they can poison long-running
  policy training.
- The same policy loss-batch boundary validates optional dense `heatmap_image` metadata when it is
  present. It must be floating, finite, non-negative, batch-aligned, shaped as `[B, H, W]` or
  `[B, 1, H, W]`, and use the configured fixed-codec image size (`256 x 256` in the main method).
  A provided `has_heatmap_image` mask must be a boolean `[B]` tensor and cannot appear without the
  matching dense heatmap tensor.
- Optional action metadata is also checked at the policy loss boundary when present.
  `action_abs` must be `[B, T_action, D_action]`, `action_base_abs` must be `[B, D_action]`, both
  must be floating and finite, and their `has_action_abs` / `has_action_base_abs` masks must be
  boolean `[B]` tensors attached to the matching metadata tensor. Open-source rows must never mark
  these action-metadata masks true, because their action metadata is placeholder-only.
- Rows with `has_action_abs=False`, `has_action_base_abs=False`, or `has_heatmap_image=False` must
  have zero placeholders in the matching optional metadata tensor.
- Open-source rows are the exception for action metadata: even if an open batch accidentally
  provides `action_abs` or `action_base_abs`, the mixed-batch builder zero-fills those optional
  action tensors and sets their `has_*` masks to false. Open `heatmap_image` metadata is still
  preserved for visualization and heatmap metrics.
- Zarr validation also checks optional presence-mask arrays when present. `has_action_abs`,
  `has_action_base_abs`, and `has_heatmap_image` must be `[N]` or `[N,1]` boolean/0-1 arrays with
  first-axis length equal to the zarr step count.
- The zarr dataset adapters emit these optional masks when the arrays exist in the source zarr.
  `has_action_abs` is collapsed over the sampled future action window and is true only if every
  target action row is available; `has_action_base_abs` and `has_heatmap_image` are sampled at the
  current row. This keeps evaluator masks tied to real metadata availability instead of dense
  zero-filled placeholders.
- Robot canonicalization and the robot prepare pipeline preserve same-name optional presence-mask
  arrays when the raw zarr already contains them. The output summary records copied
  `presence_mask_keys`, and validation reports their counts before training.
- Open-source manifest conversion writes `has_heatmap_image=True` for dense heatmap-label exports.
  This makes open heatmap-image availability explicit for preview/evaluation while leaving the
  main heatmap token loss unchanged.

### 3.2 Model Token Contract

Fixed main configuration:

- Image resolution: `256 x 256`.
- Visual encoder: DINOv3 ViT/16 by default.
- Visual token grid per frame: `16 x 16 = 256` patch tokens.
- Heatmap token grid: `16 x 16 = 256` latent tokens.
- Default heatmap token channel dimension: `heatmap_dim=16`, matching the frozen Cosmos
  `Cosmos-Tokenizer-CI16x16` continuous latent map `[C=16, H=16, W=16]`. The heatmap branch
  denoises these scheduler-space latent tokens and the frozen Cosmos decoder maps the
  denormalized clean prediction to full-resolution `256 x 256` logits for loss and preview.
- `image_size` and `heatmap_token_grid` dimensions must be positive integers. The dataset,
  fixed heatmap codec, and zarr validator all fail fast on zero or negative geometry values.
  Dataset constructors also reject boolean, fractional, non-finite, or non-integer string values
  for positive-integer geometry and horizon fields instead of silently truncating them through
  Python `int(...)`; string-form integers such as `"16"` remain accepted for Hydra override
  compatibility. `n_latency_steps` follows the same strict parser but permits zero, because zero
  latency is the first-run default. Dataset-local temporal/geometry validation wrappers delegate
  to the shared Gaze-WAM config normalizers, so dataset construction stays aligned with
  workspace/preflight parsing.
- The fixed heatmap codec applies the same no-boolean/no-fraction geometry rule and also requires
  `sigma_tokens` to be a finite positive float, so label-token construction cannot inherit invalid
  codec geometry from a partially parsed config.
- `HeatmapTokenCodec` and `GaussianSpatialEncoder` local geometry/sigma validators delegate to the
  shared Gaze-WAM config normalizers. This keeps fixed label-token construction and gaze-condition
  encoding aligned with dataset, workspace, preflight, policy, and transformer parsing.
- The policy constructor must pass `heatmap_token_grid` through the fixed codec validator before
  checking its product against `heatmap_num_tokens`; it must not pre-coerce grid entries with
  `int(...)`, because that would silently turn invalid booleans or fractional Hydra overrides into
  apparently valid token grids.
- Workspace `training_contract.json` and preflight image/policy contract summaries use the same
  strict positive-integer sequence parser for `task.image_shape`, robot/open dataset `image_size`,
  and `task.heatmap_token_grid`. String-form Hydra integer lists remain accepted, while booleans,
  fractional values, non-finite values, non-integer strings, and wrong-length geometry sequences
  fail before policy training or preflight can silently record a truncated token contract.
- Policy and joint-transformer constructors apply the same positive-integer parsing to core shape
  fields such as `action_dim`, `action_horizon`, `heatmap_dim`, `heatmap_num_tokens`,
  `max_image_tokens`, `n_layer`, `n_head`, and `n_emb`; string-form integers remain allowed, while
  booleans, fractional values, non-finite values, zeros, and negatives fail before any trainable
  layer is constructed. Their local validation wrappers delegate to the shared Gaze-WAM config
  normalizer, so policy and transformer construction cannot drift from workspace/preflight parsing.
  `n_emb` must also be divisible by `n_head`.
- Dataset runtime booleans that affect sample construction, including `gaze_is_normalized` and
  `action_padding`, use the shared strict boolean parser. Quoted Hydra values such as `"false"`,
  `"off"`, or `"0"` therefore change actual sample indexing/gaze normalization instead of becoming
  truthy Python strings. Workspace `training_contract.json` and preflight sampling summaries use
  the same parser for these fields.
- Observation history is supported. `T_obs` controls how many image frames are encoded.
- Default first setting: `T_obs=2`, matching the desired two-frame observation history.
- The algorithmic default assumes a pretrained DINOv3 ViT/16 checkpoint. Debug, smoke, and
  checkpoint-free debug configs may set `policy.obs_encoder.pretrained=false` to avoid external
  checkpoint dependencies; that is an execution convenience, not a change to the token contract.
- The current timm identity is `vit_base_patch16_dinov3`; the expected pretrained source is
  `timm/vit_base_patch16_dinov3.lvd1689m` unless explicitly overridden by
  `policy.obs_encoder.pretrained_cfg`, `checkpoint_path`, or `cache_dir`.
- The visual transform must include the checkpoint normalization stats after any non-geometric
  augmentation. For the default DINOv3 config this is mean `[0.485, 0.456, 0.406]` and std
  `[0.229, 0.224, 0.225]`.

Let:

- `B`: batch size.
- `T_obs`: number of observation frames.
- `N_v_per_frame = 256`: number of image tokens per frame for the main DINOv3/16 path.
- `N_v = T_obs * N_v_per_frame`: total image tokens after temporal flattening.
- `N_g = 1`: gaze token count.
- `N_a = action_horizon`, default `16`.
- `N_h = heatmap_horizon * 256`, default `256` when `T_heatmap=1`.
- `D`: transformer embedding dim, default inherited from the repo as `768`.

Historical-frame tokenization:

- Each frame is encoded by the same DINOv3 ViT/16 encoder.
- Patch tokens are flattened in time-major order:

```text
[frame_0_patch_tokens, frame_1_patch_tokens, ..., frame_T_patch_tokens]
```

- Main implementation distinguishes historical frames by time-major flattening plus the joint
  transformer's learned full-sequence positional embedding.
- Optional explicit frame embeddings are implemented as an ablation switch.
  - Default: `use_frame_embedding=False`.
  - If enabled, set `image_tokens_per_frame=256` and `max_obs_frames=T_obs`.
  - `image_tokens_per_frame` and `max_obs_frames` must be positive integers; string-form integers
    are accepted for Hydra overrides, while booleans, fractional values, zero, and negatives fail.
  - The embedding adds one learned frame id to all patch tokens from the same observation frame.
- The gaze token is appended after all image-history tokens.
- The single gaze token corresponds to the latest observation frame by default.
- `T_heatmap=1` also targets the latest observation frame by default.
- Action slicing remains based on the computed `N_v = T_obs * 256`.
- If a later config uses `T_obs=1`, the same contract naturally reduces to `N_v=256`.

Training sequence:

```text
[image_tokens, gaze_token, noisy_action_tokens, noisy_heatmap_tokens]
```

Inference sequence:

```text
[image_tokens, gaze_token, noisy_action_tokens]
```

Inference intentionally drops noisy heatmap tokens to reduce sequence length.
The forward call treats this as a hard contract: when `is_inference=True`,
`noisy_heatmap` must be omitted instead of being passed and silently ignored.
The reverse is also a hard contract: when `is_inference=False`, `noisy_heatmap` must be present so
training cannot silently fall back to an action-only, inference-length sequence.

### 3.3 Output Contract And Slicing

The joint transformer returns:

```python
output: FloatTensor[B, N_v + 1 + N_a + optional(N_h), D]
```

Slice rules:

```python
action_start = N_v + 1
action_end = action_start + N_a
action_features = output[:, action_start:action_end]

heatmap_features = output[:, action_end:]  # training only
```

Important:

- Never rely on `output[:, -N_h:]` for the action path.
- Action slicing must be anchored to `N_v + 1`, because inference removes heatmap tokens.
- `image_tokens`, `gaze_token`, `noisy_action`, and training-time `noisy_heatmap` must share the
  same batch size. Shape mismatches fail before token concatenation.
- Training forward rejects missing `noisy_heatmap`; only `is_inference=True` may omit heatmap
  tokens.
- `N_v` must be a positive integer and must not exceed `max_image_tokens`; transformer forward,
  block-mask construction, and attention-contract summaries all share this check.
- `include_heatmap` in block-mask construction uses the shared strict boolean parser, so string
  values such as `"false"` do not accidentally build a training-length heatmap mask through Python
  truthiness.
- Transformer forward timesteps must be integer scalar or `[B]` inputs. Shape/type mismatches fail
  at the model boundary before timestep embeddings are applied.

## 4. Model Architecture Plan

### 4.1 Visual Condition

Reuse `TransformerObsEncoder` initially, configured for DINOv3 ViT/16.

Main decision:

- Input images are direct-stretch resized to `256 x 256` in the main contract
  (`image_resize_mode: stretch`).
- Crop or letterbox preprocessing is not part of the main method; if introduced later, it must
  remap gaze coordinates and dense heatmap labels with the exact same geometric transform before
  dataset sampling.
- DINOv3 ViT/16 produces a `16 x 16 = 256` image-token grid per frame.
- Historical observations are supported by encoding each frame independently and concatenating
  the frame token grids in time-major order. The main config uses learned full-sequence
  positional embeddings to distinguish token positions across frames, with optional explicit
  frame embeddings available as an ablation.
- Heatmap tokens use the same `16 x 16` spatial grid.

Reasons:

- This avoids the `224 x 224 -> 14 x 14 = 196` mismatch.
- Token-level heatmap supervision stays aligned with visual patch centers.
- It keeps the transformer sequence compact enough for mixed action and heatmap diffusion.
- With `T_obs=2`, the training sequence length is approximately
  `512 image + 1 gaze + 16 action + 256 heatmap = 785` tokens.
- Because gaze labels are spatial labels, geometric image augmentations such as random crop must
  not be applied in the visual encoder unless the dataset remaps gaze coordinates and heatmap
  labels with the exact same transform. The default Gaze-WAM config uses non-geometric color
  augmentation only, followed by the DINOv3 checkpoint normalization.

### 4.2 Gaze Condition

Add `GaussianSpatialEncoder`.

Input:

```python
gaze_xy: FloatTensor[B, 2]
```

Output:

```python
gaze_embedding: FloatTensor[B, 64]
```

Then project:

```python
gaze_token = Linear(64, D)(gaze_embedding).unsqueeze(1)
```

Expected behavior:

- Converts a 2D point into a smooth spatial basis.
- Helps the model learn spatially meaningful gaze conditioning rather than treating `(x, y)` as
  two raw numbers.

Coordinate convention:

- Use normalized image coordinates in `[0, 1]`.
- Pixel coordinates are normalized against the source image width/height before direct stretch
  resize; under `image_resize_mode: stretch`, normalized gaze coordinates remain valid after
  resizing.
- Crop or letterbox modes must not be enabled without explicit gaze/heatmap remapping.
- Point-label rows are expected to be valid after data preparation; invalid or missing point labels
  are a data-prep concern, not something the main training loop silently repairs.
- `GaussianSpatialEncoder` and `GazeConditionEncoder` reject non-tensor, non-floating,
  non-finite, or out-of-range `gaze_xy` inputs before Gaussian encoding; valid point labels must be
  normalized to `[0, 1]`.
- Rows with unusable point gaze set `has_gaze_label=False`, zero the `gaze_xy` placeholder, route
  gaze conditioning through `[MASK]`, and do not contribute the main DSNT+JS heatmap loss.
- `GaussianSpatialEncoder.valid_mask`, `GazeConditionEncoder.has_gaze_label`, and
  `GazeConditionEncoder.use_gaze_condition` are strict boolean masks with the same leading shape as
  `gaze_xy[..., 0]`; the encoder must not numerically coerce float/int masks.
- If `has_gaze_label` and `use_gaze_condition` are both provided, the encoder enforces
  `use_gaze_condition=True` only where `has_gaze_label=True`. Missing point-gaze rows must route to
  the trainable `[MASK]` token.
- Main-contract dataset configs keep `gaze_key: gaze_xy`. `heatmap_key: gaze_heatmap` remains a
  nullable compatibility hook for stored-target ablations and visualization metadata, but the
  canonical first run can omit `data/gaze_heatmap`.
- Canonical data-prep tools default to `gaze_bounds_policy=error`, so out-of-frame or NaN gaze
  points fail fast instead of being silently clipped into edge labels.
- Gaze-condition dropout or missing-gaze inference sets `use_gaze_condition=False` and uses the
  trainable `[MASK]` token.

Suggested first Gaussian basis:

- `8 x 8` centers, giving 64 basis values.
- Start with `sigma=0.15` in normalized coordinate space.
- `GaussianSpatialEncoder.grid_size` must be a positive integer pair and `sigma` must be a finite
  positive float; bools, fractional grid dimensions, and non-finite sigma values fail at
  construction time through the same shared config normalizers used by the fixed heatmap codec.

### 4.3 Trainable Gaze Mask Token

Add:

```python
self.gaze_mask_token = nn.Parameter(torch.zeros(1, 1, D))
```

Usage:

- Open-source branch in the main method.
- Missing-gaze inference.
- Classifier-free guidance style ablations.
- Robot gaze-condition dropout.
- Robot "no gaze" ablation.

Important:

- The `[MASK]` token is trainable.
- It is not a hard-coded zero feature.
- It represents the learned image-only or gaze-missing condition.
- Because some robot rows use `[MASK]` while still carrying action supervision, the mask token is
  trained for the action path, not only for the open-source heatmap path.

### 4.4 Action Target

Raw robot zarr should store complete absolute TCP trajectories.

Training target:

- The dataset adapter samples an absolute action chunk.
- It converts the chunk into a relative action target before returning the batch.
- The model diffuses and predicts this relative action target.

Default single-arm action shape:

```text
D_action = 10
action = [x, y, z, rot6d(6), gripper_width]
```

Robot dataset compatibility:

- The canonical model target is always 10D.
- If a raw or semi-canonical robot zarr stores `action_abs_tcp` as 9D pose-only TCP and stores
  gripper separately, the dataset must append the synchronized gripper scalar before relative
  action conversion. `gripper_width` is exactly one scalar per timestep, stored as `[N]` or
  `[N, 1]`; multi-column gripper arrays must be canonicalized before validation/training.
- If `n_latency_steps > 0`, the dataset composes the full `action_horizon + n_latency_steps`
  10D future chunk first, then drops the latency prefix, so action pose and gripper stay aligned.

Relative transform rule:

- Let `action_abs[t]` be the absolute TCP pose/gripper at action step `t`.
- Let `action_base_abs` be the latest observed absolute TCP pose after observation slicing and
  latency alignment.
- If `n_latency_steps > 0`, first sample `action_horizon + n_latency_steps` future absolute
  actions, drop the first `n_latency_steps` actions, and keep the next `action_horizon` actions as
  the model target. This mirrors `W:/umi_base` while keeping the model output horizon unchanged.
- `action_base_abs` remains the latest observed TCP pose at the current observation index; it must
  not be taken from the latency-shifted future action chunk.
- The first real main-method policy run keeps `n_latency_steps=0`. Nonzero latency stays supported
  in the dataset for a later explicit synchronization contract or ablation, but real-data
  main-contract launch readiness fails if `task.n_latency_steps` is nonzero.
- Convert TCP pose components by:

```text
relative_pose[t] = inverse(action_base_abs_pose) * action_abs[t]_pose
```

- Keep gripper width as an absolute scalar in the main method.
- The shared conversion utilities first coerce action/base inputs to finite floating-point arrays,
  so integer arrays cannot truncate relative pose outputs and NaN/Inf actions fail before pose
  matrix conversion.

Implementation reference:

- Mirror the structure of `absolute_actions_to_relative_actions` in
  `W:/umi_base/diffusion_policy/common/action_utils.py`.
- Align with the `W:/umi_base` dataset convention:
  - build `obs_dict` from the observation window,
  - use `obs_dict['*_robot_tcp_pose'][-1]` or `_get_base_obs_from_sample(...)[-1]` as the base,
  - drop latency-shifted action prefixes before conversion when `n_latency_steps > 0`,
  - convert absolute future action chunks relative to that latest observed TCP base.
- Fit the action normalizer on the relative action distribution, not on the stored absolute TCP
  distribution.
- The normalizer provenance is audited in code: `training_contract.json.normalizer` and
  `preflight.policy_contract.normalizer_contract` must report
  `source="robot_dataset_relative_actions_only"`, `normalizer_keys=[camera_key, "action"]`,
  `action_dim=10`, and `excludes_open_source_dummy_actions=true`.
- For open-only heatmap-pretraining/debug configs there is no robot action distribution to fit.
  In that boundary mode the workspace writes an identity action normalizer with
  `source="identity_action_placeholder_for_heatmap_only"`. It exists only because the policy
  boundary normalizes the zero action placeholders before loss routing; it is not action
  supervision and should not be used for action-policy evaluation.
- Workspace `training_contract.json.checks` also includes normalizer-specific canonical checks, so
  `canonical_main_config_ok` fails if the first-run contract stops using robot-relative action
  normalizer provenance or accidentally allows open-source dummy actions into normalizer fitting.
- Keep `action_base_abs` so policy/evaluation tooling can convert relative predictions back to
  absolute TCP when needed. Live deployment can reuse the same metadata convention later.

Latency/base convention:

- Main Gaze-WAM convention: base is the last TCP pose in the observation window, matching
  `W:/umi_base`.
- If a future runner or data pipeline uses latency compensation, the dataset must first align the
  observation/action slices the same way as `W:/umi_base`, then convert the future action chunk
  relative to the latest observed TCP base.
- Do not use the first future action frame as the base, because that can make `action[0]`
  degenerate and is not available as a known deployment state.
- This should be unit-tested before real-robot training.

### 4.5 Heatmap Target, Frozen Cosmos Codec, And Full-Resolution Loss

Main decision:

- Canonical zarr rows store normalized `gaze_xy`; `gaze_heatmap` is no longer required for the
  main training contract. Dense heatmap images may still appear as optional compatibility,
  visualization, or future SAM-mask metadata.
- Inside the DiT, heatmaps are represented as compact continuous latent tokens, not a sequence of
  `65,536` one-pixel tokens. The online target image generated from `gaze_xy` is encoded by the
  frozen Cosmos `Cosmos-Tokenizer-CI16x16` encoder to `16 x 16 x 16` raw latent values before
  scheduler noising. The predicted clean latent tokens are denormalized and decoded by the frozen
  Cosmos decoder to `256 x 256` logits for DSNT/JS loss and preview.
- The default token channel dimension is `heatmap_dim=16`, corresponding exactly to the Cosmos
  continuous latent channel count. The stopped `heatmap_dim=256` patch path is obsolete because it
  directly carried `16 x 16` pixel patches instead of learning/generating in tokenizer latent
  space.
- The active heatmap codec is frozen. The encoder is used only to create clean latent labels from
  the online full-resolution target image; the decoder receives predicted latents without
  `torch.no_grad`, so DSNT/JS gradients flow through the frozen decoder input back to the heatmap
  DiT while decoder weights remain fixed.

Define the frozen Cosmos latent codec concept:

```text
encode_point_to_tokens:
  normalized gaze point (x, y) -> optional 16 x 16 scalar token map for lightweight ablations

encode_latent_image:
  1-channel 256 x 256 heatmap
  -> repeat/adapt to RGB Cosmos input
  -> frozen Cosmos CI16x16 encoder
  -> raw 16 x 16 x 16 latent
  -> project affine scale/offset
  -> scheduler-space heatmap tokens [256, 16]

decode_tokens_to_image:
  scheduler-space heatmap tokens [256, 16]
  -> inverse affine scale/offset
  -> frozen Cosmos CI16x16 decoder
  -> 256 x 256 heatmap logits/image
```

NVIDIA's standalone `Cosmos-0.1-Tokenizer-CI16x16` JIT package does not provide a checkpoint-specific
latent normalizer. The main config therefore uses a project-estimated affine map for the exact
Gaze-WAM chain `gaze_xy -> online Gaussian heatmap -> frozen Cosmos CI16x16 encoder`. The current
stats file is `data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json`,
with raw latent min/max `[-3.921875, 3.375]`, `abs_max=3.921875`, and `abs_p99.5=3.5`.
The active default is the conservative clip-safe setting `heatmap_latent_scale=0.25` and
`heatmap_latent_offset=0.0`, which maps the observed label range to
`[-0.98046875, 0.84375]` before scheduler `clip_sample=[-1, 1]`. The frozen decoder always receives
denormalized raw Cosmos latents.

Point/dense-label encoding:

- The zarr conversion/preparation stage is responsible for normalizing and validating
  `data/gaze_xy`. It should not write dense heatmaps by default, because the full HOT3D-style open
  zarr is large and point supervision is sufficient for the current DSNT+JS objective.
- The policy owns the default `gaze_xy -> target_heatmap_image` transform at training time. The
  current implementation uses a normalized Gaussian centered at `gaze_xy` with
  `policy.heatmap_dsnt_target_sigma_px`; later variants may switch this online target builder to
  SAM-derived masks, edge-softening, time-window aggregation, confidence/null mass, or object-aware
  weighting.
- If a future conversion script writes dense heatmap metadata, the label-generation choices must be
  written to zarr metadata and treated as an explicit alternate target source, not an implicit main
  requirement.
- The codec rejects non-tensor, non-floating, non-finite, or out-of-range point labels before
  Gaussian encoding; point labels must already be normalized to `[0, 1]`.
- Optional point-label masks must be boolean or numeric 0/1 tensors with exactly the point-label
  leading shape. Soft masks such as `0.5` and broadcast-only shapes are rejected so missing-gaze
  rows cannot quietly create partial labels.
- Dense heatmap-image labels passed directly to the fixed codec must already be floating, finite,
  non-negative, single-channel tensors with spatial size equal to the configured `image_size`.
  Supported codec shapes are `[..., H, W]` and `[..., 1, H, W]`; raw dataset-specific formats such
  as `HWC` are normalized before calling the codec.
- Under the main latent contract, dense heatmap labels produce scheduler-space token labels with
  shape `[..., 256, 16]`. Scalar token rows shaped `[..., 256]` or `[..., 256, 1]` are allowed only
  for explicit scalar-token ablations. Patch rows such as `[..., 256, 256]` and compact
  deterministic rows such as `[..., 256, 4]` are historical ablations, not the canonical method.
- Heatmap image decode accepts floating, finite Cosmos latent token tensors with shape
  `[..., 256, 16]` under the main `16 x 16` token grid and decodes them to `[..., 256, 256]`.
  Diagnostic paths may spatial-softmax predicted clean heatmap logits before visualization/metrics.

Current DSNT+JS diffusion version:

- The main method keeps compact latent heatmap tokens as the model output shape, but the heatmap
  head is supervised after reconstructing the clean latent prediction and decoding it to a dense
  `256 x 256` spatial map.
- For `heatmap_objective=dsnt_js`, heatmap tokens follow diffusion clean-estimation semantics:
  when the scheduler predicts epsilon, the heatmap head output is first converted back to a clean
  heatmap-logit estimate with the same scheduler equation used by action diffusion.
- Heatmap supervision replaces latent MSE/token-KL in the main method:
  - DSNT coordinate loss constrains the expected `(x, y)` location of the predicted spatial
    distribution to match `data/gaze_xy`;
  - pixel-space Jensen-Shannon loss constrains the full predicted heatmap distribution to match
    the online target distribution generated from `data/gaze_xy`.
- The predicted clean heatmap image is treated as logits and normalized with spatial softmax for
  DSNT/JS. The target heatmap image is clamped non-negative and sum-normalized.
- The separate heatmap-objective ablation name for direct clean-token regression is
  `heatmap_objective=clean_token`. This is intentionally distinct from the scheduler's
  `prediction_type='sample'`.
- The previous diffusion-MSE/token-KL path is retained only as an ablation/backward-compatible
  objective under `heatmap_objective=diffusion`, but it is not the default main method and is not
  mixed into `heatmap_objective=dsnt_js`.
- Heatmap visualization and validation-preview decoding must mirror the configured objective:
  for `heatmap_objective=dsnt_js`, convert the model output back to clean heatmap logits, decode
  the latent tokens to `256 x 256`, and spatial-softmax it for visualization; for
  `heatmap_objective=clean_token`, treat the heatmap head output itself as the clean token
  prediction. This keeps paper/debug previews aligned with the trained heatmap head.

Token-to-image visualization:

- Decode predicted Cosmos latent tokens from `[256, 16]` to a `256 x 256` heatmap image.
- For scalar-token ablations only, convert `16 x 16` scalar tokens to a `256 x 256` visualization
  by bilinear upsampling or Gaussian splatting from token centers.
- Any override decode `image_size` follows the same positive-integer geometry rule as the codec
  constructor; booleans, fractional values, zero, and negatives are rejected.
- For paper visualizations, show the decoded full-resolution heatmap over RGB and optionally a
  grayscale heatmap panel. The main training loss is DSNT coordinate loss plus dense spatial JS.

Relationship to `GaussianSpatialEncoder`:

- `GaussianSpatialEncoder` converts the input gaze point into one condition token.
- `CosmosHeatmapCodec` converts the online target heatmap image into 256 frozen-Cosmos latent
  target tokens and decodes predicted tokens back to dense images for loss/preview. `HeatmapTokenCodec`
  remains a lightweight geometry/debug helper; scalar token and patchified encodings are kept only
  for ablations/debugging.
- They may share grid/sigma conventions, but they serve different roles.

### 4.6 Joint Transformer Backbone

The current `TransformerForActionDiffusion` uses decoder cross-attention, not joint sequence
self-attention. Gaze-WAM uses a dedicated joint-sequence backbone:

```text
input tokens + position embeddings + modality embeddings + timestep embeddings
  -> TransformerEncoder
  -> output tokens
```

Inputs:

- Projected image tokens.
- One gaze token.
- Projected noisy action tokens.
- Projected noisy heatmap tokens during training.

Current implementation:

- Use one `nn.TransformerEncoder`.
- Add modality/type embeddings:
  - image
  - gaze
  - action
  - heatmap
- Add learnable positional embeddings for the max token budget.
- Add diffusion timestep embedding to noisy target tokens.
- Add a modality-level block attention mask.

Why not reuse the current transformer unchanged:

- It predicts only action samples.
- It separates target and condition via decoder memory.
- It does not naturally support dropping heatmap tokens at inference while keeping stable action
  slicing in one shared sequence.

### 4.7 Block Attention Mask

Use a modality-level attention mask where rows are query tokens and columns are key/value tokens.

Fixed main-method mask:

```text
query token \ key token    image/gaze   action   heatmap
image/gaze                 yes          no       no
action                     yes          yes      no
heatmap                    yes          no       yes
```

Rationale:

- Heatmap queries cannot read action keys, so open-source heatmap loss cannot use dummy action
  tokens.
- Image/gaze condition queries cannot read action or heatmap keys, so target placeholders cannot
  write back into condition tokens and later leak into either target path.
- Action queries cannot read heatmap keys, because heatmap tokens are removed at inference.
- Action queries can read image/gaze and action keys; heatmap queries can read image/gaze and
  heatmap keys.
- Both target paths can still use the shared image/gaze condition.

This is not two independent models:

- Action and heatmap target tokens do not read each other in the main method.
- They still share the visual encoder, gaze encoder, token embeddings, timestep embeddings,
  transformer block weights, normalization layers, and visual-gaze condition tokens as context.
- Heatmap supervision improves shared visual-gaze representations and transformer weights without
  making action prediction depend on heatmap tokens at deployment.
- `JointGazeWamTransformer.attention_contract_summary()` exposes these review invariants in code:
  condition tokens do not read target tokens, action tokens do not read heatmap tokens, heatmap
  tokens do not read action tokens, and action inference drops the heatmap segment.
- For the no-block-mask ablation, the same summary must report `condition_reads_targets=True`,
  `action_reads_heatmap=True`, and `heatmap_reads_action=True`, because the ablation deliberately
  removes the modality isolation. Review artifacts must not present that ablation as if the main
  method's block mask were still active.
- Attention-contract summaries validate `num_image_tokens` as a positive integer that does not
  exceed `max_image_tokens`, matching the transformer forward sequence-length contract.

Relation to Fast-WAM:

- Fast-WAM trains video and action experts together: video/action noisy targets are denoised in one
  MoT forward and share the mixed-attention backbone.
- Its fast action path then skips test-time future-video imagination. It encodes the observed first
  frame through the video/world expert once, caches per-layer video K/V, and denoises action tokens
  against that cache.
- Therefore the transferable principle is not "feed generated future video pixels to action";
  it is "train a world expert with auxiliary prediction, then let action consume stable world
  K/V cache at inference."
- Gaze-WAM should follow the same principle: heatmap supervision trains a gaze/world expert, while
  action prediction consumes the stable heatmap/world K/V cache rather than decoded heatmap pixels
  or noisy heatmap target tokens.

### 4.8 Implemented V2: Cached Dual-Stream DiT

The active model is the Fast-WAM-style cached dual-stream DiT implemented inside this codebase
rather than a rebase onto Fast-WAM. The old `joint` prototype remains only as low-level
reference/test code because the cached branch still reuses a few shared helper types from it.
Hydra policy construction rejects `policy.model_architecture` values other than
`cached_dual_stream`, so the cached branch is the only active policy-training architecture.

V2 module split:

```text
heatmap/world expert:
  image tokens + gaze or [MASK] token
  -> world hidden states / per-layer K/V cache

heatmap DiT decoder:
  noisy latent heatmap tokens
  -> mixed attention over cached heatmap/world K/V + current heatmap K/V
  -> heatmap head

action DiT decoder:
  noisy action tokens
  -> mixed attention over cached heatmap/world K/V + current action K/V
  -> action head
```

Shared-cache rule:

- The cache is for stable heatmap/world expert tokens computed from current image/gaze inputs, not
  for noisy action tokens or noisy heatmap target tokens.
- Action reads the heatmap/world K/V cache. It still does not read decoded heatmap pixels, noisy
  heatmap target tokens, or heatmap outputs that are absent from the fast action path.
- Heatmap and action target tokens remain isolated from each other; the coupling happens through
  the world expert cache.
- During action inference, compute the heatmap/world cache once, then run only the action decoder
  for each DDIM/DDPM denoising step.
- If classifier-free guidance is enabled, build two world caches: one for real gaze and one for
  `[MASK]` gaze.
- During open-only heatmap training or preview, the action decoder can be skipped entirely.

Full-resolution heatmap output without pixel-length attention:

- Do not represent a `256 x 256` heatmap as `65,536` Transformer tokens.
- Keep heatmap token count tied to the visual patch grid and use a compact per-token latent
  dimension.
- Preferred V2 full-resolution contract:

```text
heatmap_token_grid = 16 x 16
heatmap_num_tokens = 256
heatmap_latent_subgrid = 2 x 2
heatmap_dim = 4

pred_heatmap_tokens: [B, 256, 16]
denormalize -> frozen Cosmos decoder -> [B, 256, 256]
```

- A stricter future variant can compare Cosmos CI8x8, FLUX/SANA-style image codecs, or a
  heatmap/mask-specialized frozen AE/VAE while preserving the same policy-side latent-token
  interface.
- The prediction head can keep the same original-repo style:

```python
heatmap_ln_f = nn.LayerNorm(D)
heatmap_head = nn.Linear(D, heatmap_dim)
```

Training objective for full-resolution heatmaps:

- Diffuse the RGB-aligned heatmap target in frozen Cosmos latent-token space, not as
  low-resolution scalar gaze tokens, not as direct `16 x 16` image patches, and not as `65,536`
  pixel tokens.
- For the canonical `heatmap_objective=dsnt_js` path, reconstruct the clean latent prediction,
  decode it to full-resolution logits, then optimize DSNT coordinate loss plus spatial JS against
  the online Gaussian distribution generated from `gaze_xy`. Token MSE/KL is disabled in the main
  method and retained only for ablation objectives.
- SAM-derived labels can be stored as dense RGB-aligned heatmaps, but the main training path should
  still encode them through the same frozen Cosmos encoder before noising. The zarr metadata must
  record the SAM model/source, mask-selection rule, edge-softening rule, and whether the final
  heatmap is sum-normalized or has an additional confidence/null mass.
- Dataset adapters use `heatmap_dim` to choose the heatmap target representation:
  `heatmap_dim=16` is canonical, `heatmap_dim=1` keeps the old scalar-token ablation,
  `heatmap_dim=patch_area=256` is an obsolete direct patch path, and other square compact values
  such as `4` remain ablation-only.
- In Hydra configs, `task.heatmap_dim` is the single source for both dataset heatmap placeholder
  shape and `policy.heatmap_dim`. The canonical full-resolution default is `task.heatmap_dim=16`
  with `policy.heatmap_spatial_decoder=cosmos_tokenizer`; use
  `policy.model_architecture=cached_dual_stream` or the dedicated
  `train_gaze_wam_cached_dual_stream_workspace` config for the FastWAM-style path.

Lightweight runtime smoke once the project venv has `torch` installed:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_cached_dual_stream_gaze_wam.py
```

Why keep V2 in this repository:

- This repository already owns the canonical robot/open zarr schemas, mixed loss routing,
  UMI-aligned relative TCP action conversion, DINOv3 observation encoder, preflight checks,
  validation previews, and policy-training workspace.
- Fast-WAM's code is most useful as an implementation reference for per-layer video/world K/V
  prefill and action-branch mixed-attention cache consumption. Its web/video agent stack is not a convenient
  replacement for the Diffusion Policy / Gaze-WAM data and training infrastructure.

### 4.9 Prediction Heads

Follow the original repository's transformer action output style as closely as possible.

Reference:

- `diffusion_policy/model/diffusion/transformer_for_action_diffusion.py`
- Original output path:

```python
x = self.decoder(...)
x = self.ln_f(x)
x = self.head(x)  # nn.Linear(n_emb, output_dim)
```

Gaze-WAM keeps the same simple head pattern instead of introducing a heavier MLP.

Action head:

```python
action_ln_f = nn.LayerNorm(D)
action_head = nn.Linear(D, D_action)  # D_action = 10 = 3 pos + 6 rot + 1 gripper
```

Heatmap head:

```python
heatmap_ln_f = nn.LayerNorm(D)
heatmap_head = nn.Linear(D, 1)
```

Token-level prediction:

```python
pred_action = action_head(action_ln_f(action_features)).reshape(B, T_action, 10)
pred_heatmap_tokens = heatmap_head(heatmap_ln_f(heatmap_features)).reshape(B, T_h, 256, 1)
```

Why keep separate heads:

- Action and heatmap target dimensions differ (`10` vs `1`).
- The DiT hidden tokens stay in shared dimension `D`.
- Separate `LayerNorm + Linear` heads preserve the original action-output style while allowing
  modality-specific final projection.

Visualization:

- Decode predicted heatmap tokens with the fixed `HeatmapTokenCodec`.
- Render to `256 x 256`.
- During training diagnostics, visualize reconstructed clean heatmap tokens:
  - if the scheduler predicts `sample`, decode `pred_heatmap_tokens` directly;
  - if the scheduler predicts `epsilon`, use the scheduler formula to estimate `x0` before
    decoding.
- Do not add pixel heatmap loss in the main method.

## 5. Diffusion Objective And Loss Routing

Use existing diffusers schedulers for both action and heatmap targets.

Training flow:

1. Build a mixed batch with robot and open-source gaze samples.
2. Sample diffusion timestep `t`.
   - Default: use one timestep per sample and share it between action and heatmap noisy targets.
   - This matches the original action diffusion style while keeping multi-task denoising aligned.
3. Add noise to action rows using zero-filled placeholders where `has_action=False`.
4. Add noise to the heatmap tensor for every row. Rows with `has_heatmap=False` must carry zero
   heatmap placeholders; their heatmap loss is masked out and action/condition tokens cannot read
   heatmap tokens.
5. Run the unified joint transformer.
6. Compute scheduler target based on `prediction_type`:
   - `epsilon`: predict the exact noise added to action/heatmap tokens.
   - `sample`: predict the clean action/heatmap tokens.
7. Apply per-sample loss masks.

Before these steps run, `GazeWamPolicy.compute_loss_components` validates the mixed-batch tensor
contract:

- `action` must be `[B, T_action, D_action]`.
- `heatmap` must be `[B, 1, 256, 1]` or `[B, 256, 1]`.
- `gaze_xy` must be `[B, 2]`.
- `is_open`, `has_action`, `has_heatmap`, `has_gaze_label`, `use_gaze_condition`, and
  `is_gaze_condition_dropped` must be boolean vectors of length `B`.
- Legacy `valid_mask` is not part of the Gaze-WAM mixed-batch contract; if present,
  `compute_loss_components` raises an explicit error and routing must use `has_action`,
  `has_heatmap`, and source masks instead.
- Floating action, heatmap, and gaze tensors must contain finite values.
- Heatmap token labels must stay in `[0, 1]`.
- Rows with `has_action=False` must have zero action placeholder tokens.
- Rows with `has_heatmap=False` must have zero heatmap placeholder tokens.
- Rows with `has_gaze_label=True` must have normalized `gaze_xy` values in `[0, 1]`.
- Rows with `has_gaze_label=False` must have zero `gaze_xy` placeholder values.
- Rows where optional metadata masks are false must have zero optional metadata placeholders.
- Full image finite/range checks stay in zarr validation and preflight so the training hot path
  does not scan every image tensor twice.
- `use_gaze_condition=True` is allowed only when `has_gaze_label=True`.
- Open-source rows must have `has_action=False`, `has_heatmap=True`, and
  `use_gaze_condition=False`.
- Robot rows must have `has_action=True`.
- Robot rows with `use_gaze_condition=True` must have `has_heatmap=False`, so a real gaze
  condition cannot also create a shortcut heatmap reconstruction loss.
- `is_gaze_condition_dropped` must equal `~use_gaze_condition`; the policy loss path treats this
  as a required routing-audit field rather than optional metadata.

This is a training guardrail only; valid batches keep the same loss routing and model behavior.
`GazeWamPolicy.loss_routing_contract_summary()` exposes the same rules as a code-level contract.
Preflight records this under `policy_contract.loss_routing_contract`, and workspace runs copy it
into `training_contract.json` under `routing` before optimization starts.
`_check_policy_contract()` treats the loss-routing validation flags as required readiness
guardrails, so a policy that omits the zero-placeholder or source-routing invariants cannot pass
preflight by merely reporting the high-level action/heatmap mask formulas.
`launch_gaze_wam_training.py` also reads the same nested preflight contract after preflight
returns, records `preflight_routing_validation_guardrails_ok` at the top level of the launch
report, and blocks `--run` if the guardrail check is false even when a malformed or future
preflight artifact reports `ok=true`.
Workspace `training_contract.json.checks.routing_validation_guardrails` enforces the same flag set
for real training runs, so `canonical_main_config_ok` also fails if the workspace artifact loses
these loss-routing invariants.
Both gates use the shared
`gaze_wam_loss_routing_validation_guardrails_ok(...)` helper from
`diffusion_policy.common.gaze_wam_training_config`, so the required flag list has one code owner.
The policy summary itself also generates its `validation` block from the same shared flag list,
keeping the producer and the readiness gates aligned.

The routing-summary helper used by preflight and workspace logs also validates every source and
loss mask as a length-`B` `BoolTensor`. It must not coerce float/int masks with `.to(bool)`, because
that would make corrupted custom batches appear as valid routing counts in audit artifacts.

Main masks:

```python
action_loss_mask = (~batch["is_open"]) & batch["has_action"]
heatmap_loss_mask = batch["has_heatmap"]
```

Under the main data policy:

- Open-source rows always have `has_heatmap=True`.
- Robot rows have `has_heatmap=True` when gaze-condition dropout is applied, or when a robot row
  lacks a usable point-gaze condition but carries a valid dense heatmap target and
  `robot_heatmap_on_gaze_dropout=True`.
- Robot rows with real gaze condition have `has_heatmap=False`.

Masked action loss:

```python
per_sample_action_loss = mse(pred_action, diffusion_target_action).mean(dim=(1, 2))
action_loss = distributed_masked_mean(per_sample_action_loss, action_loss_mask)
```

Masked heatmap token loss:

```python
per_sample_heatmap_loss = mse(pred_heatmap_tokens, diffusion_target_heatmap).mean(non_batch_dims)
heatmap_loss = distributed_masked_mean(per_sample_heatmap_loss, heatmap_loss_mask)
```

This token-MSE path is no longer the canonical main method. It remains an ablation/backward-
compatible objective for `heatmap_objective=diffusion`.

DSNT+JS heatmap objective:

```python
pred_clean_heatmap_tokens = reconstruct_clean_tokens(noisy_heatmap, pred_heatmap_tokens, timestep)
pred_heatmap_logits = decode_latent_tokens(pred_clean_heatmap_tokens)  # [B, 256, 256]
target_heatmap = build_dense_gaussian_from_xy(gaze_xy)                 # [B, 256, 256]

per_sample_heatmap_xy_loss = dsnt_coordinate_loss(pred_heatmap_logits, gaze_xy)
per_sample_heatmap_js_loss = spatial_js_loss(pred_heatmap_logits, target_heatmap)

heatmap_xy_loss = distributed_masked_mean(per_sample_heatmap_xy_loss, heatmap_xy_loss_mask)
heatmap_js_loss = distributed_masked_mean(per_sample_heatmap_js_loss, heatmap_loss_mask)
heatmap_loss = heatmap_xy_loss_weight * heatmap_xy_loss + heatmap_js_loss_weight * heatmap_js_loss
```

`heatmap_xy_loss_mask = has_heatmap & has_gaze_label`. In canonical open-source gaze data this
should match `heatmap_loss_mask`, because the dense target heatmap is generated from a normalized
gaze point or from a gaze-point plus object/mask post-processing pipeline.

Total:

```text
loss = action_loss_weight * action_loss
     + heatmap_loss_weight * heatmap_loss
     + heatmap_token_kl_loss_weight * heatmap_token_kl_loss
```

For the main method, `heatmap_token_kl_loss_weight=0.0`; token KL/MSE remains available only for
explicit ablations.

Fixed first-run weights:

- `action_loss_weight = 1.0`
- `heatmap_loss_weight = 1.0`
- `heatmap_token_kl_loss_weight = 0.0`
- `heatmap_xy_loss_weight = 1.0`
- `heatmap_js_loss_weight = 1.0`
- `heatmap_dsnt_temperature = 1.0`

The default method keeps diffusion-style clean heatmap reconstruction for denoising, but the loss is
applied in dense spatial distribution space because heatmap labels represent gaze probability rather
than Euclidean action vectors.

Important:

- Robot rows update the action path.
- Open-source rows update the heatmap path and shared backbone.
- Robot rows with gaze-condition dropout update both the action path and the heatmap path.
- The action head is not frozen.
- Open-source rows do not contribute action gradients.
- Robot rows with real gaze condition do not contribute heatmap gradients.

Distributed masked mean requirement:

- In DDP/Accelerate training, masked losses must reduce numerator and denominator across all ranks.
- Do not compute a local rank-only mean and then average rank losses, because a rank may contain
  a different number of robot/open/dropout rows.
- Logged `*_mask_count` and training `train_routing_*` counts use the same cross-rank mask-count
  reduction as the loss denominator. Multi-GPU logs must not mix globally reduced losses with
  local-only source/mask counts.
- Required reducer shape:

```python
masked_sum = (per_sample_loss * mask.float()).sum()
mask_count = mask.float().sum()
all_reduce(masked_sum, op="sum")
all_reduce(mask_count, op="sum")
loss = masked_sum / mask_count.clamp_min(1.0)
```

- If `mask_count == 0` globally for a task, return a zero loss connected to the graph or skip that
  task for the step.
- `distributed_masked_mean` and `distributed_mask_count` require route masks to be `BoolTensor[B]`;
  numeric or broadcast-shaped masks are rejected so source routing cannot turn into soft loss
  weighting by accident.

## 6. Mixed Training Strategy

### 6.1 Data Streams

Use two zarr sources, two source datasets/dataloaders, and one mixed batch builder.

This is a fixed policy-training contract, not just an implementation preference:

- `task.robot_dataset_path` points to the robot zarr.
- `task.open_dataset_path` points to the post-processed open-source gaze zarr.
- The two paths must stay distinct in the main mixed-training method.
- Robot rows are managed by `GazeWamRobotDataset` and `robot_dataloader`.
- Open-source gaze rows are managed by `GazeWamOpenDataset` and `open_dataloader`.
- Do not pre-merge robot and open-source rows into a single zarr for the main method.

`RobotDataLoader`:

- Real robot trajectories.
- Provides image, absolute TCP trajectory, action base pose, and either a point gaze coordinate or
  a dense one-channel gaze heatmap label.
- Dataset adapter converts absolute TCP action chunks into relative action targets.
- Main method sets `has_action=True` for every robot row.
- A configurable subset applies gaze-condition dropout:
  - no dropout: `use_gaze_condition=True`, `has_heatmap=False`;
  - dropout in the main method: `use_gaze_condition=False`, `has_heatmap=True`;
  - missing point-gaze condition in the main method: `use_gaze_condition=False`,
    `has_heatmap=True` when a heatmap target is available and
    `robot_heatmap_on_gaze_dropout=True`;
  - dropout in pure no-gaze action baselines: `use_gaze_condition=False`, `has_heatmap=False`.

`OpenDataLoader`:

- Ego-Exo4D or similar open gaze data.
- Provides image/video frame and gaze point or gaze heatmap label.
- Does not provide robot action.
- Main method sets `has_action=False`, `has_heatmap=True`, `use_gaze_condition=False`.

### 6.2 Batch Composition

Target composition:

```text
75% robot samples : 25% open-source gaze samples per optimizer step
```

Implementation contract:

- The ratio is defined by the two source dataloader batch sizes, not by sampling a single merged
  dataset.
- Each optimizer step fetches one mini-batch from `robot_dataloader` and, when enabled, one
  mini-batch from `open_dataloader`; only after both source-specific dataset managers have emitted
  normalized samples does `build_gaze_wam_mixed_batch(...)` concatenate and route the rows.
- When robot batch size is positive, `robot_dataloader` drives the training epoch and action
  normalizer fitting. `open_dataloader` is then an auxiliary stream with `restart_on_exhaustion`
  semantics, so a smaller post-processed open-source zarr can be cycled without changing the robot
  epoch length.
- The same mixed loop handles boundary ratios. If `robot_dataloader.batch_size=0` and
  `open_dataloader.batch_size>0`, `open_dataloader` becomes the primary epoch driver, all rows
  are open-source heatmap rows, action targets are zero placeholders, and action loss weight should
  be `0.0`. This is the open-only heatmap-pretraining/debug mode.
- If `open_dataloader.batch_size=0`, the loop becomes robot-only and never constructs the open
  dataloader.
- Each source keeps its own zarr path, split, shuffle/sampler policy, validation, and schema
  checks. Cross-source interaction starts at the mixed-batch builder boundary.
- The workspace and preflight record this as
  `source="two_zarr_two_dataset_online_mixed_batch"` in the data-stream contract.
- Full mixed config uses `robot_dataloader.batch_size=48` and `open_dataloader.batch_size=16`,
  giving exactly `48 / (48 + 16) = 75%` robot rows and `25%` open-source rows.
- Debug mixed config uses `3` robot rows and `1` open row for the same ratio.
- Open-only config uses `robot_dataloader.batch_size=0`, `open_dataloader.batch_size>0`,
  `val_robot_dataloader.batch_size=0`, `policy.action_loss_weight=0.0`, and heatmap losses enabled.
- If a future config exposes a single global batch size, set
  `robot_batch_size = round(batch_size * 0.75)` and
  `open_batch_size = batch_size - robot_batch_size`.
- Sample from both dataloaders each step.
- Concatenate along batch dimension.
- Always fill open-source action targets with zeros and set `has_action=False`, even if an open
  source batch accidentally provides an `action` tensor. The builder may shape-check the supplied
  tensor, but its values are not used as noisy action targets.
- For robot rows, sample a Bernoulli `robot_gaze_dropout_prob`.
  - If dropout is false, use real gaze condition and set `has_heatmap=False`.
  - If dropout is true, use `[MASK]` as the gaze condition.
  - Main method sets `robot_heatmap_on_gaze_dropout=True`, so dropout rows also set
    `has_heatmap=True`.
  - Robot rows whose point-gaze condition is unavailable also use `[MASK]`; in the main method
    they set `has_heatmap=True` when a heatmap target is available.
  - Pure no-gaze action baselines set `robot_heatmap_on_gaze_dropout=False`, so dropout rows do
    not train heatmap loss.
- The mixed-batch builder validates these routing knobs through the shared config normalizers:
  `robot_gaze_dropout_prob` must be a finite float in `[0, 1]`, and
  `robot_heatmap_on_gaze_dropout` must be a strict bool or bool-like string such as `"false"`.
  It must not rely on Python string truthiness for loss routing.
- Fill missing heatmap tokens with zeros only for rows where `has_heatmap=False`.
- Fill missing point-gaze values with zero `gaze_xy` placeholders wherever
  `has_gaze_label=False`.
- Shuffle the concatenated batch so source order is not a positional cue.
- Before concatenation, the mixed-batch builder validates that robot/open batches share identical
  observation keys, matching non-batch tensor shapes for image/action/heatmap/gaze fields, and
  one-dimensional `BoolTensor[B]` route masks such as `has_gaze_label` and optional metadata
  masks such as `has_heatmap_image`. It must not numerically coerce `0/1` float or int masks at
  this stage; real-data shape/type errors should fail here with a source-specific message rather
  than reaching `torch.cat` or the optimizer.

Fixed first-run value:

```text
robot_gaze_dropout_prob = 0.2
```

The sweep range for ablations remains `0.15` to `0.30`, but the first executable main-method run
uses `0.2`. This trains the `[MASK]` token on robot action while avoiding a full collapse into
image-only control.

### 6.3 Masked Multi-task Routing

Critical rules:

- Do not fabricate action supervision for open-source samples.
- Use robot heatmap loss only for robot rows whose gaze condition has been dropped.
- Do not dynamically freeze the action head in the main method.
- Route supervision through per-sample masks.

Implementation pattern:

```python
mixed_batch = concat_robot_and_open_batches(robot_batch, open_batch)
pred = model(mixed_batch)

loss_action = distributed_masked_mean(
    mse_per_sample(pred.action, diffusion_target_action),
    (~mixed_batch["is_open"]) & mixed_batch["has_action"],
)
loss_heatmap = distributed_masked_mean(
    mse_per_sample(pred.heatmap, diffusion_target_heatmap),
    mixed_batch["has_heatmap"],
)
loss = loss_action + loss_heatmap
```

Why this is preferred:

- Robot samples update the action head every optimizer step.
- Open-source samples do not create misleading action gradients.
- Robot gaze-dropout samples train action and heatmap jointly under `[MASK]`, which makes the
  mask token meaningful for action prediction.
- Open-source gaze supervision broadens the visual distribution seen by the shared backbone.
- The training loop matches the intended data-scaling story without freezing modules.

Open-source action tokens:

- Open-source samples should not use meaningful action tokens.
- Keep fixed sequence length with zero-filled action placeholders.
- The block attention mask prevents heatmap tokens from attending to dummy action tokens.
- In open-only heatmap training, the policy still receives zero noisy-action placeholders for
  sequence-shape consistency, but every row has `has_action=False`, so action loss is zero-masked
  and the identity action normalizer is only a policy-boundary placeholder.

Robot heatmap tokens:

- Robot samples without gaze-condition dropout use zero heatmap placeholder tokens and
  `has_heatmap=False`.
- Robot samples with gaze-condition dropout, or missing point-gaze rows with a dense heatmap label,
  use real heatmap tokens and `has_heatmap=True`.
- Since action and condition tokens cannot attend to heatmap tokens, heatmap-supervised robot rows
  cannot create train-test mismatch for action inference.

## 7. Inference Strategy

Inference is action-only.

Inputs:

- Image observation tokens.
- Real gaze token if available, otherwise trainable `[MASK]` token.
- Noisy action tokens.

Do not include:

- Noisy heatmap tokens.
- Open-source/video auxiliary targets.

Policy sampling:

- Use DDIM/DDPM loop, not single-step denoising.
- Default executable config uses `num_inference_steps=8`.
- The policy constructor rejects `num_inference_steps < 2`; debug/smoke runs use at least `2`
  scheduler steps.
- The recommended policy-evaluation sweep is `5` to `10` steps; changing this is an inference-speed
  ablation, not a model-shape change.

Pseudo-flow:

```python
action = torch.randn(B, T_action, D_action)
scheduler.set_timesteps(num_inference_steps)

for t in scheduler.timesteps:
    pred_action, _ = model.forward(
        obs,
        noisy_action=action,
        noisy_heatmap=None,
        gaze_xy=gaze_xy,
        is_inference=True,
    )
    action = scheduler.step(pred_action, t, action).prev_sample
```

Policy-level post-processing:

- Model output is relative action.
- Convert relative TCP predictions back to absolute TCP commands with `action_base_abs`.
- Use the same convention as training for rotation representation and gripper width.
- `GazeWamPolicy.predict_action` expects any direct `action_base_abs` to already be canonical
  `[B, 10]` and finite. Adapter/data-prep utilities are responsible for composing pose-only 9D bases
  with a scalar gripper width before calling the policy.
- If `has_action_base_abs` is supplied with `action_base_abs`, every row must be true before the
  policy performs inverse conversion; zero-filled mixed-batch placeholders are not valid bases for
  absolute command output or absolute-action metrics.
- Policy-level action inference, CFG, GDR, and heatmap diagnostics validate gaze routing before
  encoding: `gaze_xy` must be `[B, 2]`, explicitly supplied route masks must be `BoolTensor[B]`
  without numeric coercion, and rows with `has_gaze_label=True` must have normalized gaze values in
  `[0, 1]`.
- These policy inference/diagnostic paths share a non-model-observation filter before
  normalization and visual encoding. Gaze labels, source/loss masks, action/heatmap labels, dense
  heatmap visualization tensors, optional action metadata, and all `has_*` presence masks are
  excluded from `normalizer.normalize(model_obs_raw)`; only real observation tensors such as
  `camera0_rgb` should reach the observation encoder. If filtering leaves no real observation
  tensor, the policy must raise a clear boundary error instead of falling through to the normalizer
  or observation encoder.

Deployment/runner boundary for this session:

- The current closeout is policy-training only. Runner, hardware binding, command scheduling,
  safety clipping, and provider plumbing are not part of the acceptance criteria for this milestone.
- Any existing adapter, runner, split-provider config, or zarr-rehearsal utilities should be treated
  as reference-only scaffolding for a later live-deployment phase.
- Do not expand or tune runner-related code until the policy has been trained on real robot/open
  data and the main ablations have produced stable metrics.

Important:

- The transformer sequence is shorter during inference.
- Action slicing remains:

```python
output[:, N_v + 1 : N_v + 1 + N_a]
```

## 8. Dataset Adapter Plan

### 8.0 Task Config As The Zarr Schema Anchor

The first real policy-training task config is `diffusion_policy/config/task/gaze_wam.yaml`.
It is the canonical schema anchor for the two zarr roots:

```text
robot_dataset_path: data/gaze_wam_robot.zarr
open_dataset_path: data/gaze_wam_open.zarr
camera_key: camera0_rgb
gaze_key: gaze_xy
heatmap_key: gaze_heatmap  # optional compatibility metadata; canonical open zarr may omit it
action_abs_key: action_abs_tcp
tcp_pose_key: tcp_pose_abs
gripper_key: gripper_width
```

The same config fixes the first-run tensor geometry and sampling window:

```text
image_size: 256 x 256
n_obs_steps: 2
n_latency_steps: 0
obs_downsample_steps: 1
action_horizon: 16
action_downsample_steps: 1
action_dim: 10
heatmap_token_grid: 16 x 16
```

Therefore, a robot zarr stores absolute world-frame trajectories and the dataset performs the
model-target conversion online:

1. Sample `n_obs_steps` RGB frames ending at the current observation index.
2. Read `tcp_pose_abs[current_idx]` plus `gripper_width[current_idx]` as the absolute base pose.
3. Sample the future `action_abs_tcp` chunk of length `action_horizon + n_latency_steps`.
4. Drop the first `n_latency_steps` actions; the first-run default is `0`, so nothing is skipped.
5. Convert each absolute TCP action in the chunk into a relative action against the latest observed
   base pose, matching the UMI `relative_action=True` / `use_absolute_action=True` convention.
6. Return this relative `[16, 10]` tensor as `batch["action"]`; keep `action_abs` and
   `action_base_abs` as metadata for validation, metrics, and future inverse conversion.

An open-source gaze zarr uses the same image/gaze geometry but contains no robot action
supervision. `GazeWamOpenDataset` emits a zero `[16, 10]` action placeholder only to keep batch
shapes static; `has_action=False` prevents those rows from entering action loss. For the canonical
DSNT+JS path, heatmap targets are generated online from `gaze_xy` inside the policy.

### 8.1 Robot Data Adapter

The robot zarr should store absolute data, not pre-relative model targets.

Canonical Gaze-WAM robot zarr keys after mapping/canonicalization:

```text
camera0_rgb
action_abs_tcp
tcp_pose_abs
gripper_width
gaze_xy          # required normalized point label / condition source
gaze_heatmap     # optional dense/token metadata for preview or ablation, not required
has_gaze_label   # optional row-level point-gaze validity mask
timestamp       # optional but recommended aligned sample timestamp
episode_ends
```

Robot gaze-label contract:

- A canonical robot zarr must contain `gaze_xy`; `gaze_heatmap` is optional metadata.
- `gaze_xy` must already be normalized to `[0, 1]` in zarr and is the point-gaze condition source
  for rows where `use_gaze_condition=True`.
- If provided, `gaze_heatmap` must be a dense one-channel heatmap aligned with the stored RGB frame
  or a configured token representation with explicit metadata.
- If raw logs have only gaze points, the canonical main path keeps the zarr xy-only. Optional
  preview/ablation scripts may generate dense or token heatmaps explicitly.
- If optional `has_gaze_label[t]` is false for a row, the dataset treats the row as missing a usable
  point-gaze condition: `gaze_xy` is zeroed, `has_gaze_label=False`,
  `use_gaze_condition=False`, and the row does not contribute the main DSNT+JS heatmap loss.
- Dense-heatmap-only zarrs are no longer part of the main contract. They can remain a compatibility
  or ablation path only if a separate config explicitly opts out of the canonical schema.

Optional keys:

```text
left_robot_tcp_pose
right_robot_tcp_pose
head_tcp_pose
image_timestamp
robot_state_timestamp
action_timestamp
gaze_timestamp
```

Main adapter responsibilities:

- Direct-stretch resize RGB to `256 x 256` (`image_resize_mode: stretch`).
- Convert gaze points into normalized `[0, 1]` source-image coordinates before stretch resize.
- If raw gaze is stored in pixels, normalize it using the source image width/height before resizing
  the image. Direct stretch does not change the normalized gaze coordinate.
- Crop or letterbox variants require explicit gaze and heatmap remapping and are not part of the
  first main contract.
- Sample absolute TCP action chunks.
- Convert absolute action chunks into relative action targets.
- Return `action_base_abs` for inverse conversion during policy evaluation and future deployment.
- Generate heatmap token labels from robot gaze points or dense heatmap labels, but activate their
  loss only when the gaze condition is `[MASK]`.
- Return `has_action=True`; return `has_gaze_label=True` only when a point gaze label is present;
  and return per-sample `use_gaze_condition / has_heatmap` flags.

Relative action conversion:

- Use the latest observed TCP pose as the base absolute TCP pose, matching `W:/umi_base`.
- Use SE(3) relative transforms for TCP pose dimensions.
- Keep gripper width absolute in the main method.
- Fit action normalizer on relative actions.

Recommended implementation path:

- First create a generic `GazeWamRobotDataset` for already-processed zarr fields.
- Later add a `UmiGazeWamDataset` wrapper if raw UMI pose conversion is required.
- Reuse the UMI-style action utility logic rather than reimplementing pose math ad hoc.
- If a real robot log uses non-canonical key names, first inspect it to understand likely key
  mappings:

```text
py scripts/inspect_gaze_wam_zarr.py \
  --dataset-path path/to/raw_robot.zarr \
  --dataset-type robot \
  --output-json data/raw_robot_schema_report.json
```

- The inspector reports every zarr array path, shape, dtype, sampled numeric range, episode-boundary
  information, and candidate mappings for camera/action/TCP/gripper/gaze/heatmap/timestamp roles.
- For robot-like logs, it also reports suggested `canonicalize_robot_gaze_wam_zarr.py` arguments
  when all required roles can be inferred.
- Inspector reports also include `source_key_map`, `mapping_status`, and
  `canonicalizer_command_template`. The command template is a reviewable bridge from raw field
  names to the canonical robot schema; check the candidate scores first, then replace
  `<output_robot.zarr>` with the intended canonical output path before running it.
- `inspect_gaze_wam_zarr.py --help` keeps NumPy/zarr imports lazy, so raw robot/open zarr mapping
  arguments can be reviewed in lightweight environments. Actual inspection still requires zarr and
  numeric runtime dependencies.
- For the normal real-data onboarding path, use the one-shot prepare command. It chains inspect,
  canonicalize, validate, and preview:

```text
py scripts/prepare_robot_gaze_wam_zarr.py \
  --input path/to/raw_robot.zarr \
  --output data/gaze_wam_robot.zarr \
  --report-json data/gaze_wam_robot_prepare_report.json \
  --preview-dir data/preview/gaze_wam_robot_prepare \
  --image-resize-mode stretch \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --no-gaze-is-normalized \
  --overwrite
```

- Before writing a canonical zarr, run the same command with `--dry-run`. Dry-run mode still
  inspects the raw store, resolves the key map, writes the optional report JSON, and records the
  exact `canonicalizer_command`, but it skips output zarr creation, validation, and preview. Use
  this first when reviewing a new physical robot log so incorrect camera/action/TCP/gripper/gaze
  mappings can be caught before any conversion output is produced.
- If auto-inferred keys are wrong, pass explicit `--camera-key`, `--action-key`,
  `--tcp-pose-key`, `--gripper-key`, `--gaze-key`, and optional `--timestamp-key` overrides to the
  prepare command.
- The robot prepare report records a `canonicalizer_command` built from the resolved key map and
  actual input/output paths, so a reviewer can reproduce the canonicalization stage without
  reconstructing the CLI from nested report fields.
- When a timestamp candidate is inferred or explicitly provided, `prepare_robot_gaze_wam_zarr.py`
  preserves it as canonical `data/timestamp` by default.
- If the raw robot zarr contains `image_timestamp`, `robot_state_timestamp`, `action_timestamp`, or
  `gaze_timestamp`, the prepare/canonicalize path preserves those streams under the same canonical
  names and passes them into validation for bounded cross-stream drift checks. For nonstandard
  names, pass the explicit `--image-timestamp-key`, `--robot-state-timestamp-key`,
  `--action-timestamp-key`, and `--gaze-timestamp-key` options.
- The robot prepare/canonicalize path defaults to `--gaze-bounds-policy error`.
  Use `--gaze-bounds-policy clip` only for deliberate compatibility checks, because clipped robot
  gaze points become real heatmap labels.
- Then convert with the robot zarr canonicalizer and validate the output:

```text
py scripts/canonicalize_robot_gaze_wam_zarr.py \
  --input path/to/raw_robot.zarr \
  --output data/gaze_wam_robot.zarr \
  --camera-key front_rgb \
  --action-key future_tcp_pose \
  --tcp-pose-key current_tcp_pose \
  --gripper-key gripper_width \
  --gaze-key gaze_xy \
  --no-gaze-is-normalized \
  --overwrite
```

- The standalone canonicalizer remains useful when key mapping is already known and no preview or
  report artifact is needed.
- `canonicalize_robot_gaze_wam_zarr.py --help` keeps NumPy/zarr/gaze utility/validator imports
  lazy, so known-key robot conversion arguments can be reviewed in lightweight environments. Actual
  canonicalization and optional output validation still require the normal zarr and dataset runtime.
- `prepare_robot_gaze_wam_zarr.py --help` keeps canonicalize/validate/preview imports lazy, so
  robot-data argument review works in lightweight environments. Actual robot zarr preparation still
  requires the normal zarr, image, numeric, and dataset runtime dependencies.
- The canonicalizer writes the standard `data/` + `meta/episode_ends` layout and canonical keys:
  `camera0_rgb`, `action_abs_tcp`, `tcp_pose_abs`, `gripper_width`, normalized `gaze_xy`, optional
  heatmap metadata if requested, and `episode_ends`.
- The canonicalizer records `meta.attrs.dataset_type="robot"` so validator/preflight reports can
  distinguish canonical robot data from open-source gaze data without relying only on file names.
- It also records `meta.attrs.image_size` from the actual output RGB tensor and
  `meta.attrs.image_resize_mode="stretch"` so validator, preflight, and `training_contract.json`
  can audit robot geometry provenance with the same metadata contract used by debug/open zarrs.
- If the raw robot log has only point gaze, the canonical main output remains xy-only. If it also
  has dense heatmaps, pass `--heatmap-key <raw_dense_heatmap_key>` only for preview/ablation
  metadata that explicitly uses stored heatmaps.
- Raw dense-heatmap-only robot logs are not canonical main-contract inputs because `gaze_xy` is the
  gaze condition source.
- If source action or TCP pose arrays are 9D pose-only, the canonicalizer appends the gripper
  scalar where a 10D canonical action/base is required. The gripper source must be `[N]` or
  `[N,1]` with finite values; multi-column or non-finite gripper arrays fail instead of being
  silently truncated. It does not change trajectory semantics.
- Source action and TCP pose arrays must also contain only finite values; the canonicalizer rejects
  non-finite labels before writing canonical training data.
- The inspector is only a mapping/report aid; the canonicalizer and validator remain the
  authoritative conversion and contract checks before training.

Unified robot/open onboarding review:

- Before the first physical robot + open-source gaze training run, use the combined dry-run review
  gate to collect both preparation reports under one policy-training-scope JSON artifact:

```text
py scripts/review_gaze_wam_data_onboarding.py \
  --robot-input-zarr path/to/raw_robot.zarr \
  --robot-output-zarr data/gaze_wam_robot.zarr \
  --robot-report-json data/gaze_wam_robot_prepare_dry_run.json \
  --open-video-metadata path/to/raw_video_gaze.jsonl \
  --open-output-zarr data/gaze_wam_open.zarr \
  --open-metadata-inspect-json data/open_video_gaze_metadata_inspect.json \
  --open-adapted-metadata data/open_video_gaze_canonical.csv \
  --open-output-manifest data/open_gaze_manifest.csv \
  --open-frames-dir data/open_gaze_frames \
  --open-report-json data/gaze_wam_open_prepare_dry_run.json \
  --open-key-map '{"video_path":"clip.path","episode_id":"clip.uid","frame_idx":"frame.number","gaze_x":"gaze.point.0","gaze_y":"gaze.point.1"}' \
  --open-root-dir path/to/videos \
  --no-open-gaze-is-normalized \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --output-json data/gaze_wam_data_onboarding_review.json
```

- This command is dry-run only. It delegates to robot/open prepare dry-runs, records the fixed
  policy-training contract (`256 x 256`, `T_obs=2`, `T_action=16`, `16 x 16` heatmap tokens,
  timestamp gates), marks `deployment_runner_scope="deferred"`, and does not write canonical zarrs
  or launch training.
- The combined onboarding review parses robot/open child-stage `ok` fields with the shared strict
  boolean parser. A child dry-run summary containing `"ok": "false"` therefore fails the review
  instead of passing through Python string truthiness.
- Use the combined report for code/data review. Once the resolved robot key map, open metadata
  mapping, and planned commands look correct, rerun the individual prepare commands without
  `--dry-run`, then validate/preview/preflight the written zarrs.

Policy-training readiness bundle:

- For the first real training attempt, collect data-onboarding, DINO-source, and launcher dry-run
  reports under one JSON review bundle before starting any long job:

```text
py scripts/review_gaze_wam_training_readiness.py \
  --robot-input-zarr path/to/raw_robot.zarr \
  --robot-output-zarr data/gaze_wam_robot.zarr \
  --robot-report-json data/gaze_wam_robot_prepare_dry_run.json \
  --open-video-metadata path/to/raw_video_gaze.jsonl \
  --open-output-zarr data/gaze_wam_open.zarr \
  --open-metadata-inspect-json data/open_video_gaze_metadata_inspect.json \
  --open-adapted-metadata data/open_video_gaze_canonical.csv \
  --open-output-manifest data/open_gaze_manifest.csv \
  --open-frames-dir data/open_gaze_frames \
  --open-report-json data/gaze_wam_open_prepare_dry_run.json \
  --open-key-map '{"video_path":"clip.path","episode_id":"clip.uid","frame_idx":"frame.number","gaze_x":"gaze.point.0","gaze_y":"gaze.point.1"}' \
  --open-root-dir path/to/videos \
  --no-open-gaze-is-normalized \
  --dino-checkpoint-path path/to/dinov3.ckpt \
  --preflight-require-timestamps \
  --preflight-timestamp-max-step 0.08 \
  --preflight-fail-on-zarr-warning \
  --output-json data/outputs/gaze_wam_readiness/summary.json
```

- The bundle stays in policy-training scope and records `deployment_runner_scope="deferred"`.
  It can run the combined onboarding dry-run, write a standalone DINO verifier report, then call
  `launch_gaze_wam_training.py` with `--real-data` semantics but `run=False`.
- The top-level bundle also cross-checks the standalone DINO verifier report against the launcher
  report's nested `real_data_readiness.dino_source_verifier`. The DINO signatures must match for
  model name, pretrained flag, checkpoint/cache source, token geometry, heatmap grid, and Normalize
  stats, otherwise the bundle fails even if both individual reports are marked `ok`.
- When launch preflight is enabled, the bundle also checks the launch report's top-level
  `preflight_routing_validation_guardrails_ok` field and records the result under
  `cross_checks.launch_preflight_routing_guardrails`. The bundle fails if this value is missing or
  not `true`, so the review artifact carries the same loss-routing validation guarantee as the
  launcher report.
- If the onboarding review was already created, pass `--onboarding-review-json <path>` instead of
  raw robot/open source arguments. If a review is intentionally not required for a debug audit, use
  `--no-require-data-onboarding-review`; in that mode the launcher stage does not receive a
  fabricated onboarding path.
- The bundle writes sibling artifacts by default:
  `summary_onboarding.json`, `summary_dino.json`, and `summary_launch.json`, unless explicit
  `--onboarding-stage-json`, `--dino-report-json`, or `--launch-report-json` paths are provided.

### 8.2 Open Data Adapter

Canonical Gaze-WAM open-data zarr keys after conversion:

```text
camera0_rgb
gaze_xy          # required normalized point label, retained for labels/eval
gaze_heatmap     # optional dense/token metadata for preview or ablation
timestamp        # optional frame/gaze timestamp, recommended for real video data
episode_ends
```

Main adapter responsibilities:

- Direct-stretch resize RGB to `256 x 256` (`image_resize_mode: stretch`).
- Convert gaze points into normalized `[0, 1]` source-image coordinates before stretch resize and
  write them as canonical `gaze_xy`.
- If raw gaze is stored in pixels, normalize it using the source image width/height before resizing
  the image. Direct stretch does not change the normalized gaze coordinate.
- Crop or letterbox variants require explicit gaze and heatmap remapping and are not part of the
  first main contract.
- Do not write `gaze_heatmap` by default. The policy generates the Gaussian DSNT+JS target online
  from `gaze_xy`; optional dense/token heatmaps are reserved for preview and stored-target
  ablations.
- The open manifest converter records `meta.attrs.dataset_type="open"` alongside label mode,
  image resize mode, image size, gaze normalization, heatmap-source metadata, timestamp key, and
  presence-mask metadata.
- Fill action placeholders with zeros.
- Return `has_action=False`, `has_heatmap=True`, `use_gaze_condition=False`, and
  `has_gaze_label=True` for canonical point-gaze rows.

Important:

- Open-source gaze labels are targets, not conditions, in the main method.
- The model receives the trainable `[MASK]` token for open-source rows.
- Default one-shot preprocessing for an existing image/gaze manifest:

```text
py scripts/prepare_open_gaze_wam_zarr.py \
  --manifest path/to/open_gaze_manifest.csv \
  --output-zarr data/gaze_wam_open.zarr \
  --report-json data/gaze_wam_open_prepare_report.json \
  --preview-dir data/preview/gaze_wam_open_prepare \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --label-mode auto \
  --overwrite
```

- Before writing an open zarr, run the same command with `--dry-run`. Dry-run mode writes the
  optional report JSON and, for video metadata, the optional metadata-inspection JSON, but it does
  not extract frames, adapt metadata, write the manifest/zarr, run validation, or generate preview
  images. The report records `planned_commands` for conversion, validation, preview, and, when
  needed, metadata adaptation/export so the open-data path can be reviewed before heavy IO.
- The open-data converter and prepare command default to `--gaze-bounds-policy error`.
- `--gaze-bounds-policy clip` keeps old permissive behavior for smoke/legacy manifests.
- `--gaze-bounds-policy drop` is not used by the canonical first-run path because canonical open
  zarrs require valid point gaze on every row.
- For point-only manifests, the converter writes normalized `gaze_xy` and, in the canonical path,
  omits stored heatmaps. Manifests with dense heatmap paths are legacy/ablation inputs; use an
  explicit stored-target mode only when that experiment needs it.

- HOT3D Aria two-stage preprocessing path:

```text
py scripts/preprocess_hot3d_aria.py \
  --manifest path/to/hot3d_manifest.json \
  --hot3d-repo path/to/hot3d_repo \
  --work-root W:/HOT3D_processed_work \
  --final-root W:/HOT3D_processed \
  --temp-root W:/HOT3D_temp \
  --all \
  --jobs 4
```

```text
py scripts/convert_hot3d_processed_to_open_zarr.py \
  --processed-root W:/HOT3D_processed \
  --output-zarr data/hot3d_open.zarr \
  --image-size 256 256 \
  --heatmap-storage none \
  --heatmap-token-grid 16 16 \
  --heatmap-method gaussian_point \
  --point-heatmap-sigma-tokens 2.0 \
  --point-heatmap-window 3 \
  --preview-overlay-dir data/preview/hot3d_open_overlay \
  --preview-sigma-compare-tokens 1.25 2.0 3.0 \
  --overwrite \
  --validate
```

- `scripts/preprocess_hot3d_aria.py` is the imported stage-1 HOT3D Aria compacting script. It
  downloads/extracts only the training-relevant Aria streams and writes one directory per sequence
  containing `raw_rgb.mp4`, `gaze_projected_raw_rgb_normalized.csv`, and `processing_summary.json`.
- `convert_hot3d_processed_to_open_zarr.py` is the stage-2 streaming zarr writer. It reads the
  compact packages directly, treats each sequence as one episode, stores RGB as `camera0_rgb`,
  stores `upright_x_norm/upright_y_norm` as canonical `gaze_xy`, writes `timestamp_ns`, sets
  `has_heatmap_image=False` when `--heatmap-storage none`, and records HOT3D provenance in
  `meta.attrs`.
- Stage 2 may render temporary `gaze_xy -> heatmap` images for preview videos, but the default
  full-training zarr uses `--heatmap-storage none` to avoid storing redundant dense or token
  heatmaps. The policy generates the DSNT+JS target distribution online from `gaze_xy` during
  training.
- `--heatmap-storage token` and `--heatmap-storage dense` remain explicit ablation/debug modes for
  testing stored supervision targets. They are not the canonical open-data storage mode.
- Pass `--preview-overlay-dir ...` to write the main visual QA mp4:
  `overlay_heatmap_side_by_side.mp4`. Each frame is `red heatmap over RGB | black/white heatmap`,
  so the point spread can be judged against the original video and as a standalone supervision
  target. When `imageio_ffmpeg` is available, preview mp4s are transcoded to H.264/yuv420p with
  faststart for broad player compatibility; otherwise the script falls back to OpenCV `mp4v`.
  The default preview length is 80 frames; override `--preview-overlay-max-frames` and
  `--preview-overlay-fps` for shorter or longer review clips.
- Pass `--preview-sigma-compare-tokens 1.25 2.0 3.0` together with `--preview-overlay-dir` to write
  `sigma_compare.mp4`, a dynamic sigma comparison video using the same
  `red overlay | black/white heatmap` layout for each sigma row.
- The preview directory also contains `token_decode_vs_native_frame_*.png`, comparing one native
  dense heatmap, the stored token heatmap decoded back to image space, and their absolute
  difference.
- For quick smoke tests, pass `--sequence P0001_10a27bf7 --max-frames-per-sequence 40`. For
  full training, omit those limits and point `task.open_dataset_path` to the resulting zarr while
  using `train_gaze_wam_open_only_workspace` or the normal mixed config.

- Default one-shot preprocessing for video-style or Ego-Exo4D-style metadata:

```text
py scripts/prepare_open_gaze_wam_zarr.py \
  --video-metadata path/to/raw_video_gaze.jsonl \
  --metadata-inspect-json data/open_video_gaze_metadata_inspect.json \
  --adapted-metadata data/open_video_gaze_canonical.csv \
  --output-manifest data/open_gaze_manifest.csv \
  --frames-dir data/open_gaze_frames \
  --output-zarr data/gaze_wam_open.zarr \
  --key-map '{"video_path":"clip.path","episode_id":"clip.uid","frame_idx":"frame.number","gaze_x":"gaze.point.0","gaze_y":"gaze.point.1"}' \
  --filter split=train \
  --root-dir path/to/videos \
  --no-gaze-is-normalized \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --report-json data/gaze_wam_open_prepare_report.json \
  --preview-dir data/preview/gaze_wam_open_prepare \
  --overwrite
```

- The prepare command chains optional metadata inspection, metadata adaptation when needed, frame
  extraction, open-zarr conversion, zarr validation, preview artifact writing, and a JSON report.
- Pass `--metadata-inspect-json` for first-real-dataset onboarding so the raw metadata field
  candidates, suggested key map, mapping status, and adapter command template are archived next to
  the conversion report. When called from `prepare_open_gaze_wam_zarr.py`, the inspector command
  template uses the actual prepare output metadata, manifest, frames, root, and open-zarr paths
  instead of generic placeholders.
- Lower-level generic preprocessing entrypoint:

```text
py scripts/convert_open_gaze_manifest.py \
  --manifest path/to/open_gaze_manifest.csv \
  --output data/gaze_wam_open.zarr \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --label-mode auto
```

- The lower-level converter accepts the same null-like key spellings. In point mode, `gaze_key`
  must be non-null; in heatmap mode, `heatmap_key` must be non-null.
- `convert_open_gaze_manifest.py --help` keeps OpenCV, zarr, NumPy, and gaze utility imports lazy
  so manifest-conversion arguments can be reviewed in lightweight environments. Actual image/
  heatmap loading and zarr writing still require the normal conversion runtime.

- The manifest can be CSV, JSON, or JSONL.
- Supported point-label columns include `image_path`, `gaze_x`, `gaze_y`, optional
  `image_width/image_height`, and optional `episode_id`.
- Supported dense-label columns include `image_path`, `heatmap_path`, and optional `episode_id`.
- Relative paths are resolved against the manifest directory or `--root-dir`.
- The converter writes the standard open zarr schema consumed by `GazeWamOpenDataset`.
- For video-style open gaze data, first export frames into the generic manifest format:

```text
py scripts/export_video_gaze_manifest.py \
  --metadata path/to/video_gaze_metadata.csv \
  --output-manifest data/open_gaze_manifest.csv \
  --frames-dir data/open_gaze_frames \
  --root-dir path/to/videos \
  --no-gaze-is-normalized \
  --output-zarr data/gaze_wam_open.zarr \
  --zarr-image-size 256 256 \
  --image-resize-mode stretch \
  --overwrite
```

- The video exporter supports rows with `video_path`, `frame_idx` or `timestamp`, and gaze point
  labels. This covers the common Ego-Exo4D-style metadata shape once its raw field names are mapped
  into those aliases.
- `export_video_gaze_manifest.py --help` keeps video/image and optional zarr-conversion imports
  lazy. Actual frame extraction still requires OpenCV/NumPy, and `--output-zarr` also requires the
  open-manifest conversion runtime.
- For dataset-specific metadata with different or nested field names, first inspect a small sample
  and write a reviewable field-mapping report:

```text
py scripts/inspect_open_video_gaze_metadata.py \
  --metadata path/to/raw_video_gaze.jsonl \
  --sample-rows 200 \
  --output-json data/open_video_gaze_metadata_inspect.json
```

- The inspector reads CSV/JSON/JSONL metadata without requiring video-decoding dependencies, flattens
  nested fields, reports candidate dotted paths for `video_path`, `episode_id`, `frame_idx`,
  `timestamp`, `gaze_x`, `gaze_y`, `image_width`, and `image_height`, and emits a
  `suggested_key_map`, `mapping_status`, copyable `adapter_args`, and an
  `adapter_command_template` when confidence is sufficient. The command template can include
  placeholder canonical metadata, manifest, frame directory, video root, and open-zarr paths for
  first-real-open-dataset review.
- Review the suggested key map and split/filter candidates, then canonicalize the metadata with the
  configurable adapter:

```text
py scripts/adapt_open_video_gaze_metadata.py \
  --metadata path/to/raw_video_gaze.jsonl \
  --output-metadata data/open_video_gaze_canonical.csv \
  --key-map '{"video_path":"clip.path","episode_id":"clip.uid","frame_idx":"frame.number","gaze_x":"gaze.point.0","gaze_y":"gaze.point.1"}' \
  --filter split=train \
  --output-manifest data/open_gaze_manifest.csv \
  --frames-dir data/open_gaze_frames \
  --root-dir path/to/videos \
  --no-gaze-is-normalized \
  --output-zarr data/gaze_wam_open.zarr \
  --zarr-image-size 256 256 \
  --overwrite
```

- The adapter supports CSV/JSON/JSONL input, dotted nested keys such as `gaze.point.0`, repeated
  `--filter KEY=VALUE` row filters, `--limit` for smoke tests, and optional direct chaining into
  frame extraction and open-zarr export.
- `adapt_open_video_gaze_metadata.py --help` does not import the video exporter, so metadata
  mapping arguments can be reviewed without OpenCV. Passing `--output-manifest` or `--output-zarr`
  still loads the exporter/converter runtime.
- When any raw video-metadata override is supplied, `prepare_open_gaze_wam_zarr.py` routes through
  the adapter stage before manifest export so timestamp and other mapped fields are preserved
  consistently.
- The standalone converter/exporter/adapter remain useful for debugging one stage at a time; the
  prepare command is the default path before training.
- Real Ego-Exo4D integration still requires choosing the exact camera stream, split, clip subset,
  and raw metadata fields, but the Gaze-WAM side now has a reusable mapping/export path.
- `prepare_open_gaze_wam_zarr.py --help` keeps metadata-adapter/export/conversion/validation/
  preview imports lazy, so open-data argument review works in lightweight environments. Actual
  open zarr preparation still requires the normal video/image, zarr, numeric, and dataset runtime
  dependencies. `--dry-run` stays lightweight for manifest review and only imports the metadata
  inspector when video-metadata inspection is requested.

Schema validation:

- Use the Gaze-WAM zarr validator before training on a new robot or open-source dataset:

```text
py scripts/validate_gaze_wam_zarr.py \
  --dataset-path data/gaze_wam_robot.zarr \
  --dataset-type robot \
  --action-horizon 16 \
  --n-obs-steps 2 \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --heatmap-token-grid 16 16

py scripts/validate_gaze_wam_zarr.py \
  --dataset-path data/gaze_wam_open.zarr \
  --dataset-type open \
  --action-horizon 16 \
  --n-obs-steps 2 \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --heatmap-token-grid 16 16
```

- The validator checks required keys, episode boundaries, tensor ranks, strict finite/in-bounds
  point gaze labels when present, finite robot action/TCP/gripper arrays, valid non-negative dense
  heatmap labels when present, and an optional adapter sample against the same dataset code used by
  training. The canonical DSNT+JS path requires `gaze_xy`; dense/token heatmap fields are validated
  only when present.
- `validate_gaze_wam_zarr.py --help` keeps heavyweight `numpy`/`zarr`/dataset imports lazy, so
  argument review remains available in lightweight environments. Running real schema validation
  still requires the normal zarr, numeric, and dataset runtime dependencies.
- The dataset adapter repeats the critical sample-time numeric gates even when validation is
  skipped: camera images and robot `action_abs` / `action_base_abs` must be finite. Optional dense
  heatmaps, when present, must be finite and non-negative before tokenization.
- Validator scalar arguments such as `action_dim` and `sample_index` are checked before adapter
  sampling. `action_dim` must be positive and `sample_index` must be non-negative, so bad CLI or
  programmatic inputs are reported in the validation JSON instead of silently selecting an
  unintended sample.
- Its JSON summary also reports episode-length statistics, how many unpadded action chunks are
  available under the requested `n_obs_steps/action_horizon/n_latency_steps`, image layout and
  numeric range, robot action/TCP/gripper/heatmap numeric summaries, and open dense-heatmap numeric
  summaries. Treat warnings about no unpadded chunks or out-of-range images as real-data onboarding
  blockers unless you are intentionally relying on padding or custom image preprocessing.
- The JSON summary echoes zarr metadata attributes from `meta.attrs` when present. If metadata
  explicitly records `dataset_type`, `image_resize_mode`, or `image_size`, those values must match
  the validator's `--dataset-type`, `--image-resize-mode`, and `--image-size` arguments;
  mismatches fail validation instead of being left as ambiguous provenance.
- Adapter sample summaries in validator, preflight, and preview reports include
  `use_gaze_condition` and `is_gaze_condition_dropped` so real gaze vs `[MASK]` routing can be
  audited before training. Validator and preflight sample summaries also expose optional metadata
  shapes and row-level `has_*` values when `action_abs`, `action_base_abs`, or `heatmap_image` are
  present, proving that zarr presence-mask arrays survive adapter sampling.
- If timestamp arrays are present or requested, the validator checks finite nondecreasing values and
  can compare modality-specific timestamp streams against a base stream:

```text
py scripts/validate_gaze_wam_zarr.py \
  --dataset-path data/gaze_wam_robot.zarr \
  --dataset-type robot \
  --timestamp-key timestamp \
  --image-timestamp-key image_timestamp \
  --robot-state-timestamp-key robot_state_timestamp \
  --action-timestamp-key action_timestamp \
  --gaze-timestamp-key gaze_timestamp \
  --timestamp-max-delta 0.02 \
  --timestamp-max-step 0.08 \
  --require-timestamps
```

- `--require-timestamps` is recommended for real robot datasets once timestamp fields are preserved
  by canonicalization. Synthetic/debug zarrs may omit timestamps.
- `--timestamp-max-step` catches single-stream timestamp gaps or frame drops before they silently
  become observation/action misalignment.
- For robot zarrs, the adapter sample check also verifies relative-action roundtrip consistency by
  converting the sampled relative `action` back through `action_base_abs` and comparing it to
  `action_abs`. The default tolerance is `1e-4`; pass `--skip-action-roundtrip-check` only when
  debugging legacy data. This catches pose-representation or non-finite action issues, but it does
  not replace real timestamp/synchronization validation because both directions use the sampled
  `action_base_abs`.
- Point gaze outside `[0, 1]` or containing NaN makes validation fail; pixel-space gaze should be
  normalized during prepare/canonicalize rather than passed directly to training zarrs.
- After validation, preview a few samples before training:

```text
py scripts/preview_gaze_wam_dataset.py \
  --dataset-path data/gaze_wam_robot.zarr \
  --dataset-type robot \
  --output-dir data/preview/gaze_wam_robot_sample0 \
  --sample-index 0 \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --heatmap-token-grid 16 16

py scripts/preview_gaze_wam_dataset.py \
  --dataset-path data/gaze_wam_open.zarr \
  --dataset-type open \
  --output-dir data/preview/gaze_wam_open_sample0 \
  --sample-index 0 \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --heatmap-token-grid 16 16
```

- The preview tool writes `rgb.png`, `rgb_gaze.png`, `heatmap.png`, `overlay.png`, and
  `summary.json`.
- `preview_gaze_wam_dataset.py --help` keeps heavyweight `cv2`/`numpy`/`torch`/dataset imports
  lazy, so argument review remains available in lightweight environments. Writing actual preview
  artifacts still requires the normal image, zarr, numeric, and dataset runtime dependencies.
- This is the recommended first check for gaze normalization mistakes, resized-image alignment
  errors, dense heatmap pooling issues, and token-to-image decode problems.
- Before launching long training or ablation jobs, run a local preflight check:

```text
py scripts/preflight_gaze_wam.py \
  --config-name train_gaze_wam_debug_workspace \
  --device cpu \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --output-json data/outputs/preflight_debug.json
```

- The preflight check composes the Hydra config, instantiates robot/open datasets, optionally runs
  zarr validation, instantiates the policy and normalizer, builds one mixed batch, and runs
  `compute_loss_components`. When loss smoke is enabled, the JSON artifact includes a
  `loss_smoke.routing` block with robot/open row counts, real-gaze versus masked-gaze robot counts,
  and per-source action/heatmap loss-mask counts. This makes the first batch's supervision routing
  reviewable without reading tensors by hand.
- Preflight policy-contract output also includes `normalizer_contract`, which states that the
  action normalizer is fitted from `GazeWamRobotDataset` relative actions only and excludes
  open-source zero action placeholders. `_check_policy_contract()` validates the same fields, so a
  missing contract or a source/key/action-dim mismatch fails preflight before long policy training.
- Preflight records `dataset_lengths` through the same lightweight dataset-length helper used by
  direct workspace launches. Robot train samples must be positive; if open-source training is
  enabled by `open_dataloader.batch_size > 0`, open train samples must also be positive. Empty
  validation sets remain allowed and are recorded as zero validation samples.
- Preflight also constructs safety dataloaders with `num_workers=0`, `pin_memory=false`, and
  `persistent_workers=false`, then records `dataloader_batches`. This mirrors the workspace
  zero-batch gate before expensive runs: dataset samples can be positive while `batch_size` and
  `drop_last` still produce no train batches, and that must fail during preflight.
- The dataloader batch-count helper lives in lightweight
  `diffusion_policy.common.gaze_wam_dataloader_checks`, so preflight and launcher-adjacent checks
  can reuse the same zero-batch contract without importing the full workspace runtime. Its
  `open_batch_size` gate delegates to the shared non-negative integer normalizer used by the
  training-config parser, keeping preflight/workspace batch-count checks aligned on booleans,
  fractions, non-finite values, negatives, and string-form integers.
- After the training-loop config validates, preflight applies the same workspace normalization
  helper used by `TrainGazeWamWorkspace.run()`. String-form Hydra overrides for batch sizes,
  dataloader booleans, worker counts, accumulation, and loop cadence therefore resolve to native
  Python types before dataset, dataloader, policy-contract, and loss-smoke checks.
- If the training-loop config is invalid, preflight keeps the structured `training_config.errors`
  plus lightweight config/sampling/image summaries, then marks robot/open dataset summaries,
  `dataset_lengths`, `dataloader_batches`, zarr validation, `policy_contract`, and `loss_smoke` as
  skipped. This keeps malformed overrides such as `open_dataloader.batch_size=oops` as clean JSON
  configuration failures instead of mixing them with downstream dataset or policy noise.
- `preflight_gaze_wam.py --help` keeps heavyweight Hydra/Torch/dataset imports lazy, so argument
  review remains available in lightweight environments. Running preflight still requires the normal
  training dependencies and dataset access.
- In checkpoint mode, preflight also reapplies CLI `--override` values to the checkpoint-loaded
  policy/config path before policy-contract checks. This keeps checkpoint preflight aligned with
  real robot/open zarr path overrides instead of silently using only the paths saved at train time.
- When zarr validation is enabled, the JSON artifact also includes a top-level
  `zarr_presence_masks` block that mirrors robot/open validator `presence_masks` counts. Launch
  reports include this nested preflight field, so optional metadata coverage is reviewable before
  long policy-training runs.
- Its `policy_contract` summary records the visual encoder model/pretraining source/patch size,
  local checkpoint/cache path existence, transform normalization stats, aggregation/downsample
  ratio, total and per-frame visual token counts, expected heatmap-token counts, inferred patch
  size, head dimensions, block-mask setting, model-sourced attention contract, policy-sourced
  loss-routing contract, and source batch ratios. It fails fast if the DINO/ViT patch-token count,
  patch size, checkpoint normalization, attention policy, loss-routing masks, row supervision
  semantics, or action/heatmap sequence lengths drift away from the `16 x 16 = 256`
  heatmap-token contract. When `policy.obs_encoder.pretrained=true`, standalone preflight also
  requires a configured existing local DINO source via `policy.obs_encoder.checkpoint_path` or
  `policy.obs_encoder.cache_dir`; checkpoint paths must point to files and cache directories must
  point to directories. Preflight records both fine-grained path fields and the aggregate
  `obs_encoder_local_weight_source_valid` flag so the local weight-source contract matches
  experiment-plan and ablation-metric provenance. These policy-contract boolean fields use the
  shared strict parser, so string values such as `"false"` or `"off"` cannot accidentally enable
  pretrained-source gates through Python truthiness. Debug configs with `pretrained=false` still
  record the fields without failing on an intentionally empty local source.
- For a dependency-light DINO source/preprocess check before full Hydra preflight, run
  `py scripts/verify_gaze_wam_dino_source.py --checkpoint-path <dinov3.ckpt>` or pass
  `--cache-dir <hf-cache-dir>`. This standard-library gate reads the main config YAMLs, verifies
  the `vit_base_patch16_dinov3` identity, `256 x 256` / patch-16 / 256-token geometry,
  checkpoint normalization stats, and local checkpoint/cache path structure without importing
  torch, timm, Hydra, zarr, or OpenCV. It does not prove that timm can load the weights; that remains
  the job of full preflight/real training in the target environment.
- The top-level config summary also records the robot/open temporal sampling contract:
  `n_obs_steps`, observation downsample steps, action horizon, `n_latency_steps`, action downsample
  steps, and action padding. Use this when reviewing real zarr launches so latency or downsampling
  overrides are visible in the preflight artifact. Preflight fails if robot/open dataset
  `n_obs_steps`, `action_horizon`, or `n_latency_steps` differ from the task-level values;
  downsample and padding fields remain dataset-level sampling strategy fields that are recorded for
  review. Workspace and preflight sampling summaries use the same strict integer parsers as the
  dataset constructors: positive fields reject booleans, fractions, non-finite values, and
  non-integer strings, while `n_latency_steps` is strictly non-negative and may be zero.
- The standalone `image_geometry` summary records task/robot/open resize modes and image sizes. It
  uses the shared strict positive-integer sequence parser before comparison and fails fast unless
  all three use the same `image_resize_mode=stretch` and robot/open dataset `image_size` values
  match `task.image_shape[-2:]`.
- Use this as the local gate before expensive multi-GPU runs or checkpoint evaluation.
- Every Gaze-WAM workspace run writes `training_contract.json` in the Hydra output directory before
  optimization starts. This artifact records dataset paths and split sizes, robot/open per-process
  batch sizes and ratios, `gradient_accumulate_every`, Accelerator `num_processes`,
  `mixed_precision`, distributed type, effective robot/open/total batch size per optimizer step,
  robot gaze-dropout routing, policy-sourced row/loss-mask semantics, loss weights/objective,
  action/image/heatmap token shapes, sequence lengths, block-attention invariants read from the
  instantiated transformer when available, DINO local weight-source configuration/path-existence
  and file-vs-directory checks, zarr `meta.attrs` provenance for robot/open dataset type, resize
  mode, and image size, action-normalizer provenance proving robot-relative-only fitting and
  open-dummy-action exclusion, the `data_stream` contract proving two distinct zarr roots, the
  expected robot/open dataset classes, separate source dataloaders, and online
  `build_gaze_wam_mixed_batch` composition rather than an offline merged zarr,
  optional metadata/presence-mask semantics plus first-sample shape/mask audits, and whether the
  config exactly matches the canonical first-run main method. The contract also checks that task,
  robot, and open dataset resize modes are all `stretch` and consistent, that robot/open dataset
  `image_size` values match `task.image_shape[-2:]`, that robot/open dataset `n_obs_steps`,
  `action_horizon`, and `n_latency_steps` match the task sampling contract, and that enabled zarr
  sources expose matching `dataset_type`, `image_resize_mode`, and `image_size` metadata. The first
  normalizer provenance is also folded into `checks`, so `canonical_main_config_ok` covers
  robot-relative-only normalizer source, robot camera/action normalizer keys, 10D action stats, and
  open-dummy-action exclusion. The first JSON log row records
  the contract path, source ratios, Accelerator process count, effective train batch size, and key
  routing settings, so a later review can identify accidental non-75/25 runs, launcher-bypassed
  DINO source issues, mismatched zarr provenance, or mismatched multi-GPU batch scale even if
  training was launched directly through `train.py` instead of the guarded launcher.
- The workspace also stamps the same training-scale fields into the checkpoint Hydra config before
  saving checkpoints. Checkpoint-based metric runs can therefore reconstruct the training-scale
  provenance from `payload["cfg"]` and keep metric `provenance_contract_id` aligned with the
  matching `training_contract.json` and experiment-plan row. When checkpoint metrics are run with
  eval-time dataloader overrides, provenance prefers the stamped training batch sizes/effective
  batch fields over the overridden eval dataloader batch sizes, so evaluation convenience settings
  do not mutate the recorded training contract.
- The workspace also performs its own dataset-length gate before DataLoader construction and
  normalizer fitting. Any source with a positive train batch size must have positive train samples:
  robot for mixed/robot-only runs, open-source for mixed/open-only runs. Empty validation sets are
  allowed and simply skip the corresponding validation stream. This gate uses the same lightweight
  helper and strict non-negative integer parsing for its robot/open batch-size enable switches as
  the dataloader batch-count gate.
- Workspace, preflight, and real-data launcher readiness share the same training-loop config gate.
  It checks non-negative robot/open train and validation batch sizes with a positive train-source
  sum, requires `val_robot_dataloader.batch_size=0` when robot training is disabled, and checks positive
  `gradient_accumulate_every`, `num_epochs`, and `checkpoint_every`, valid optional
  `max_train_steps`/`max_val_steps`, non-negative validation/sample/GDR cadence fields, and
  non-negative tqdm interval before Accelerator/DataLoader setup. It also validates all four
  dataloader runtime blocks (`robot`, `open`, `val_robot`, `val_open`) for non-negative
  `num_workers`, boolean `pin_memory`/`persistent_workers`/`drop_last`, and the PyTorch constraint
  that `persistent_workers=true` requires `num_workers > 0`. Non-integer, non-numeric, or
  non-boolean overrides are reported as structured `training_config.errors` instead of crashing the
  preflight serializer. Boolean values are rejected for integer/float fields instead of being
  treated as `0` or `1`; fractional or non-finite values are rejected for integer fields instead of
  being truncated; and float fields must be finite. Valid string-form integer, finite-float, and
  dataloader-boolean overrides remain accepted. After this gate passes, the workspace writes the
  parsed int/float/bool values back into the run config before constructing Accelerator,
  DataLoader, LR scheduler, tqdm, and train/val cadence logic, so valid string-form overrides do not
  survive into long-running policy training.
  The resulting `training_config` block is copied into `training_contract.json`, preflight JSON,
  and real-data launch readiness reports. In standalone preflight, an invalid `training_config`
  short-circuits dataset, dataloader, zarr, policy-contract, and loss-smoke checks with explicit
  `{"skipped": "invalid training_config"}` artifacts.
- Workspace, preflight, and real-data launcher readiness also share a task-routing config gate for
  `task.robot_gaze_dropout_prob` and `task.robot_heatmap_on_gaze_dropout`. It accepts finite
  dropout probabilities in `[0, 1]` and string-form booleans such as `"true"`/`"false"`, then writes
  native float/bool values back into the run config before mixed-batch construction and
  `training_contract.json` generation. Invalid routing overrides short-circuit standalone preflight
  with `{"skipped": "invalid task_routing_config"}` artifacts, preventing strings like `"false"`
  from being interpreted as truthy Python objects. Real-data launcher readiness reads the parsed
  values from this same `task_routing_config` report instead of re-parsing with raw `float(...)` or
  `bool(...)`, so the policy-training launch gate cannot drift from workspace/preflight routing.
  The task-routing validator itself delegates to the same public unit-interval float and strict
  boolean normalizers used by the mixed-batch builder, so the report, workspace, preflight, and
  launcher share one parser for these fields.
- Standalone preflight emits the matching `data_stream_contract` before dataset instantiation
  whenever the training-loop and routing configs are valid. It fails if robot/open zarr paths are
  not distinct, if the dataset classes drift from `GazeWamRobotDataset`/`GazeWamOpenDataset`, or if
  the source ratio is no longer defined by `robot_dataloader.batch_size/open_dataloader.batch_size`
  and online `build_gaze_wam_mixed_batch` composition.
- Real-data launch readiness, the DINOv3 source verifier, and the experiment-plan generator use the
  same strict boolean parser for policy-training gates such as `training.debug`,
  `task.robot_heatmap_on_gaze_dropout`, `policy.obs_encoder.pretrained`, and
  `policy.use_block_attention_mask`. This keeps quoted Hydra overrides like `"false"` from passing
  readiness checks or provenance summaries through Python string truthiness. This is a
  policy-training launch guardrail; runner and hardware-deployment code remain outside the current
  scope.
- Real-data launch readiness also routes core geometry and sampling fields through the shared
  positive/non-negative integer normalizers before building the launch report. This covers
  `task.image_shape`, robot/open `image_size`, task/robot/open `n_obs_steps`, `action_horizon`,
  `n_latency_steps`, `task.action_dim`, `task.heatmap_token_grid`, `task.heatmap_num_tokens`, and
  `policy.obs_encoder.downsample_ratio`. Invalid values are reported through
  `core_config_parse_valid=false` instead of being truncated by raw `int(...)` calls or crashing
  before the policy-training readiness artifact can be reviewed.
- The same parser is also applied before policy/encoder construction and for workspace booleans
  used before the main `run()` training-config gate, including `training.use_ema`,
  `training.resume`, `training.debug`, `training.freeze_encoder`,
  `training.save_val_heatmap_preview`, checkpoint save toggles, `policy.use_frame_embedding`, and
  obs-encoder `pretrained`/`frozen`/normalization sharing toggles. This prevents a quoted
  `policy.obs_encoder.pretrained="false"` or `policy.use_block_attention_mask="false"` override
  from changing model construction before readiness checks can catch it.
- Preflight and offline evaluation call the same early-boolean normalization immediately after
  composing or checkpoint-merging configs and before policy instantiation, so preflight, eval,
  ablation tables, workspace construction, and launch readiness all share one boolean contract.
- After datasets and dataloaders are constructed, the workspace separately checks actual dataloader
  batch counts before training starts. This catches real-data settings where a dataset is non-empty
  but `batch_size`/`drop_last` leaves the robot or enabled open-source train loader with zero
  batches. The resulting per-epoch batch counts are recorded in
  `training_contract.json.dataloader_batches`. The shared lightweight helper strictly parses its
  `open_batch_size` gate as a non-negative integer, so boolean, fractional, non-finite, negative, or
  non-integer string values cannot silently change whether open-source training is considered
  enabled.
- The real workspace training path uses the normalized `cfg.open_dataloader.batch_size` and
  `cfg.val_open_dataloader.batch_size` fields for open train/validation gating. It must not
  re-parse raw Hydra values with `int(...)` after validation.
- Each training JSON log row also records `train_routing_*` fields derived from the same source-wise
  routing summary as preflight. These fields expose robot/open row counts, real-gaze versus
  masked-gaze robot counts, and per-source action/heatmap mask counts for the actual mixed batch
  used by that optimizer step.
- With gradient accumulation, `global_step`, `accelerator.log`, local JSON logging, EMA updates,
  LR scheduler steps, and `training.max_train_steps` all follow completed optimizer steps rather
  than raw micro-batches. This keeps `train_acc8_amp` logs/checkpoints comparable when
  `gradient_accumulate_every > 1`. Per-step training loss, action/heatmap mask counts, and
  `train_routing_*` counts are aggregated over all micro-batches in the completed accumulation
  window; `train_accumulated_microbatches` records the number of local micro-batches represented by
  that optimizer-step log row.
- The robot dataloader remains the epoch driver. Open train/val batches are streamed through a
  restart-on-exhaustion iterator rather than `itertools.cycle`, so long runs do not cache a full
  open-source epoch in memory. When the open iterator restarts, it rebuilds the dataloader iterator
  and therefore keeps the dataloader's shuffle semantics for short open datasets. This behavior is
  recorded in `training_contract.json.batch_streaming`.
- Under Accelerate/DDP, local JSON logging is rank0-only to avoid multiple workers appending to
  the same `logs.json.txt`. W&B logging still goes through `accelerator.log`, while count-like
  training diagnostics such as `train_routing_*`, `train_*_mask_count`, and `train_gdr_count`
  use cross-rank reductions before they are emitted.
- Training GDR diagnostics must avoid rank-local collective branches. All ranks first compute the
  global eligible-row count, and when it is positive they contribute local GDR sums or zero to a
  cross-rank sum before logging `train_feature_gdr` and `train_output_gdr` as global means.

For a one-command debug smoke gate that exercises data generation, schema validation, and preflight
loss, use the smoke pipeline as an optional extended check. The smoke pipeline defaults to the
policy-training-only path; optional deployment rehearsal outputs are reference-only for a later
live-deployment phase:

```text
py scripts/gaze_wam_smoke_pipeline.py \
  --config-name train_gaze_wam_debug_workspace \
  --output-dir data/outputs/gaze_wam_smoke_pipeline \
  --debug-data-dir data/debug_gaze_wam_smoke_pipeline \
  --device cpu \
  --policy-only \
  --num-inference-steps 1 \
  --fail-on-zarr-warning
```

- The pipeline writes debug robot/open zarr data, validates both zarrs, runs
  `preflight_gaze_wam`, and writes a top-level `summary.json` plus `preflight.json`.
  By default, and also with explicit `--policy-only`, it stops there and records
  `rehearsal=None` / `split_rehearsal=None` in the summary so the gate stays aligned with the
  current policy-training milestone.
- If a later deployment pass needs the reference rehearsal path, opt in with
  `--with-deployment-rehearsal`. That additionally runs config-mode zarr and split-provider
  deployment rehearsals and writes `rehearsal.json`, `split_rehearsal.json`,
  `split_rehearsal_config.json`, and `split_commands.jsonl`; review these only as optional
  reference artifacts for later deployment.
- `generate_gaze_wam_debug_data.py --help` and `gaze_wam_smoke_pipeline.py --help` keep
  heavyweight zarr, numeric, image/video, preflight, and rehearsal imports lazy, so smoke/debug
  argument review remains available in lightweight environments. Actually writing debug zarrs or
  running the smoke gate still requires the normal project runtime dependencies.
- For real zarr smoke checks, pass `--skip-generate-debug-data --robot-dataset-path ...`
  and `--open-dataset-path ...`.
- For real synchronized datasets, add `--require-timestamps`, `--timestamp-max-delta`, and
  `--timestamp-max-step`; the smoke pipeline applies those gates to both direct zarr validation
  and the nested `preflight_gaze_wam` call.
- Add `--fail-on-zarr-warning` for real-data smoke checks to promote validator health warnings
  such as short episodes, no unpadded action chunks, or out-of-range images into a failed gate.

### 8.3 Heatmap Label Generation

If only `(x, y)` gaze points exist:

- Generate a Gaussian target directly on the `16 x 16` heatmap-token grid.
- Each token corresponds to a `16 x 16` image patch under the `256 x 256` image setting.
- Keep token labels in `[0, 1]` for first-version MSE diffusion.

If full-resolution labels exist:

- Direct-stretch resize labels to `256 x 256`, matching the RGB image geometry.
- Area/average-pool them into `16 x 16` token labels.
- Optionally renormalize after pooling.

For visualization:

- Decode token labels or predictions to `256 x 256` heatmaps.
- Prefer Gaussian splatting from token centers for paper figures.

## 9. Normalization Plan

Reuse `LinearNormalizer`.

Action:

- Normalize relative action targets, not stored absolute TCP trajectories.
- Keep `action_base_abs` outside the model normalizer path.
- The action normalizer must be fit from `GazeWamRobotDataset` relative actions only. Open-source
  datasets contain zero-filled dummy actions for shape compatibility and must not fit or contribute
  to the action normalizer.
- Use UMI-style quantile normalizer if `use_quantiles=True`.

Image:

- Use the image normalization expected by the visual encoder.
- Main image shape is `[3, 256, 256]`.

Gaze coordinate:

- Use normalized `[0, 1]` coordinates.
- Do not apply additional learned normalization to `gaze_xy`.

Heatmap tokens:

- Identity normalization for first-version max-normalized token maps.
- Later test sum-normalized distributions or `[-1, 1]` scaling only if diffusion training is
  unstable.

## 10. Config Contract

The dedicated config path is:

```text
diffusion_policy/config/train_gaze_wam_workspace.yaml
```

Current executable targets:

```yaml
_target_: diffusion_policy.workspace.train_gaze_wam_workspace.TrainGazeWamWorkspace

policy:
  _target_: diffusion_policy.policy.gaze_wam_policy.GazeWamPolicy

task:
  robot_dataset:
    _target_: diffusion_policy.dataset.gaze_wam_dataset.GazeWamRobotDataset

  open_dataset:
    _target_: diffusion_policy.dataset.gaze_wam_dataset.GazeWamOpenDataset
```

Core config semantics:

- Executable YAML splits these values across `task.*`, `policy.*`, `robot_dataloader.*`,
  `open_dataloader.*`, and `training.*`.
- The full mixed config keeps `policy.obs_encoder.pretrained=true`. Smoke/debug configs may override
  this to `false` only for local wiring checks.
- The task config sets `image_resize_mode: stretch`; current datasets and open-data converters
  reject crop/letterbox modes because those modes require explicit gaze and heatmap remapping.
- The 75/25 source ratio is currently controlled by dataloader batch sizes:
  `robot_dataloader.batch_size=48` and `open_dataloader.batch_size=16` in the full mixed config,
  and `3`/`1` in the debug mixed config. These are the source composition contract for one
  optimizer step before gradient accumulation.
- `heatmap_horizon=1` is an implicit first-version constraint: datasets emit `[B, 1, 256, 1]`,
  while `GazeWamPolicy.compute_loss_components` collapses it to `[B, 256, 1]`.
- Robot and open datasets use deterministic episode-level train/validation splits through
  `val_ratio` and `seed`. Train and validation samples must not share episodes.
- `seed` must be a non-negative integer and `val_ratio` must be a finite float in `[0, 1)`.
  String-form Hydra values such as `"123"` and `"0.02"` remain accepted, while booleans,
  fractional/non-finite seeds, non-numeric ratios, and `val_ratio=1.0` fail before the episode
  split is built. A ratio of `1.0` is invalid because the training split must retain at least one
  episode by configuration contract, not only by sampler fallback behavior.
- The Gaze-WAM workspace builds matching robot/open validation dataloaders from those validation
  splits, logs `val_loss`, `val_action_loss`, `val_heatmap_loss`,
  `val_heatmap_token_kl_loss`, and mask counts at `training.val_every`, and uses `val_loss` as
  the default top-k checkpoint monitor when validation metrics are available. The validation
  losses are computed as global masked averages, not per-rank or per-batch means.
- Validation also logs source-split diagnostics:
  `val_robot_action_loss`, optional `val_robot_heatmap_loss`, `val_open_heatmap_loss`, and matching
  mask counts. These are diagnostic metrics only; top-k checkpointing still uses the weighted
  combined `val_loss` by default.
- Validation mixed-batch routing uses a deterministic generator derived from
  `training.val_mixing_seed`, epoch, and validation batch index. This keeps robot gaze-dropout
  masks reproducible for validation and top-k checkpoint ranking while leaving training dropout
  stochastic.
- When `training.save_val_heatmap_preview=true`, the first validation batch with heatmap
  supervision writes up to `training.val_heatmap_preview_max_samples` local diagnostic samples
  under `media/val_heatmap/epoch_XXXX/sample_YYY/`: RGB frame, predicted heatmap, target heatmap,
  predicted overlay, and target overlay. The epoch directory also keeps the first sample under the
  legacy top-level filenames (`rgb.png`, `pred_heatmap.png`, `target_heatmap.png`,
  `pred_overlay.png`, `target_overlay.png`) plus a `summary.json` that lists every saved sample.
  The workspace gate reads the normalized `cfg.training.save_val_heatmap_preview` bool and must
  not re-wrap raw Hydra values with `bool(...)`, because quoted `"false"` should stay disabled.
  The preview and GDR diagnostic paths validate optional `noisy_action` / `noisy_heatmap` tensors
  and supplied diffusion timesteps at the policy boundary: noisy tensor shapes must match the
  configured action horizon, action dimension, heatmap token count, and heatmap dimension, values
  must be finite, and timesteps must be integer scalar or `[B]` inputs within the scheduler's
  training timestep range.
- The joint transformer optimizer grouping must cover every named transformer parameter exactly
  once. Gaze-WAM does not require the legacy `_dummy_variable` parameter used by older transformer
  modules; optimizer group construction treats that name as optional so `policy.get_optimizer()`
  can build AdamW cleanly for the policy-training workspace.
- `policy.get_optimizer()` splits visual encoder backbone parameters by the
  `obs_encoder.named_parameters()` prefix `key_model_map`. If a run config supplies a distinct
  `obs_encoder_lr` or `obs_encoder_weight_decay`, at least one backbone parameter must be found
  under that prefix; otherwise policy construction fails before long training silently uses the
  default policy LR for all obs-encoder parameters.

```yaml
image_shape: [3, 256, 256]
image_size: [256, 256]
image_resize_mode: stretch
n_obs_steps: 2
n_latency_steps: 0
seed: 42
val_ratio: 0.02

visual_encoder: dinov3_vit
visual_patch_size: 16
visual_token_grid: [16, 16]
visual_num_tokens_per_frame: 256
visual_num_tokens: ${eval:'${n_obs_steps} * 256'}
temporal_order: time_major_flatten
frame_identity: full_sequence_position_embedding
use_frame_embedding: false
image_tokens_per_frame: 256
max_obs_frames: ${n_obs_steps}
n_emb: 768

action_horizon: 16
action_dim: 10
zarr_action_storage: absolute_tcp
action_representation: relative
relative_action_base: latest_observed_tcp
use_absolute_action: true
relative_action: true

gaze_encoder: gaussian_spatial
gaze_encoder_grid: [8, 8]
gaze_encoder_sigma: 0.15
use_trainable_gaze_mask_token: true
robot_gaze_dropout_prob: 0.2
robot_heatmap_on_gaze_dropout: true

heatmap_horizon: 1
heatmap_token_grid: [16, 16]
heatmap_num_tokens: 256
heatmap_codec: fixed_gaussian_token_codec
heatmap_token_sigma: 1.25
heatmap_loss_source: open_and_robot_gaze_dropout
robot_heatmap_loss_requires_gaze_dropout: true
heatmap_objective: dsnt_js
heatmap_token_kl_loss_weight: 0.0
heatmap_xy_loss_weight: 1.0
heatmap_js_loss_weight: 1.0
heatmap_dsnt_temperature: 1.0

robot_sampling_ratio: 0.75
open_sampling_ratio: 0.25
use_block_attention_mask: true

num_inference_steps: 8

training:
  gradient_accumulate_every: 1
  mixed_precision: bf16
  accelerator_config: accelerate/8gpu-amp.yaml
  use_distributed_masked_mean: true
  val_every: 1
  max_val_steps: null
  val_mixing_seed: 100003
  save_val_heatmap_preview: true
  val_heatmap_preview_max_samples: 4

checkpoint:
  topk_monitor_key: val_loss
```

Scalar policy guardrails:

- `policy.input_pertub`, `policy.action_loss_weight`, `policy.heatmap_loss_weight`,
  `policy.heatmap_token_kl_loss_weight`, `policy.heatmap_xy_loss_weight`,
  `policy.heatmap_js_loss_weight`, and `policy.cfg_scale` must be finite non-negative floats.
  Native booleans are rejected instead of being coerced to `0.0` or `1.0`.
- `policy.heatmap_dsnt_temperature` must be a finite positive float.
- `policy.num_inference_steps` must be an integer greater than or equal to `2`; single-step
  denoising is rejected.
- String-form integers greater than or equal to `2`, such as Hydra override value `"8"`, are
  accepted and normalized at the policy boundary; booleans, fractional floats, non-finite floats,
  one-step values, and non-integer strings are rejected.
- These fail fast during policy construction and when runtime overrides such as evaluation-time
  `cfg_scale` or checkpoint `num_inference_steps` are supplied, so invalid train/inference configs
  are caught before long runs.
- Policy construction, workspace `training_contract.json` generation, standalone preflight,
  real-data launch readiness, and ablation comparison provenance use the same finite non-negative
  float guard for policy loss weights and `cfg_scale`, so readiness/audit records cannot silently
  coerce booleans into scalar values.

Task config semantics should mirror the useful fields from
`W:/umi_base/diffusion_policy/config/task/q3_place_cup_no_tcp.yaml`:

- `shape_meta`
- image observations
- action shape and horizon
- deterministic episode-level validation split fields
- relative-action semantics: the zarr stores absolute TCP, while the dataset emits relative
  model targets and preserves `action_base_abs` for inverse conversion.
- dataset path / zarr path
- policy/evaluation metadata fields for relative-to-absolute action conversion

The current executable Gaze-WAM config encodes these semantics through `task.robot_dataset`,
`action_abs_key`, `tcp_pose_key`, `gripper_key`, and the dataset-side conversion path rather than
requiring the literal UMI YAML keys `relative_action`, `use_absolute_action`, and
`action_representation`. Add literal alias keys only if a future UMI-compatible wrapper needs them.

## 11. Multi-GPU And AMP Training Plan

Gaze-WAM training must use AMP. The preferred precision is bf16, matching `umi_base`:

```text
make train_acc8_amp WKSPACE=<gaze_wam_workspace_config> TASK=<gaze_wam_task_config>
```

Reference behavior from `W:/umi_base/Makefile`:

```text
HF_HUB_OFFLINE=1
HYDRA_FULL_ERROR=1
accelerate launch --config_file accelerate/8gpu-amp.yaml train.py \
  --config-name ${WKSPACE} \
  task=${TASK}
```

Recommended launch gate:

```text
py scripts/launch_gaze_wam_training.py \
  --config-name train_gaze_wam_workspace \
  --task gaze_wam \
  --accelerate \
  --accelerate-config accelerate/8gpu-amp.yaml \
  --real-data \
  --preflight-device cpu \
  --preflight-require-timestamps \
  --preflight-timestamp-max-step 0.08 \
  --preflight-fail-on-zarr-warning \
  --data-onboarding-review-json data/gaze_wam_data_onboarding_review.json \
  --require-data-onboarding-review \
  --output-json data/outputs/gaze_wam_launch_report.json
```

- For first-real-data review, `scripts/review_gaze_wam_training_readiness.py` can create a
  higher-level bundle that includes this launcher dry-run report together with the onboarding and
  DINO verifier artifacts. The launcher remains the direct preflight/command builder; the bundle is
  a review wrapper around it.
- The readiness bundle compares standalone DINO-verifier and launcher DINO sub-report signatures
  with the same strict boolean parser used elsewhere, so `"false"`/`"off"` style JSON values for
  `dino_source.pretrained` do not become truthy Python strings during cross-checks.
- The bundle also parses child-stage `ok` fields with the strict boolean parser; a child report with
  `"ok": "false"` is treated as failed instead of truthy.
- By default this command runs `preflight_gaze_wam`, builds the exact train command, writes a JSON
  report, and does not start the long training job.
- `launch_gaze_wam_training.py --help` keeps Hydra/OmegaConf/preflight imports lazy, so launch
  arguments can be reviewed before activating the full training environment.
- Root `scripts/*.py` wrappers for policy training, data preparation, validation, metric evaluation,
  experiment planning, smoke, and readiness review insert the repository root at the front of
  `sys.path` before importing `diffusion_policy.scripts.*`. This prevents an installed or stale
  external `diffusion_policy` package from shadowing the current Gaze-WAM worktree during command
  review or launch.
- The direct-entry bootstrap in `diffusion_policy/workspace/train_gaze_wam_workspace.py` follows
  the same repo-root-first import rule for rare direct workspace execution paths.
- When `--run` is enabled and all gates pass, the launcher writes the report once before
  `subprocess.run(...)` starts the long training command, then rewrites it after the command returns
  with the final return code. This leaves a reviewable command/preflight/readiness artifact even if
  the long job is interrupted.
- The launch report includes an `acceleration` section parsed from the Hydra training config and
  Accelerate YAML: robot/open batch size per process, `gradient_accumulate_every`, number of
  processes, mixed precision mode, and effective robot/open/total batch size per optimizer step.
  `effective_train_batch_size` is kept as a backward-compatible alias for
  `effective_train_batch_size_per_optimizer_step`. Review this before launching long multi-GPU jobs.
- When preflight runs, the launch report also includes top-level
  `preflight_routing_validation_guardrails_ok`, extracted from
  `preflight.policy_contract.loss_routing_contract` through the shared loss-routing guardrail
  helper. A false value is launch-blocking, so the command report proves the zero-placeholder and
  source-routing validation contract without requiring a reviewer to inspect nested preflight JSON.
- The same section embeds the shared lightweight `training_config` validation summary from
  `diffusion_policy.common.gaze_wam_training_config` and launch-level training-config `errors`.
  These errors block `--run` even when standalone preflight is skipped, so malformed batch sizes,
  accumulation, dataloader runtime fields, or loop cadence values still produce a reviewable
  launcher report instead of reaching the long training command.
- For physical robot/open-data training, pass `--real-data`. This adds a launch-blocking
  `real_data_readiness` report and requires:
  - non-debug config and `training.debug=false`;
  - preflight, zarr validation, and loss-smoke enabled;
  - `--preflight-require-timestamps`, a finite positive timestamp threshold such as
    `--preflight-timestamp-max-step 0.08`, and `--preflight-fail-on-zarr-warning`; boolean values
    are not accepted as timestamp thresholds;
  - `--output-json` so the launch report is saved before any long run starts; the path must point
    to a report file candidate, not an existing directory or a path whose nearest existing parent is
    not a directory;
  - when `--require-data-onboarding-review` is set, `--data-onboarding-review-json` must point to a
    combined dry-run artifact from `scripts/review_gaze_wam_data_onboarding.py`; launch readiness
    checks that this report has `ok=true`, `dry_run=true`, `policy_training_scope=true`,
    `deployment_runner_scope=deferred`, includes both robot and open stages, and matches the
    configured robot/open zarr paths, image geometry, temporal sampling, heatmap grid, and timestamp
    gates. Boolean fields in this JSON are parsed with the strict boolean parser, so string values
    such as `"false"` do not become truthy Python strings;
  - distinct, existing robot/open dataset paths that point to `.zarr` stores and do not look like
    debug, smoke, synthetic, or temp data;
  - robot/open zarr metadata can be read before training and `meta.attrs.dataset_type`,
    `meta.attrs.image_size`, and `meta.attrs.image_resize_mode` match the expected source
    (`robot` or `open`), `256 x 256`, and `stretch` contract; the launch report embeds these attrs
    under `real_data_readiness.zarr_metadata`;
  - `policy.obs_encoder.pretrained=true`, plus an explicit existing local DINO weight source via
    `policy.obs_encoder.checkpoint_path` or `policy.obs_encoder.cache_dir`; configured checkpoint
    paths must be files, configured cache paths must be directories, and if `cache_dir` is the only
    configured local source it must contain at least one file; the launch report exposes
    `obs_encoder_local_weight_source_valid` plus the cache-content gate result;
  - the launch `real_data_readiness.dino_source_verifier` sub-report passes against the same DINO
    source, image/token geometry, and Normalize stats; the standalone
    `scripts/verify_gaze_wam_dino_source.py` can also write an explicit JSON artifact for review,
    and `--require-cache-files` is available when a cache directory with no files should be
    launch-blocking;
  - `image_resize_mode=stretch` consistently on `task`, robot dataset, and open dataset, with
    robot/open dataset `image_size=[256,256]` matching `task.image_shape[-2:]`;
  - robot/open dataset `n_obs_steps`, `action_horizon`, and `n_latency_steps` matching task-level
    sampling values;
  - for `--real-data-contract main`, `task.n_latency_steps=0` for the first policy-training run;
  - valid training-loop parameters under `real_data_readiness.training_config`, including positive
    robot batch size, positive `gradient_accumulate_every`, positive `num_epochs`, positive
    `checkpoint_every`, non-negative open batch size, non-negative validation/sample/GDR cadence
    values, and valid optional max-step limits;
  - the fixed main tensor/model contract: `image_shape=[3,256,256]`, `n_obs_steps=2`,
    `action_horizon=16`, `action_dim=10`, `heatmap_token_grid=[16,16]`,
    `heatmap_num_tokens=256`, DINOv3 ViT/16 as `vit_base_patch16_dinov3`, and
    `policy.use_block_attention_mask=true`;
  - the main mixed-training contract: positive robot/open source batches with the configured
    `75%` robot / `25%` open-source gaze ratio, `task.robot_gaze_dropout_prob=0.2`,
    `task.robot_heatmap_on_gaze_dropout=true`, `policy.heatmap_objective=dsnt_js`,
    `policy.action_loss_weight=1.0`, `policy.heatmap_loss_weight=1.0`,
    `policy.heatmap_token_kl_loss_weight=0.0`, `policy.heatmap_xy_loss_weight=1.0`, and
    `policy.heatmap_js_loss_weight=1.0`;
  - Accelerate multi-process bf16 training with an existing Accelerate config.
- `--real-data-contract main` is the default for a single main-method launch and enforces the
  fixed 75/25 ratio, dropout, and DSNT+JS heatmap settings above. For planned ablation training,
  use `--real-data-contract ablation`; it keeps the real-data safety gates (preflight, timestamps,
  zarr warnings, DINO local source, geometry, tensor contract, and Accelerate) while allowing
  planned variants such as robot-only, no-gaze, no-block-mask, heatmap-objective, and source-ratio
  sweeps.
- `--data-onboarding-review-json` is optional unless paired with
  `--require-data-onboarding-review`. Supplying it always embeds a `data_onboarding_review` block
  inside `real_data_readiness`, so launch reports can prove the data-prep review and training config
  refer to the same canonical zarr outputs.
- Add `--run` only after the preflight report is clean.
- Use `--preflight-require-timestamps`, `--preflight-timestamp-max-delta`, and
  `--preflight-timestamp-max-step` to make the launch report enforce the same timestamp gates as
  the standalone preflight CLI.
- Use `--preflight-fail-on-zarr-warning` for real training launches so validator health warnings
  block the launch report before expensive multi-GPU jobs.
- Add repeated `--override KEY=VALUE` entries for dataset paths, DINO checkpoint toggles, debug
  limits, or ablation settings.
- `--no-accelerate` is only for short CPU/shape debug paths that explicitly set
  `training.require_amp=false`. It is not a valid path for real robot/open-data training or
  open-only HOT3D heatmap pretraining.

Reference Accelerate config:

```yaml
distributed_type: MULTI_GPU
num_processes: 8
gpu_ids: 0,1,2,3,4,5,6,7
mixed_precision: bf16
```

Current implementation requirements:

- `accelerate/8gpu-amp.yaml` and the `train_acc8_amp` Makefile target exist in `gaze-wam`; keep
  them aligned with the UMI-style launch convention.
- Non-debug training configs set `training.require_amp=true`; the workspace checks
  `accelerator.mixed_precision` and fails before dataloader/model training if it is not `bf16` or
  `fp16`. The HOT3D open-only PowerShell entrypoint launches
  `accelerate --mixed_precision bf16` by default.
- Keep `gradient_accumulate_every` in the workspace config.
- Use bf16 before fp16 for DINO/DiT stability.
- Diffusion loss noise tensors for action and heatmap targets inherit the corresponding target
  dtype, so bf16/AMP training does not silently introduce default float32 noise in the hot path.
- Masked losses use cross-rank numerator/denominator reduction; keep this covered when changing
  loss routing.
- Log effective global batch size:

```text
effective_batch = per_gpu_batch * num_processes * gradient_accumulate_every
```

Risks to test:

- Heatmap token loss scale under bf16.
- DINOv3 checkpoint loading in offline/HF cache mode.
- Mixed robot/open batch ratio under DDP sharding.

## 12. Experiment Matrix

### 12.1 Main Variants

Baseline:

- Robot-only action diffusion.
- Uses real gaze condition if available.
- No open-source heatmap auxiliary task.
- Full config: `diffusion_policy/config/train_gaze_wam_robot_only_workspace.yaml`.
- Debug config: `diffusion_policy/config/train_gaze_wam_robot_only_debug_workspace.yaml`.
- This disables open-source data with `open_dataloader.batch_size=0` and sets
  `robot_gaze_dropout_prob=0.0` and `robot_heatmap_on_gaze_dropout=false`, so robot rows train
  action only even when a robot zarr contains dense heatmap labels but no point-gaze key.

Ours (Gaze-WAM):

- Mixed batch: `75%` robot, `25%` open-source gaze.
- Robot rows always train action loss.
- Robot gaze-dropout rows also train heatmap token loss under the `[MASK]` condition.
- Open-source rows train heatmap token loss only.
- Uses trainable `[MASK]` token for open-source gaze condition.
- Uses block attention mask.

Ours (CFG):

- Adds mask-token based classifier-free guidance for action inference.
- Full config: `diffusion_policy/config/train_gaze_wam_cfg_workspace.yaml`.
- Debug config: `diffusion_policy/config/train_gaze_wam_cfg_debug_workspace.yaml`.
- Sets `policy.cfg_scale=1.5`; direct eval/compare CLIs can still override all variants with
  `--cfg-scale`.
- `GazeWamPolicy.predict_action` returns scalar metadata `cfg_scale` and `cfg_enabled` so policy
  outputs, adapter/replay artifacts, and ablation review can confirm which CFG setting was actually
  used for a sampled action.
- During inference, combine conditional and masked predictions:

```text
pred = pred_masked + cfg_scale * (pred_gaze - pred_masked)
```

### 12.2 Useful Ablations

No gaze:

- Always use the trainable `[MASK]` token for robot action prediction.
- Full config: `diffusion_policy/config/train_gaze_wam_no_gaze_workspace.yaml`.
- Debug config: `diffusion_policy/config/train_gaze_wam_no_gaze_debug_workspace.yaml`.
- Sets `robot_gaze_dropout_prob=1.0`, `open_dataloader.batch_size=0`, and
  `robot_heatmap_on_gaze_dropout=False`.

Robot gaze-dropout probability sweep:

- `0.0`
- `0.1`
- `0.2`
- `0.3`
- Tests how much robot action supervision the `[MASK]` token needs.

Robot heatmap without dropout diagnostic:

- Enable robot heatmap loss while still feeding real gaze.
- This is expected to be shortcut-prone and should be treated as a diagnostic, not the main method.

No block mask:

- Remove the modality-level block attention mask.
- Tests whether dummy open-source action tokens poison heatmap or shared representations.
- Its attention-contract artifact should explicitly show cross-target/condition reads as enabled,
  distinguishing it from the main method rather than reusing the main block-mask invariants.
- Full config: `diffusion_policy/config/train_gaze_wam_no_block_mask_workspace.yaml`.
- Debug config: `diffusion_policy/config/train_gaze_wam_no_block_mask_debug_workspace.yaml`.

Open ratio sweep:

- 100/0
- 90/10
- 75/25
- 50/50

Heatmap codec ablation:

- Bilinear visualization only.
- Gaussian splatting visualization.
- Optional learned 1-channel lightweight decoder, not in the main method.

Heatmap objective ablation:

- Diffusion-style heatmap target with the same `prediction_type` as action.
- Direct clean-token regression as a simpler auxiliary objective:
  `heatmap_objective=clean_token`.
- Scheduler `prediction_type='epsilon'` vs scheduler `prediction_type='sample'` / `x0`
  prediction if scheduler support is clean. This is separate from
  `heatmap_objective=clean_token`.
- Full clean-token config:
  `diffusion_policy/config/train_gaze_wam_heatmap_clean_token_workspace.yaml`
- Debug clean-token config:
  `diffusion_policy/config/train_gaze_wam_heatmap_clean_token_debug_workspace.yaml`
- Canonical policy values are `diffusion` and `clean_token`.
- Legacy `heatmap_sample` config names remain as compatibility aliases. They pass the old
  `heatmap_objective=sample` value in Hydra, and the policy normalizes it to `clean_token` at
  construction time. New runs and paper tables should use the `heatmap_clean_token` naming.

Inference steps:

- 1
- 5
- 8
- 10
- 16

### 12.3 Metrics

Success Rate:

- Main robotics metric.

Action MSE:

- Use relative action target space.
- Optionally also report absolute TCP reconstruction error after inverse conversion.

Heatmap MSE / KL:

- Token-level gaze quality on the `16 x 16` grid.
- Pixel-space metrics can be computed after decoding to `256 x 256`.
- Offline evaluator reports token MSE, token KL after sum-normalization, and token-grid argmax
  coordinate error.

Gaze-Dependency Ratio:

- Compare action feature or prediction change with real gaze vs mask gaze.
- GDR is defined only for rows with action supervision and a real point-gaze label. Offline
  evaluation filters with source masks before calling the policy, and
  `GazeWamPolicy.compute_gaze_dependency_ratio` rejects unfiltered rows where
  `has_gaze_label=False`.

Candidate definition:

```text
GDR = ||f_action(image, gaze) - f_action(image, mask)||_2
      / (||f_action(image, gaze)||_2 + eps)
```

Better report both:

- feature-level GDR from action token features.
- output-level GDR from predicted relative action trajectories.

Current offline metric entrypoint:

```text
py scripts/eval_gaze_wam_metrics.py \
  --config-name train_gaze_wam_debug_workspace \
  --device cpu \
  --validate-zarr \
  --require-timestamps \
  --timestamp-max-step 0.08
```

- With `--checkpoint`, it loads a saved workspace checkpoint and uses EMA weights by default when
  available.
- Checkpoint evaluation treats the checkpoint's stored Hydra config as the base, then reapplies CLI
  `--override` values before dataset validation, dataset instantiation, and provenance reporting.
  This is required for real-data metrics where the trained checkpoint should be evaluated on
  `task.robot_dataset_path` / `task.open_dataset_path` values that differ from the paths saved at
  train time.
- Without `--checkpoint`, it instantiates the Hydra config and fits the action/image normalizer
  from the configured robot dataset, which is useful for synthetic debug and smoke checks.
- It reports robot action MSE, optional absolute action MSE, heatmap token MSE/KL/argmax error,
  and feature/output GDR where the source masks make those metrics valid.
- Relative action MSE is computed on all rows with `has_action=True`; absolute action MSE is
  computed only on rows where both `has_action_abs=True` and `has_action_base_abs=True`, so
  zero-filled optional metadata placeholders are never converted into absolute predictions.
- The JSON includes explicit supervision and optional-metadata coverage fields such as
  `*_action_supervision_count`, `*_heatmap_supervision_count`, `*_has_action_abs_count`,
  `*_has_action_base_abs_count`, `*_has_heatmap_image_count`, and
  `*_action_abs_metric_eligible_count`. The legacy `*_action_abs_supervision_count` is kept as a
  backward-compatible alias for the same absolute-action eligibility count.
- Action metadata coverage counts are gated by `has_action=True`, so open-source or other no-action
  rows cannot appear as action-metadata coverage even if dense placeholder tensors or masks are
  present.
- The evaluator uses the same strict mask contract as training: supervision masks and optional
  presence masks must be `BoolTensor[B]` values. It must not numerically coerce `0/1` float or int
  masks while producing paper metrics.
- When evaluating the robot source through the config-level entrypoint, the evaluator mirrors the
  training gaze-dropout routing: it applies `task.robot_gaze_dropout_prob`, keeps action
  supervision on robot rows, and enables robot heatmap supervision only on rows whose gaze condition
  is replaced by the learned `[MASK]` token when `task.robot_heatmap_on_gaze_dropout=true`.
  Direct low-level dataset evaluation can still be used to inspect raw dataset masks without this
  synthetic dropout pass.
- By default the evaluator validates selected robot/open zarr sources before computing metrics and
  includes `*_zarr_validation` summaries in the JSON output. Use `--no-validate-zarr` only for
  narrow debugging.
- For real synchronized data, add `--require-timestamps`, `--timestamp-key`,
  `--timestamp-max-delta`, and `--timestamp-max-step` so paper metrics cannot be computed on
  missing or gappy timestamp streams.

Ablation comparison entrypoint:

```text
py scripts/compare_gaze_wam_ablation_metrics.py \
  --variant main=train_gaze_wam_workspace:path/to/main.ckpt \
  --variant robot_only=train_gaze_wam_robot_only_workspace:path/to/robot_only.ckpt \
  --variant no_gaze=train_gaze_wam_no_gaze_workspace:path/to/no_gaze.ckpt \
  --variant cfg=train_gaze_wam_cfg_workspace:path/to/cfg.ckpt \
  --variant no_block_mask=train_gaze_wam_no_block_mask_workspace:path/to/no_block.ckpt \
  --variant heatmap_clean_token=train_gaze_wam_heatmap_clean_token_workspace:path/to/heatmap_clean.ckpt \
  --validate-zarr \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --output-json data/outputs/gaze_wam_ablation_metrics.json \
  --output-csv data/outputs/gaze_wam_ablation_metrics.csv
```

- Without explicit `--variant`, the script compares the six default configs without checkpoints:
  `main`, `robot_only`, `no_gaze`, `cfg`, `no_block_mask`, and `heatmap_clean_token`.
- Each row includes `variant`, `config_name`, optional `checkpoint`, and the same offline metrics
  and `*_zarr_validation` summaries produced by `eval_gaze_wam_metrics.py`.
- Each CSV/JSON row also records experiment provenance needed for paper tables: whether a
  checkpoint was provided, eval sources and limits, policy/eval/effective CFG scale, robot/open
  batch sizes and ratios, training-scale fields (`gradient_accumulate_every`, Accelerator
  `num_processes`, `mixed_precision`, `distributed_type`, and effective robot/open/total batch size
  per optimizer step), robot gaze-dropout settings, temporal sampling fields such as
  `n_latency_steps`, downsample steps, and action padding, block-mask setting, heatmap objective,
  observation/action/heatmap token shape fields, image shape, task/robot/open image resize modes,
  visual encoder model name, pretrained flag, configured local DINO checkpoint/cache paths, their
  existence/type checks, and whether a local DINO weight source is configured and structurally
  valid.
- Boolean provenance fields in offline eval/comparison, including robot heatmap routing, action
  padding, block attention, and DINO pretrained status, use the same strict boolean parser as
  policy-training readiness so Hydra strings such as `"false"` or `"off"` are never recorded as
  truthy Python strings.
- Each row also includes `provenance_contract_version` and a stable
  `provenance_contract_id`. The id is a short SHA256 digest over the policy-training contract
  fields, including training scale but not eval runtime knobs such as `eval_batch_size`; use it to
  match a metric row back to the planned train/eval contract row that produced the checkpoint and
  to distinguish otherwise identical runs with different gradient accumulation or effective batch
  scale.
- Use `--variant-override <variant> <hydra_override>` for checkpoint-free sweep evaluation or for
  extra per-variant eval wiring. The comparison script fails if a variant-specific override names an
  unknown variant, so typoed sweep rows cannot be silently ignored.
- By default eval uses each policy config's `policy.cfg_scale`; pass `--cfg-scale` only when a
  global override is desired.
- The offline eval and comparison CLIs keep heavyweight `torch`/policy imports lazy, so `--help`
  and argument review remain available in lightweight environments. Running metrics still requires
  the normal training environment with Torch, Hydra, model dependencies, and dataset access.
- For smoke tests, use debug configs with `--max-batches 1`.

Experiment plan generation:

```text
py scripts/plan_gaze_wam_experiments.py \
  --use-accelerate \
  --real-data-launch \
  --include-sweep gaze_dropout \
  --include-sweep open_ratio \
  --checkpoint-template "data/outputs/{config}/checkpoints/latest.ckpt" \
  --data-onboarding-review-template "data/reviews/{name}_onboarding_review.json" \
  --require-data-onboarding-review \
  --validate-zarr \
  --require-timestamps \
  --timestamp-max-step 0.08 \
  --output-json data/outputs/gaze_wam_experiment_plan.json \
  --output-csv data/outputs/gaze_wam_experiment_plan.csv \
  --output-script data/outputs/run_gaze_wam_experiments.sh
```

- This does not launch training by default; it writes reproducible train/eval command plans.
- For real robot/open-data plans, add `--real-data-launch`. Planned train commands then call
  `scripts/launch_gaze_wam_training.py --run --real-data --real-data-contract ablation` instead of
  launching `train.py` directly, so the generated run script preserves the preflight, zarr-warning,
  timestamp, local-DINO-source, report-writing, and Accelerate gates while allowing planned ablation
  variants. Use `--real-data-contract main` only for a plan that should enforce the fixed
  main-method training contract on every train job. Use `--train-via-launcher` without
  `--real-data-launch` when a debug/smoke plan should exercise the launcher but not the real-data
  hard gates.
- Default full variants are:
  `main`, `robot_only`, `no_gaze`, `cfg`, `no_block_mask`, and `heatmap_clean_token`.
- `--debug` switches to the matching smoke-test configs.
- `--train-override` and `--eval-override` attach Hydra overrides to all planned commands.
- `--launcher-report-template` controls per-variant launch reports for launcher-routed train
  commands. It accepts `{name}` and `{config}` tokens and can be set to an empty string to omit
  launcher report paths in generated commands.
- `--data-onboarding-review-template` controls per-variant onboarding review artifacts for
  launcher-routed train commands. It accepts `{name}` and `{config}` tokens and emits
  `--data-onboarding-review-json <path>` in each generated train command when non-empty. Pair it
  with `--require-data-onboarding-review` for real-data plans whose launch jobs must prove that
  the dry-run data-prep review and the train config point to the same canonical robot/open zarrs.
- `--eval-cfg-scale` adds a global eval-time CFG override; omit it for config-specific CFG
  variants.
- `--include-sweep gaze_dropout` appends the `0.0/0.1/0.2/0.3` robot gaze-dropout sweep as
  per-variant Hydra overrides.
- `--include-sweep open_ratio` appends the `100/0`, `90/10`, `75/25`, and `50/50` source-ratio
  sweep as robot/open dataloader batch-size overrides. Full mode uses a total batch of `64`;
  debug mode uses small integer batches for smoke plans.
- The generated eval command calls `compare_gaze_wam_ablation_metrics.py` with checkpoint paths
  derived from the checkpoint template.
- For every planned variant override, the generated eval command also emits a matching
  `--variant-override <variant> <override>` argument. This keeps checkpoint-free smoke/sweep eval
  faithful to the planned train contract instead of evaluating every sweep row with the same base
  config.
- The generated JSON plan includes a fixed Gaze-WAM contract summary and each train job records
  review-friendly provenance for robot/open batch sizes and ratios, robot gaze-dropout settings,
  CFG scale, block-mask setting, heatmap objective, token/image shape fields, and task/robot/open
  image resize modes.
- Train and eval plan rows also record DINO source provenance: pretrained flag, configured
  checkpoint/cache paths, path existence/type checks, and whether a local DINO weight source is
  configured and structurally valid. Train and eval plan rows include the same
  `provenance_contract_version` and `provenance_contract_id` when they refer to the same intended
  training contract. Eval provenance is expanded per variant in JSON/CSV, while the actual eval
  command remains a single comparison command; when evaluation loads the corresponding checkpoint,
  the metric row's contract id should match the planned eval row.
- The generated CSV expands the same per-job provenance columns and includes an explicit checkpoint
  column for eval rows derived from the checkpoint template.
- The planner exposes the same eval-side validation gates as the comparison script:
  `--validate-zarr/--no-validate-zarr`, `--timestamp-key`, `--require-timestamps`,
  `--timestamp-max-delta`, and `--timestamp-max-step`.

## 13. Failure Modes To Watch

Information leakage:

- Robot heatmap loss is enabled only when robot gaze condition is dropped.
- Robot rows with real gaze condition must not train heatmap loss, because the heatmap label is
  derived from the same gaze point.
- Open-source gaze labels are used as heatmap targets, not gaze conditions.
- Do not let future gaze or future frames condition current action unless explicitly intended.
- Training and validation datasets must split by episode, not by frame/sample, so adjacent frames
  from the same trajectory cannot appear in both sides.
- Point-label gaze rows with missing or out-of-frame coordinates should be resolved during data
  preparation. The current dataset contract only distinguishes valid point-label rows from
  rows whose point-gaze condition is disabled via `has_gaze_label`.
- Canonical converters default to failing on invalid point labels. Explicit clip/drop policy must
  be chosen at preprocessing time and recorded in zarr metadata.

Dummy action tokens:

- Open-source samples have no action labels.
- Their placeholder action tokens must be hidden from heatmap-token queries.
- Condition-token queries must also be blocked from action tokens.

Train-test mismatch:

- Action tokens cannot attend to heatmap tokens in the main method.
- Inference removes heatmap tokens entirely.

Relative action bugs:

- Zarr stores absolute TCP, but the model trains on relative action.
- Normalizer must fit relative actions.
- `action_base_abs` must be the latest observed TCP pose after observation slicing and latency
  alignment, matching `W:/umi_base`.
- Inverse conversion must use the same rotation representation as training.

Historical observation bugs:

- `N_v` is not always `256`; it is `T_obs * 256`.
- Action slice indices must use the computed `N_v`.
- Temporal embeddings must distinguish frames, otherwise two-frame observations become an
  unordered patch set.
- Dataset temporal parameters must fail fast when invalid:
  `n_obs_steps > 0`, `action_horizon > 0`, `obs_downsample_steps > 0`,
  `action_downsample_steps > 0`, and `n_latency_steps >= 0`.
- Keep dataset-sample validation enabled in validator/preflight gates so bad temporal parameters
  are caught before long training jobs.

Grid mismatch:

- Main config is fixed to `256 x 256` images with DINOv3/16.
- This gives `16 x 16 = 256` image tokens and aligns with heatmap tokens.
- Do not silently switch back to `224 x 224` without changing the token contract.

Open-data interference:

- Open-source samples should not contribute action loss.
- Track robot-only validation after introducing open data.
- If open data hurts control, first sweep open ratio and loss weight before considering module
  freezing.

Compute blow-up:

- Pixel-level heatmap tokens would produce `256 * 256 = 65536` scalar targets per frame.
- Use `16 x 16 = 256` heatmap tokens as the main target.

AMP/DDP instability:

- bf16 should be tested before fp16.
- Masked losses must behave correctly when a rank receives few open-source or robot samples.
- Use distributed masked mean with cross-rank numerator/denominator reduction.

## 14. Recommended Implementation Order

Status as of the current implementation slice:

Done:

- Fixed heatmap codec utilities:
   - point to `16 x 16` tokens
   - optional image to `16 x 16` tokens
   - tokens to `256 x 256` visualization heatmap
- Shared gaze bounds validation utility and strict-by-default preprocessing policies for robot and
  open point-label data.
- `GaussianSpatialEncoder` and the trainable gaze `[MASK]` token.
- Relative action conversion tests using UMI-style pose utilities.
- Generic robot/open zarr datasets that emit the final batch contract.
- Deterministic episode-level train/validation split for robot and open Gaze-WAM zarr datasets.
- Robot zarr canonicalizer for mapping non-canonical real-log keys into the standard Gaze-WAM
  robot schema before validation/training.
- Raw/canonical zarr inspector for reporting array shapes/ranges and suggesting Gaze-WAM key
  mappings before canonicalization.
- Joint transformer backbone with modality embeddings and block attention masks.
- Optional explicit frame embedding ablation for historical image tokens, defaulting off in the
  main config.
- `GazeWamPolicy.compute_loss` with:
   - robot action loss for all robot rows
   - robot heatmap loss only on gaze-condition-dropout rows
   - open-source heatmap loss only
- Smoke tests:
   - mixed batch loss runs
   - open rows have zero action-loss contribution
   - robot real-gaze rows have zero heatmap-loss contribution
   - robot gaze-dropout rows have nonzero action and heatmap loss contribution
   - heatmap tokens cannot attend to dummy action tokens
   - inference sampling returns `[B, 16, D_action]`
   - action slicing works for `T_obs=1` and `T_obs=2`
- Hydra configs.
- Accelerate bf16 multi-GPU launch config and Makefile target.
- Synthetic mixed-batch debug training.
- Gaze-Dependency Ratio logging.
- Policy-level heatmap prediction and `256 x 256` visualization helper for diagnostics and paper
  figures.
- Robot-only, no-gaze, CFG, and no-block-mask full-run ablation configs plus matching debug smoke
  configs.
- Heatmap clean-token objective ablation configs, defaulting the main method to diffusion-style
  heatmap targets.
- Accelerate-native gradient accumulation in the Gaze-WAM workspace.
- Workspace validation loop over episode-disjoint robot/open validation dataloaders, including
  validation loss component logging and `val_loss` top-k checkpoint monitoring.
- Workspace-level `training_contract.json` audit artifact for source ratios, dropout routing,
  token geometry, loss settings, zarr `meta.attrs` data-source provenance for dataset type, resize
  mode, and image size, task-vs-dataset image-size and temporal-sampling consistency, optional
  metadata/presence-mask semantics, DINO local weight-source provenance, and canonical-main-config
  checks.
- Validation heatmap preview artifact writer for local paper/debug figures from predicted and
  target heatmap tokens.
- Offline metric evaluator for robot/open datasets with action MSE, absolute-action MSE when
  metadata is present, heatmap token MSE/KL/argmax error, and feature/output GDR.
- Ablation metric comparison script that evaluates multiple configs/checkpoints and writes JSON
  and CSV rows for paper tables, including config provenance columns for source ratios, CFG scale,
  dropout, temporal sampling, block-mask, heatmap objective, token/image shapes, visual encoder
  identity, DINO checkpoint/cache source validity, and strict parsed boolean settings.
- Experiment-plan generator that writes reproducible train/eval command plans for the main
  Gaze-WAM ablation matrix as JSON, CSV, and optional shell script, with train/eval DINO source
  provenance aligned to the ablation metric table.
- Generic open-source gaze manifest converter that writes the Gaze-WAM open zarr schema from
  point labels or dense one-channel heatmap labels.
- Generic video-gaze manifest exporter that extracts frames from video metadata and can directly
  call the open zarr converter.
- Configurable open-video gaze metadata adapter for dataset-specific or nested metadata fields,
  including optional direct chaining into frame extraction and open-zarr export.
- Open-video gaze metadata inspector that reports nested field candidates and suggested adapter
  key maps before running a first real open dataset export.
- Open-data prepare dry-run review mode that records planned metadata adaptation, frame export,
  manifest conversion, validation, and preview commands before any heavy frame/zarr IO.
- Unified robot/open data-onboarding review CLI that runs both prepare dry-runs under one JSON
  artifact for policy-training review and explicitly leaves runner/deployment scope deferred.
- Robot/open zarr schema validator and root CLI wrapper for checking required keys, episode
  boundaries, tensor shapes, strict normalized gaze bounds, sample adapter compatibility, and robot
  relative-action roundtrip consistency.
- Optional timestamp preservation in open manifest conversion/video export and robot canonicalization,
  plus validator checks for finite nondecreasing timestamps and bounded cross-stream alignment.
- Dataset preview CLI for writing RGB, gaze overlay, decoded token heatmap, combined overlay, and
  JSON summaries from robot/open zarr samples.
- Preflight CLI for checking Hydra config, dataset instantiation, zarr validation, mixed-batch
  construction, policy instantiation, normalizer fitting, and one loss-smoke pass before long runs.
- End-to-end smoke pipeline CLI that chains debug zarr generation, zarr validation, and preflight
  into one JSON-producing gate for policy-training readiness. Optional deployment rehearsal outputs,
  when present, are reference-only and are not part of the current acceptance criteria.
- Reference-only deployment scaffolding exists for action-only adapter IO, offline rehearsal,
  provider configs, and hardware-neutral runner experiments. For this policy-training milestone,
  treat those files as deferred scaffolding and do not expand or tune them.

Remaining implementation work:

1. Integrate the physical robot dataset.
2. Run the first real open gaze dataset through the configurable metadata adapter and generic
   video exporter. Ego-Exo4D is still the intended target, but the exact camera/split/field
   selection must be chosen from the real metadata.
3. Verify the real DINOv3 checkpoint/cache path and run real mixed-batch policy training with the
   `training_contract.json`, validation losses, heatmap previews, and metric logs enabled.
4. Run the policy ablation suite and paper metrics on real tasks.

Runner/hardware binding is not a remaining item for this policy-training milestone. Existing
deployment adapters/rehearsals can stay as reference utilities, but no runner changes are required
for the current closeout.

## 15. Remaining Decisions

Most algorithmic contracts are fixed. The remaining decisions are data/environment choices and
evaluation conveniences rather than core model-shape choices:

Consistency-check result:

- All main-method model-shape, heatmap representation, loss-routing, prediction-head, and
  attention-mask direction decisions are closed.
- The items below should be resolved before real runs, but changing them should not change the
  fixed tensor contract unless explicitly called out in a future design revision.

The following are no longer open design questions:

- Image and heatmap resolution contract: `256 x 256` input images with DINOv3 ViT/16 patch tokens,
  plus `16 x 16 = 256` frozen-Cosmos heatmap latent tokens.
- Heatmap representation: frozen NVIDIA Cosmos `Cosmos-Tokenizer-CI16x16` encoder/decoder,
  `heatmap_dim=16`, project-estimated latent scale/offset, no main-method latent MSE, diffusion
  clean-latent reconstruction decoded to full resolution and supervised by DSNT coordinate loss
  plus dense spatial JS loss, and decoded `256 x 256` heatmaps for visualization/evaluation.
- Prediction outputs: separate `LayerNorm + Linear` action and heatmap heads, matching the original
  transformer action-output style.
- Mixed supervision routing: no dynamic head freezing; action/heatmap losses are routed by
  per-sample masks.
- Main source policy: robot real-gaze rows train action only, robot gaze-dropout rows train
  action + heatmap, and open-source gaze rows train heatmap only under the trainable `[MASK]`
  condition.
- Attention policy: condition/world tokens do not read target tokens, action and heatmap target
  tokens do not read each other, action does not read noisy heatmap target tokens, and fast action
  inference drops heatmap target tokens while reusing the heatmap/world K/V cache.
- Observation/action contract: default `T_obs=2`, `T_action=16`, and `D_action=10` with
  `3 pos + 6 rot + 1 gripper`; robot zarr stores absolute TCP trajectories and the dataset converts
  them to relative action targets.
- Default experimental knobs are fixed in config for the first run: `robot_gaze_dropout_prob=0.2`,
  `action_loss_weight=1.0`, `heatmap_loss_weight=1.0`,
  `heatmap_token_kl_loss_weight=0.0`, `heatmap_xy_loss_weight=1.0`,
  `heatmap_js_loss_weight=1.0`, `heatmap_objective=dsnt_js`, and `num_inference_steps=8`.
  Sweeps over these values are ablations, not unresolved model contracts.

1. Confirm the physical robot source-to-canonical mapping.
   - The canonical Gaze-WAM keys are fixed as `camera0_rgb`, `action_abs_tcp`, `tcp_pose_abs`,
     `gripper_width`, normalized `gaze_xy`, optional heatmap metadata, and `episode_ends`.
   - Real log names can differ, but they must be mapped into this canonical schema before training.
   - Use `scripts/validate_gaze_wam_zarr.py` after mapping the real log keys into the canonical
     schema.
2. Verify the local DINOv3 checkpoint artifact.
   - The architecture and default timm identity are fixed to DINOv3 ViT/16,
     `vit_base_patch16_dinov3`.
   - Use `scripts/verify_gaze_wam_dino_source.py` for a lightweight path, geometry, and
     normalization check before launching full preflight.
   - The remaining environment task is confirming that the local checkpoint/cache path, documented
     preprocessing stats, and 256-resolution positional-embedding behavior load correctly on the
     target machine.
3. Choose the first open gaze dataset.
   - Ego-Exo4D is the intended target.
   - A smaller gaze dataset can be used first for smoke testing.
4. Confirm canonical training camera keys.
   - The document uses `camera0_rgb` as the canonical key.
   - If robot logs expose wrist/head/eye camera names, the dataset adapter should map the chosen
     training stream into the canonical model input name.
5. Confirm the paper-figure heatmap visualization style.
   - The fixed codec and policy helper already support token-to-`256 x 256` visualization.
   - The model-side default is Gaussian splatting from predicted clean heatmap tokens, or from the
     reconstructed clean `x0` tokens when the scheduler uses epsilon prediction.
   - The remaining choice is only presentation style: colormap, overlay opacity, panel layout, and
     whether paper figures report token metrics, pixel metrics, or both.
6. Confirm training-data timing and synchronization policy.
   - The current first-run config uses `n_latency_steps=0` and assumes aligned image/state/gaze
     timestamps.
   - If the recorded setup has nonzero sensor or controller latency, encode it in preprocessing,
     dataset timestamp gates, or `n_latency_steps` before real policy training.

## 16. Current Implementation Status

Completed:

- Cloned `Fanqi-Lin/Data-Scaling-Laws` into `gaze-wam`.
- Inspected the base training entrypoint, transformer policy, transformer backbone, visual encoder,
  UMI dataset, and workspace loop.
- Compared the architecture figure against this plan.
- Inspected `W:/umi_base` references for:
  - `q3_place_cup_no_tcp.yaml`
  - relative action conversion utilities
  - `train_acc8_amp`
  - `accelerate/8gpu-amp.yaml`
- Updated this design document to the current agreed decisions.
- Implemented the policy-training foundation: Gaze-WAM utilities, policy, joint transformer,
  dataset adapters, mixed-batch builder, workspace, configs, synthetic debug data, tests, CFG
  inference, GDR logging, offline metric evaluation, generic open-gaze manifest conversion, robot
  zarr canonicalization, zarr schema validation, optional frame-embedding ablation, preflight, and
  training-contract audit artifacts. See `docs/implementation_log.md` for the detailed changelog.
- Reference deployment scaffolding also exists in the tree, but it is not part of the current
  policy-training acceptance criteria.

Not done yet:

- Real robot zarr integration.
- Ego-Exo4D-specific field mapping or first-real-open-dataset export using the generic manifest
  tools.
- Ablation suite and paper metrics on real tasks.
- Real DINOv3 checkpoint/cache verification and real multi-GPU mixed-batch training run.
- Runner/hardware-specific work is intentionally deferred and should not be treated as a blocker
  for this policy-training-only closeout.

Policy-only handoff:

- The next meaningful milestone is not runner editing; it is a real-data policy-training readiness
  pass: canonicalize robot/open zarrs, verify the DINOv3 source, run preflight, then launch mixed
  75/25 policy training with contract, validation, mask-count, and heatmap-preview artifacts.
- Deployment adapter and runner files can stay untouched until trained policy checkpoints and
  offline policy metrics justify moving to live-robot execution.
