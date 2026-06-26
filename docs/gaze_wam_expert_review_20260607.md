# Gaze-WAM Expert Review

Date: 2026-06-07

## Post-Cleanup Review Addendum

Supersession note:

- The learned-decoder path described in the original 2026-06-07 review is now historical. The active
  heatmap codec is the frozen NVIDIA Cosmos `Cosmos-Tokenizer-CI16x16` encoder/decoder with
  project-estimated latent scale/offset. The main heatmap loss is full-resolution DSNT+JS after the
  frozen decoder; no latent MSE is mixed into `heatmap_objective=dsnt_js`.

- The active package surface has been narrowed to the Gaze-WAM policy-training path. Original
  Data-Scaling-Laws envs, env runners, legacy policies, legacy datasets, old hardware stack, and
  obsolete scripts are archived outside the active package.
- The Cosmos heatmap path now matches the intended FastWAM-style contract more closely:
  DiT predicts compact frozen-tokenizer heatmap latent tokens, the frozen Cosmos decoder maps them
  to full-resolution `256 x 256` logits, and DSNT/JS losses supervise the decoded image from online
  `gaze_xy` targets.
- `data/gaze_heatmap` is not required for the main path. Dataset rows provide zero heatmap
  placeholders when `heatmap_key=null`; the clean target is generated online from normalized
  `data/gaze_xy`.
- A review issue was found and fixed: checkpoint heatmap previews previously used a single-step
  zero-latent diagnostic path. `predict_heatmap()` now defaults to iterative latent denoising, and
  preview summaries record `heatmap_prediction_mode=iterative_denoise`.
- Server verification after cleanup passed syntax checks, active-script checks, old-import scans,
  CUDA cached-dual-stream smoke, and an epoch-0 iterative checkpoint preview.

Residual risk:

- Cosmos CI16x16 is a pretrained frozen RGB image tokenizer, not a heatmap/mask-specialized codec.
  Future ablations should compare CI8x8, other SOTA image codecs, or a heatmap/mask-specific frozen
  codec if residual texture artifacts remain.

## Verdict

The dim4/dim16 runs and the later `heatmap_dim=256` lossless patch run should remain stopped. After
checking FastWAM, the heatmap branch should be treated as a latent generation branch with an
explicit decoder, not as direct pixel-patch unpatchification.

## Current Contract

- Active config path:
  - `diffusion_policy/config/train_gaze_wam_workspace.yaml`
  - `diffusion_policy/config/train_gaze_wam_open_only_workspace.yaml`
  - `diffusion_policy/config/task/gaze_wam.yaml`
- Main heatmap settings:
  - `task.heatmap_key: null`
  - `task.heatmap_dim: 16`
  - `task.heatmap_num_tokens: 256`
  - `task.heatmap_token_grid: [16, 16]`
  - `policy.heatmap_spatial_decoder: cosmos_tokenizer`
  - `policy.heatmap_latent_scale: 0.25`
  - `policy.heatmap_latent_offset: 0.0`
  - `policy.heatmap_objective: dsnt_js`
  - `policy.model_architecture: cached_dual_stream`
- Heatmap flow:
  `gaze_xy -> dense 256x256 Gaussian target -> frozen Cosmos CI16x16 encoder -> scaled 16x16x16 latent tokens -> DiT denoising -> inverse scale -> frozen Cosmos decoder -> dense 256x256 DSNT/JS loss and preview`.
- Open-source rows train heatmap only under the learned `[MASK]` gaze token.
- Robot real-gaze rows train action only; robot gaze-dropout rows can train action plus heatmap.
- Fast action inference drops heatmap target tokens and consumes the stable image/gaze world cache.

## Evidence

- Server status:
  - patch-fullres training and watcher sessions stopped.
  - patch-fullres output directories moved to `_obsolete_fastwam_misaligned_runs_20260607`.
  - no active Gaze-WAM training process was observed after stopping.
- Local code changes:
  - Active policy construction only accepts `policy.heatmap_spatial_decoder=cosmos_tokenizer`.
  - Policy DSNT/JS decode path denormalizes predicted heatmap latents and runs the frozen Cosmos
    decoder without cutting input gradients.
  - Configs and launch guards require `heatmap_dim=16` plus explicit Cosmos JIT paths.

## Findings

No blocking conceptual issue remains in the corrected frozen-Cosmos heatmap contract.

Watch items:

1. The frozen Cosmos codec is RGB-pretrained, not heatmap-specialized.
2. A future SAM/object-mask target can replace the online Gaussian target generator without changing
   the DiT/decoder interface, as long as the dense target is encoded through the same frozen Cosmos
   encoder and latent scale.
3. Current launch gating should use focused smoke checks plus preflight until the large historical
   test file is pruned to the active policy-training surface.

## Recommendation

Run server-side syntax/smoke checks, sync the corrected codebase to server1024, then launch a fresh
8-GPU AMP open-only run with a new `hot3d_open_cosmos_scaled_latent_8gpu_amp_*` output directory.
Start checkpoint preview watching immediately so saved checkpoints produce seeded `gaze_xy`,
predicted-heatmap, and target-heatmap comparisons.
