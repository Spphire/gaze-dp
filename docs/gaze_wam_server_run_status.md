# Gaze-WAM Server Run Status

Last checked: 2026-06-17 17:29 Asia/Shanghai

## Active Server Run

- SSH: `root@106.14.2.243 -p 1024`
- Project: `/mnt/workspace/shenyibo/gaze-wam`
- Active experiment: temporal-window `mix-LNLL` B-side A/B run.
- Training tmux: `gaze_wam_open_cosmos_temporal_mixed_nll_8gpu`
- Checkpoint-preview watcher tmux: `gaze_wam_temporal_mixed_nll_ckpt_preview_watch`
- Episode-preview watcher tmux: `gaze_wam_temporal_mixed_nll_episode_preview_watch`
- Output directory:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_`
- Watcher preview root:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched`
- Episode preview root:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched`

## A-Side Reference For Current A/B

- DSNT-mix run was stopped intentionally at 2026-06-15 15:33 +08 to free all 8 GPUs for the
  `mix-LNLL` B-side run. Its outputs/checkpoints/media are preserved for direct A/B.
- Output directory:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_`
- Latest preserved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0019-val_loss=0.016.ckpt`
- Latest preserved checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_152145_epoch=0019-val_loss=0.016`
- Latest preserved validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0019`
- Latest preserved episode 86 video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_152255_epoch=0019-val_loss=0.016_episode86/comparison.mp4`

## Previous Completed Baseline

- Pure single-point Gaussian clean-label latent MSE run:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_`
- Latest checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0199-val_loss=0.002.ckpt`
- Completed at epoch 199 / global step 267999 with final validation loss about
  `0.0015848265`.
- Visual conclusion: numerically stable but visually plateaued with fixed offsets and persistent
  background/texture/checkerboard-like artifacts.

## Current Contract

- Dataset route: 100% open HOT3D gaze zarr, no robot/action loss.
- Training route: 8-GPU Accelerate with required AMP, `mixed_precision=bf16`.
- Heatmap codec: frozen NVIDIA Cosmos Tokenizer `Cosmos-Tokenizer-CI16x16`.
- Cosmos files on server:
  - `data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit`
  - `data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit`
- Heatmap flow:
  `gaze_xy temporal neighborhood -> online 256x256 temporal-window dense target -> frozen Cosmos encoder -> scaled 16x16x16 clean latent -> DDIM epsilon/delta-noise MSE`.
  Validation/checkpoint previews decode the sampled latent through frozen Cosmos decoder and show the
  same temporal-window target used for training.
- Temporal label:
  `mode=bidirectional`, `window_radius=30`, `beta=10.0`, `sigma_px=6.0`,
  `current_weight=2.0`.
- Active heatmap supervision is now mixed: the frozen-Cosmos latent still trains with
  diffusion/noise MSE, and the reconstructed final heatmap is decoded for small distribution
  losses.
- Latent scale contract:
  - The standalone `Cosmos-0.1-Tokenizer-CI16x16` JIT package does not provide a usable
    checkpoint-specific latent normalizer for this heatmap chain.
  - Project stats file:
    `data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json`.
  - Stats basis: `gaze_xy -> online 256x256 Gaussian target -> frozen Cosmos CI16x16 encoder`,
    random 4096 HOT3D open samples, seed 42.
  - Raw latent range summary: `min=-3.921875`, `max=3.375`, `std=1.2139403820037842`,
    `abs_max=3.921875`, `abs_p99.5=3.5`.
  - Default training scale: `heatmap_latent_scale=0.25`,
    `heatmap_latent_offset=0.0`, `heatmap_scheduler_clip_sample=true`.
  - Scaled observed label range in the active contract:
    `[-0.98046875, 0.84375]`, so scheduler `clip_sample=[-1, 1]` should not clip the clean labels.
- Heatmap objective for the active B-side ablation: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`.
- Active B-side loss formula:
  `L = L_diffusion_noise_mse + alpha_bar_t * (0.001 * L_point_NLL + 0.10 * L_JS)`,
  with `L_XY` disabled. This is intended to avoid forcing the multi-modal temporal label into a
  single DSNT coordinate while still anchoring decoded probability mass at the observed gaze point.
- Decoder-output interpretation: main line treats frozen Cosmos decoder output as decoded heatmap
  intensity, not logits. Main config uses `heatmap_distribution_mode=intensity_softplus` and
  `heatmap_dsnt_temperature=0.1`; `logits_softmax` and `intensity_clamp` remain ablation modes.
- Gaze input for open data: learned `[MASK]`/empty gaze token through `GazeConditionEncoder`; `use_gaze_condition=false`.
- Action branch for open data: action loss mask is false, action output is ignored.

## Latest Observed Status

- 2026-06-17 17:29 +08 check: B-side `latest.ckpt` advanced through validation
  `epoch_0048`, and the newest checkpoint/validation/episode watcher outputs are complete.
  Training remains healthy in epoch 49; latest checked `logs.json.txt` reached
  `global_step=65924`, and the training tmux pane showed about `1060/5359` batches through epoch
  49. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_172059_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0048`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_171849_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 48 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0048_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0048_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `48`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch48 remains compact and checkerboard-safe, but it is not an improvement
  over epoch47 or the cleaner epoch43 validation sheet. Checkpoint samples 0-1 are comparable,
  while sample 2 collapses into a compact below-target carton/object texture lock and sample 3
  stays laterally biased. Validation sample 0 is closer to the gaze marker than A, but samples 1-3
  keep vertical/table-cup side modes and below/side mass. Episode 86 is still near-tied; frame 0
  remains off-target for both A and B, while later frames are compact. Net: point-NLL is stable but
  still has not reliably reduced object/table-lock.
- 2026-06-17 16:35 +08 check: B-side `latest.ckpt` advanced through validation
  `epoch_0047`, and the newest checkpoint/validation/episode watcher outputs are complete.
  Training remains healthy in epoch 48; latest checked `logs.json.txt` reached
  `global_step=64726`, and the training tmux pane showed about `1630/5359` batches through epoch
  48. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_161949_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0047`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_161730_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 47 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0047_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0047_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `47`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch47 remains numerically healthy, compact, and checkerboard-safe, but it
  is still not a clear point-NLL win. Checkpoint samples 0-1 are comparable to A, sample 2 is less
  vertically fragmented than A but keeps a below-target object/texture tail, and sample 3 remains
  laterally biased. Validation still places mass below or beside the gaze marker with table/cup
  side modes; it is not a recovery of the cleaner epoch43 sheet. Episode 86 remains near-tied and
  clean, with frame 0 still off-target for both A and B. Net: the late trend stays stable but does
  not show reliable object/table-lock reduction.
- 2026-06-17 15:37 +08 check: B-side `latest.ckpt` advanced through validation
  `epoch_0046`, and the newest checkpoint/validation/episode watcher outputs are complete.
  Training remains healthy in epoch 47; latest checked `logs.json.txt` reached
  `global_step=63496`, and the training tmux pane showed about `2070/5359` batches through epoch
  47. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_151837_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0046`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_151916_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 46 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0046_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0046_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `46`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch46 is less fragmented than epoch45 on checkpoint samples, but it is
  still not a recovery of the cleaner epoch43 validation sheet. Checkpoint sample 2 flips to a
  below-target object/texture lock and sample 3 remains sideways. Validation stays compact in
  spots but is consistently below/off-target with table/cup side modes. Episode 86 remains clean
  and near-tied. Net: the post-epoch43 trend is non-monotonic and does not yet support point-NLL
  as a reliable object/table-lock fix.
- 2026-06-17 14:56 +08 check: B-side `latest.ckpt` advanced through validation
  `epoch_0045`, and the newest checkpoint/validation/episode watcher outputs are complete.
  Training remains healthy in epoch 46; latest checked `logs.json.txt` reached
  `global_step=62690`, and the training tmux pane showed about `4204/5359` batches through epoch
  46. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_141727_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0045`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_141802_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 45 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0045_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0045_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `45`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch45 is stable and checkerboard-safe, but it further weakens the
  apparent epoch43 validation improvement. Checkpoint sample 0 adds a new off-target shelf blob,
  and the hard carton/object sample again shows a stronger vertical chain. Validation fragments
  badly on sample 0 with multiple table/monitor blobs and keeps off-target modes in later rows.
  Episode 86 is still clean and near-tied, but it does not provide a placement win. Net: point-NLL
  remains viable numerically, but the later trend is non-monotonic and not a reliable
  object/table-lock fix yet.
- 2026-06-17 13:24 +08 check: B-side `latest.ckpt` advanced through validation
  `epoch_0044`, and the newest checkpoint/validation/episode watcher outputs are complete.
  Training remains healthy in epoch 45; latest checked `logs.json.txt` reached
  `global_step=60581`, and the training tmux pane showed about `1130/5359` batches through epoch
  45. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_131615_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0044`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_131639_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 44 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0044_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0044_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `44`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch44 remains compact and checkerboard-safe but is non-monotonic after
  the cleaner epoch43 validation sheet. Checkpoint sample 0 stays tight and the hard
  carton/object sample is better than early late-epoch reads, but sample 3 still has a sideways
  lobe. Validation partially regresses: samples 2/3 bring back vertical/tabletop side modes and
  separate off-target blobs, so epoch44 is not a stronger object/table-lock result than epoch43.
  Episode 86 remains near-tied against A epoch19, with no clear placement win.

- 2026-06-17 12:25 +08 check: B-side `latest.ckpt` advanced through validation
  `epoch_0043`, and the newest checkpoint/validation/episode watcher outputs are complete.
  Training remains healthy in epoch 44; latest checked `logs.json.txt` reached
  `global_step=59185`, and the training tmux pane showed about `618/5359` batches through epoch
  44. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_121504_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0043`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_121515_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 43 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0043_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0043_mismatched_budget`
- Also packaged the complete intermediate epoch 42 fallback bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0042_mismatched_budget`
  and
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0042_mismatched_budget`.
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `43`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch43 is the best mismatched-budget trend read so far. It remains
  compact and checkerboard-safe, checkpoint sample 0 collapses closer to the table target, the
  hard carton/object sample has a weaker vertical lobe, and validation now places compact mass
  near the fixation dot on all four repeated monitor/table samples. The remaining caveat is that
  episode 86 is still near-tied against A epoch19: B is compact and clean, but not visibly closer
  on the inspected stills/video, so this is not yet a strict object/table-lock win. Epoch 42 was
  weaker than epoch43, with validation still showing vertical table/monitor streaks.

- 2026-06-17 11:00 +08 check: B-side `latest.ckpt` advanced to validation `epoch_0041`, and the
  newest checkpoint/validation/episode watcher outputs are complete. Training remains healthy in
  epoch 42; latest checked `logs.json.txt` reached `global_step=57358`, and the tmux pane showed
  about `2300/5359` batches through epoch 42. Training tmux, checkpoint watcher, and episode
  watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_101541_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0041`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_101530_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 41 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0041_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0041_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `41`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch41 remains compact and checkerboard-safe, but it still does not beat
  the preserved DSNT-mix reference. The hard checkpoint carton/object sample keeps an off-target
  vertical streak, checkpoint sample 3 spreads sideways, validation regresses into tall vertical
  streaks around the table/cup/monitor structure, and episode 86 stays near-tied without clear
  object/table-lock reduction.
- 2026-06-17 09:34 +08 check: B-side `latest.ckpt` advanced to validation `epoch_0040`, and the
  newest checkpoint/validation/episode watcher outputs are complete. Training remains healthy in
  epoch 41; latest checked `logs.json.txt` reached `global_step=55440`, and the tmux pane showed
  about `1996/5359` batches through epoch 41. Training tmux, checkpoint watcher, and episode
  watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_091433_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0040`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_091411_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 40 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0040_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0040_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `40`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch40 remains checkerboard-safe and compact, but it still does not beat
  the preserved DSNT-mix reference. Checkpoint sample 0 is cleaner than epoch39, but sample 2
  keeps separated vertical/carton-object mass; validation remains below/left of the target with
  table/cup/object bias and a split sample 3; episode 86 stays near-tied without clear
  object/table-lock reduction.
- 2026-06-17 08:21 +08 check: B-side `latest.ckpt` advanced to validation `epoch_0039`, and the
  newest checkpoint/validation/episode watcher outputs are complete. Training remains healthy in
  epoch 40; latest checked `logs.json.txt` reached `global_step=54246`, and the tmux pane showed
  about `2586/5359` batches through epoch 40. Training tmux, checkpoint watcher, and episode
  watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_081334_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0039`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_081248_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 39 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0039_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0039_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `39`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch39 remains checkerboard-safe and compact, but it still does not beat
  the preserved DSNT-mix reference. Checkpoint sample 0 adds an off-target up-left mode, sample 2
  remains vertically fragmented, validation is more fragmented with table/cup/object side modes,
  and episode 86 stays near-tied without clear object/table-lock reduction.
- 2026-06-17 07:47 +08 check: B-side `latest.ckpt` advanced to validation
  `epoch_0038`, and newest checkpoint/episode watcher outputs are complete. Training remains
  healthy in epoch 39; latest checked `logs.json.txt` reached `global_step=53063`, and the tmux
  pane showed about `3216/5359` batches through epoch 39. Training tmux, checkpoint watcher, and
  episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_071226_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0038`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_071142_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 38 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0038_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0038_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `38`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch38 remains checkerboard-safe and compact, but it still does not improve
  the preserved DSNT-mix reference. Checkpoint sample 0 is a little cleaner than epoch 37, but the
  hard carton/object sample remains ambiguous; validation is worse evidence, with separated
  off-target lobes and table/cup/object side modes; episode 86 remains near-tied without clear
  object/table-lock reduction.
- 2026-06-17 02:08 +08 check: B-side `latest.ckpt` advanced to validation
  `epoch_0032`, and newest checkpoint/episode watcher outputs are complete. The episode preview
  was briefly unreadable while ffmpeg was still writing the MP4, then completed normally with a
  valid `summary.json` and ffprobe. Training remains healthy in epoch 33; latest checked
  `logs.json.txt` reached `global_step=45370`, and the tmux pane showed about `4604/5359`
  batches through epoch 33. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_010819_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0032`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_010940_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 32 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0032_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0032_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `32`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch32 remains stable and checkerboard-safe, but still does not reverse
  the strict same-budget epoch 19 conclusion. Checkpoint sample 0 has a small side speckle and the
  hard carton/object case still keeps side/vertical mass; validation remains fragmented with
  side modes around table/cup/object regions; episode 86 remains near-tied without clear
  object/table-lock reduction.
- 2026-06-17 00:26 +08 check: B-side `latest.ckpt` advanced to validation
  `epoch_0031`, and newest checkpoint/episode watcher outputs are complete. Training remains
  healthy in epoch 32; latest checked `logs.json.txt` reached `global_step=43298`, and the tmux
  pane showed about `1678/5359` batches through epoch 32. Training tmux, checkpoint watcher, and
  episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260617_000710_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0031`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260617_000819_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 31 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0031_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0031_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `31`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch31 remains stable and checkerboard-safe, but still does not reverse
  the strict same-budget epoch 19 conclusion. Checkpoint samples stay usable but the hard
  carton/object case keeps secondary vertical mass; validation shows even clearer extra side modes
  in spots and remains more fragmented than A epoch 19; episode 86 remains near-tied without clear
  object/table-lock reduction.
- 2026-06-16 23:29 +08 check: B-side `latest.ckpt` advanced to validation
  `epoch_0030`, and newest checkpoint/episode watcher outputs are complete. The episode preview
  was briefly unreadable while ffmpeg was still writing the MP4, then completed normally with a
  valid `summary.json` and ffprobe. Training remains healthy in epoch 31; latest checked
  `logs.json.txt` reached `global_step=41993`, and the tmux pane showed about `1816/5359`
  batches through epoch 31. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_230858_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0030`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_230657_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 30 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0030_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0030_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `30`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch30 remains stable and checkerboard-safe, but still does not reverse
  the strict same-budget epoch 19 conclusion. Checkpoint samples are compact enough but B keeps
  side/vertical mass in the hard carton/object sample; validation remains more fragmented and
  object/table-biased than A epoch 19; episode 86 remains near-tied without clear tabletop-lock
  reduction.
- 2026-06-16 22:35 +08 check: B-side `latest.ckpt` advanced to validation
  `epoch_0029`, and newest checkpoint/episode watcher outputs are complete. Training remains
  healthy in epoch 30; latest checked `logs.json.txt` reached `global_step=41132`, and the tmux
  pane showed about `3732/5359` batches through epoch 30. Training tmux, checkpoint watcher, and
  episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_220746_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0029`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_220536_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 29 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0029_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0029_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `29`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and
  validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video
  is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch29 remains stable and checkerboard-safe, but it still does not reverse
  the strict same-budget epoch 19 conclusion. Checkpoint samples are mostly compact, though B
  sample 0 keeps taller side mass around the target; validation remains more fragmented and
  multi-peaked than A epoch 19 on the table/cup/object cases; episode 86 stays near-tied without
  clear object/table-lock reduction.
- 2026-06-16 21:43 +08 check: B-side `latest.ckpt` advanced to validation `epoch_0028`, and newest checkpoint/episode watcher outputs are complete. Training remains healthy in epoch 29; latest checked `logs.json.txt` reached `global_step=39489`, and the tmux pane showed about `2520/5359` batches through epoch 29. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_210635_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0028`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_210713_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 28 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0028_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0028_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `28`; B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch28 still does not reverse the strict same-budget epoch 19 conclusion. B remains stable and checkerboard-safe, but checkpoint sample 0 still carries side mass above/beside the gaze target, validation remains more fragmented/multi-peaked than A epoch 19, and episode 86 remains near-tied without clear object/table-lock improvement.
- 2026-06-16 20:24 +08 check: B-side `latest.ckpt` advanced to validation `epoch_0027`, and newest checkpoint/episode watcher outputs are complete. Training remains healthy in epoch 28; latest checked `logs.json.txt` reached `global_step=37889`, and the tmux pane showed about `1480/5359` batches through epoch 28. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_200524_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0027`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_200553_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 27 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0027_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0027_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `27`; B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch27 still does not reverse the strict same-budget epoch 19 conclusion. B remains stable and checkerboard-safe, but checkpoint sample 0 now shows a stronger side peak above the gaze target, validation is still more fragmented/multi-peaked than A epoch 19, and episode 86 remains near-tied without clear object/table-lock improvement.
- 2026-06-16 19:44 +08 check: B-side `latest.ckpt` advanced again and the newest watcher outputs are now complete through `latest/epoch26`. Training remains healthy in epoch 27; latest checked `logs.json.txt` reached `global_step=37126`, and the tmux pane showed about `3790/5359` batches through epoch 27. Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_190415_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0026`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_190429_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 26 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0026_mismatched_budget`
- Local fallback comparison bundle:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0026_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `26`; B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`, and validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch26 still does not reverse the strict same-budget epoch 19 conclusion. B remains stable and checkerboard-safe, but validation is more fragmented and multi-peaked than A epoch 19, the hard carton/object case still keeps side mass, and episode 86 remains near-tied without a clear object/table-lock improvement.
- 2026-06-16 18:22 +08 check: B-side `latest.ckpt` has advanced to validation
  `epoch_0025`, and the matching checkpoint watcher preview plus episode 86 preview are now
  complete. This is not a named checkpoint, so it is recorded as `latest/epoch25` evidence only.
  Training remains healthy in epoch 26; latest checked `logs.json.txt` reached
  `global_step=35156`, and the tmux pane showed about `1270/5359` batches through epoch 26.
  Training tmux, checkpoint watcher, and episode watcher are all alive.
- B-side latest checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/latest.ckpt`
- B-side latest checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_180251_latest`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0025`
- B-side latest episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_180250_latest_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 25 exists.
  This is explicitly a mismatched-budget latest comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0025_mismatched_budget`
- Local fallback comparison bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0025_mismatched_budget`
- Summary fields: `comparison=mismatched-budget-latest`, A epoch `19`, B validation epoch `25`;
  B records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and validation selected batch indices match A at `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch25 still does not reverse the strict same-budget epoch 19 conclusion.
  B remains stable and checkerboard-safe; checkpoint samples are broadly clean but the hard
  carton/object case still keeps side mass, validation is more fragmented and multi-peaked than A
  epoch 19 across all four samples, and episode 86 remains near-tied without a clear object/table
  lock reduction.

- 2026-06-16 16:54 +08 check: B-side named checkpoint evidence is complete through epoch 23.
  Epoch 24 currently exists only as `latest.ckpt` plus validation output, and watcher generation is
  still in progress, so it is not treated as a complete named checkpoint comparison. Training
  remains healthy in epoch 25; latest checked `logs.json.txt` reached `global_step=33521`, and the
  tmux pane showed about `86/5359` batches through epoch 25. Training tmux, checkpoint watcher, and
  episode watcher are all alive.
- B-side epoch 23 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0023-val_loss=0.018.ckpt`
- B-side epoch 23 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_160030_epoch=0023-val_loss=0.018`
- B-side epoch 23 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0023`
- B-side epoch 23 episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_160002_epoch=0023-val_loss=0.018_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 23 exists.
  This is explicitly a mismatched-budget comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0023_mismatched_budget`
- Local fallback comparison bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_epoch0023_mismatched_budget`
- Summary fields: `comparison=mismatched-budget`, A epoch `19`, B epoch `23`; B records
  `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, and validation selected batch indices match A at `6, 3, 0, 7`.
  Both episode 86 renders contain `3599/3599` frames.
- Visual read: epoch 23 fallback still does not reverse the strict epoch 19 conclusion. B remains
  stable and checkerboard-safe, but the checkpoint hard carton/box case keeps object-biased side
  mass, validation remains more fragmented than A epoch 19 across all four samples, and episode 86
  remains near-tied without visible object/table-lock improvement.

- 2026-06-16 14:13 +08 check: B-side `mix-LNLL` epoch 21 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 22; latest checked `logs.json.txt` reached `global_step=30209`, and the tmux pane showed
  about `2920/5359` batches through epoch 22. Training tmux, checkpoint watcher, and episode
  watcher are all alive.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0021-val_loss=0.018.ckpt`
- B-side checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_135811_epoch=0021-val_loss=0.018`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0021`
- B-side episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_140018_epoch=0021-val_loss=0.018_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 21 exists.
  This is explicitly a mismatched-budget comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0021_mismatched_budget`
- Local fallback comparison bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_epoch0021_mismatched_budget`
- Summary fields: `comparison=mismatched-budget`, A epoch `19`, B epoch `21`; B records
  `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, and validation selected batch indices match A at `6, 3, 0, 7`.
  Both episode 86 renders contain `3599/3599` frames.
- Visual read: epoch 21 fallback continues the epoch 20 trend. B remains stable and
  checkerboard-safe, but checkpoint sample 2 still has object/carton-biased mass, validation
  remains more fragmented than A epoch 19 across all four samples, and episode 86 stays near-tied
  without object/table-lock improvement. Further B checkpoints are mismatched-budget trend checks
  only unless a new same-budget A-side DSNT reference is restored or trained.

- 2026-06-16 13:21 +08 check: B-side `mix-LNLL` epoch 20 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 21; latest checked `logs.json.txt` reached `global_step=28756`, and the tmux pane showed
  about `2468/5359` batches through epoch 21. Training tmux, checkpoint watcher, and episode
  watcher are all alive.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0020-val_loss=0.018.ckpt`
- B-side checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_125700_epoch=0020-val_loss=0.018`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0020`
- B-side episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_125852_epoch=0020-val_loss=0.018_episode86`
- A-side fallback reference: DSNT-mix epoch 19/latest, because no A-side epoch 20 exists.
  This is explicitly a mismatched-budget comparison, not strict same-epoch A/B.
- Server fallback comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0020_mismatched_budget`
- Local fallback comparison bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_epoch0020_mismatched_budget`
- Summary fields: `comparison=mismatched-budget`, A epoch `19`, B epoch `20`; B records
  `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, and validation selected batch indices match A at `6, 3, 0, 7`.
  Both episode 86 renders contain `3599/3599` frames.
- Visual read: this fallback does not reverse the epoch 19 conclusion. B epoch 20 remains stable
  and checkerboard-safe, but checkpoint sample 2 is still object/carton-biased; validation is more
  fragmented than A epoch 19 across all four samples, with table/cup/object side modes. Episode 86
  remains near-tied and clean, with no visible object-lock improvement. Further B checkpoints
  should be treated as mismatched-budget trend checks unless a new A-side DSNT reference is
  restored or trained.

- 2026-06-16 12:08 +08 check: B-side `mix-LNLL` epoch 19 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 20; latest checked `logs.json.txt` reached `global_step=28014`, and the tmux pane showed
  about `4862/5359` batches through epoch 20. Training tmux, checkpoint watcher, and episode
  watcher are all alive. This is a true same-epoch A/B against DSNT-mix epoch 19 outputs, not a
  fallback to a mismatched budget.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0019-val_loss=0.018.ckpt`
- B-side checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_115551_epoch=0019-val_loss=0.018`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0019`
- B-side episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_115734_epoch=0019-val_loss=0.018_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_152145_epoch=0019-val_loss=0.016`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0019`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_152255_epoch=0019-val_loss=0.016_episode86`
- Server direct A/B bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0019`
- Local direct A/B bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_epoch0019`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Validation selected batch indices match A at `6, 3, 0, 7`; both episode 86 renders contain
  `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 19 still favors DSNT-mix. A has the better scalar checkpoint name
  (`0.016` vs B `0.018`). Checkpoint samples 0/1/3 remain broadly close, but B is not cleaner;
  hard sample 2 loses target-centered mass and keeps a vertical object/carton-biased structure.
  Validation is worse under B across all four samples, with more fragmented table/cup/object side
  modes. Episode 86 frames remain clean and checkerboard-safe, but frame 0 still has the same
  tabletop/object lock and later frames are near-tied rather than improved. Net: point-NLL at
  weight `0.001` has not beaten the DSNT `XY` anchor through the preserved A-side budget limit of
  epoch 19.

- 2026-06-16 11:02 +08 check: B-side `mix-LNLL` epoch 18 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 19; latest checked `logs.json.txt` reached `global_step=25666`, and the tmux pane showed
  about `830/5359` batches through epoch 19. Training tmux, checkpoint watcher, and episode
  watcher are all alive. This is a true same-epoch A/B against DSNT-mix epoch 18 outputs, not a
  fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0018-val_loss=0.017.ckpt`
- B-side checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_105441_epoch=0018-val_loss=0.017`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0018`
- B-side episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_105610_epoch=0018-val_loss=0.017_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_142035_epoch=0018-val_loss=0.015`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0018`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_142137_epoch=0018-val_loss=0.015_episode86`
- Server direct A/B bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0018_vs_nll_epoch0018`
- Local direct A/B bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0018_vs_nll_epoch0018`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples use the same selected validation indices as A: `6600, 1341, 9843, 11637`;
  validation selected batch indices match A at `6, 3, 0, 7`. Both episode 86 renders contain
  `3599/3599` frames at 15 FPS; the direct A/B video is `2592x256`, `3599` frames, `239.93s`.
- Same-epoch visual read: epoch 18 further favors DSNT-mix. A has the better scalar checkpoint
  name (`0.015` vs B `0.017`). Checkpoint samples 0/1 are close, but B adds more side/upper mass
  on sample 1, remains worse on hard sample 2 with vertical carton/object-biased lobes, and shifts
  sample 3 left/off-target relative to A. Validation is the clearest negative signal: B is more
  fragmented across all four samples, with extra table/cup/object blobs and split modes. Episode
  86 remains clean and checkerboard-safe; frames 60/120/180 are near-tied, while frame 0 still has
  the same tabletop/object lock. Net: replacing DSNT `XY` with point NLL at weight `0.001` has not
  improved the multi-modal temporal-label failure mode through epoch 18.

- 2026-06-16 10:04 +08 check: B-side `mix-LNLL` epoch 17 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 18; latest checked `logs.json.txt` reached `global_step=24182`, and the tmux pane showed
  about `252/5359` batches through epoch 18. This is a true same-epoch A/B against DSNT-mix
  epoch 17 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0017-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_095332_epoch=0017-val_loss=0.015`
- B-side validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0017`
- B-side episode 86 preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_095444_epoch=0017-val_loss=0.015_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_131925_epoch=0017-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0017`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_132024_epoch=0017-val_loss=0.014_episode86`
- Server direct A/B bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0017_vs_nll_epoch0017`
- Local direct A/B bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0017_vs_nll_epoch0017`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples use the same selected validation indices as A: `6600, 1341, 9843, 11637`;
  validation selected batch indices match A at `6, 3, 0, 7`. Episode 86 rendered `3599/3599`
  frames at 15 FPS.
- Same-epoch visual read: epoch 17 still favors DSNT-mix. A has the better scalar checkpoint name
  (`0.014` vs B `0.015`). Checkpoint samples 0/1/3 remain close and stable, but B does not recover
  hard sample 2: the carton/box case keeps extra scene-texture/object-biased mass instead of a
  cleaner target-centered distribution. Validation is again the strongest negative signal for B:
  all four B samples are more fragmented or shifted toward table/cup/object structure than A.
  Episode 86 frames 60/120/180 remain near-tied and clean, with no checkerboard regression, but
  frame 0 still shows the same tabletop/object lock. Net: point-NLL at weight `0.001` remains
  checkerboard-safe, but same-epoch evidence through epoch 17 still underperforms the DSNT `XY`
  anchor.

- 2026-06-16 09:08 +08 check: B-side `mix-LNLL` epoch 16 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 17; latest checked `logs.json.txt` reached `global_step=23124`, and the tmux pane showed
  about `1374/5359` batches through epoch 17. This is a true same-epoch A/B against DSNT-mix
  epoch 16 outputs, not a fallback to the preserved epoch 19 reference.
- Operational note: the epoch 16 A/B bundle was generated server-side at
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0016_vs_nll_epoch0016`
  and copied locally to
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0016_vs_nll_epoch0016`. `W:` remains tight,
  with about 110 MB free after this update.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0016-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_085520_epoch=0016-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0015_vs_nll_epoch0015\B_ckpt`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0016`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0015_vs_nll_epoch0015\B_val`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_085321_epoch=0016-val_loss=0.015_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0015_vs_nll_epoch0015\B_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_122112_epoch=0016-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0016`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_121906_epoch=0016-val_loss=0.013_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0015_vs_nll_epoch0015\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0016_vs_nll_epoch0016/episode86_A_dsnt_vs_B_nll_epoch0016.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0015_vs_nll_epoch0015\comparisons\episode86_A_dsnt_vs_B_nll_epoch0015.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples use the same selected validation indices as A: `6600, 1341, 9843, 11637`;
  validation selected batch indices match A at `6, 3, 0, 7`. Episode 86 rendered `3599/3599`
  frames at 15 FPS.
- Same-epoch visual read: epoch 16 still favors DSNT-mix. A has the better scalar checkpoint name
  (`0.013` vs B `0.015`). Checkpoint samples 0/1/3 remain close, but B is worse on hard sample 2:
  the carton/box case becomes a vertical multi-lobe structure and the decoded mass clings more to
  scene texture than the gaze point. Validation is the strongest negative signal for B: all four B
  samples are more fragmented or shifted toward table/cup/object structure than A. Episode 86
  frames 60/120/180 remain essentially tied and clean, with no checkerboard regression, but frame 0
  still shows the same tabletop/object lock on both sides. Net: point-NLL at weight `0.001`
  remains stable and checkerboard-safe, but same-epoch evidence through epoch 16 continues to
  underperform the DSNT `XY` anchor.

- 2026-06-16 07:08 +08 check: B-side `mix-LNLL` epoch 14 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 15; latest checked `logs.json.txt` reached `global_step=20472`, and the tmux pane showed
  about `1494/5359` batches through epoch 15. This is a true same-epoch A/B against DSNT-mix
  epoch 14 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0014-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_065309_epoch=0014-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0014_vs_nll_epoch0014\B_ckpt`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0014`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0014_vs_nll_epoch0014\B_val`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_065339_epoch=0014-val_loss=0.015_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0014_vs_nll_epoch0014\B_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_101847_epoch=0014-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0014`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_101931_epoch=0014-val_loss=0.013_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0014_vs_nll_epoch0014\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0014_vs_nll_epoch0014/episode86_A_dsnt_vs_B_nll_epoch0014.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0014_vs_nll_epoch0014\comparisons\episode86_A_dsnt_vs_B_nll_epoch0014.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples use the same selected validation indices as A: `6600, 1341, 9843, 11637`.
  Episode 86 rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 14 still favors DSNT-mix. A has the lower scalar checkpoint naming
  (`0.013` vs B `0.015`). Checkpoint samples stay close, but B is not a decisive improvement:
  hard sample 2 keeps a vertical/object-biased multi-lobe structure and sample 3 is only slightly
  more spread. Validation is the negative signal for B: all four samples are more fragmented or
  shifted toward table/object structure, with sample 2 moving away from the gaze target into a
  lower table/cup/object blob. Episode 86 frames 0/60/120/180 remain essentially tied and clean,
  with no checkerboard regression and no clear object-lock reduction. Net: point-NLL at weight
  `0.001` remains stable and checkerboard-safe, but same-epoch evidence through epoch 14 continues
  to underperform the DSNT `XY` anchor.

- 2026-06-16 06:32 +08 check: B-side `mix-LNLL` epoch 13 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 14; latest checked `logs.json.txt` reached `global_step=19662`, and the tmux pane showed
  about `3612/5359` batches through epoch 14. This is a true same-epoch A/B against DSNT-mix
  epoch 13 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0013-val_loss=0.016.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_055200_epoch=0013-val_loss=0.016`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0013_vs_nll_epoch0013\B_ckpt`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0013_vs_nll_epoch0013\B_val`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_055224_epoch=0013-val_loss=0.016_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0013_vs_nll_epoch0013\B_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_091736_epoch=0013-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0013`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_091853_epoch=0013-val_loss=0.014_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0013_vs_nll_epoch0013\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0013_vs_nll_epoch0013/episode86_A_dsnt_vs_B_nll_epoch0013.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0013_vs_nll_epoch0013\comparisons\episode86_A_dsnt_vs_B_nll_epoch0013.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples use the same selected validation indices as A: `6600, 1341, 9843, 11637`.
  Episode 86 rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 13 still favors DSNT-mix. A has the lower scalar checkpoint
  naming (`0.014` vs B `0.016`). Checkpoint samples are mostly near-tied; B reduces the vertical
  multi-peak strip on hard sample 2 but keeps a lower carton/object-biased lobe, so it is not a
  clean target lock. Validation is again the negative signal for B: samples 0/1/2/3 are more
  split or shifted toward table/object structure, with sample 2 moving away from the gaze target
  into a lower table/cup/object blob. Episode 86 frames 0/60/120/180 remain essentially tied and
  clean, with frame 0 still locked to tabletop/object structure and no clear object-lock reduction.
  Net: point-NLL at weight `0.001` remains stable and checkerboard-safe, but same-epoch evidence
  through epoch 13 continues to underperform the DSNT `XY` anchor.

- 2026-06-16 05:38 +08 check: B-side `mix-LNLL` epoch 12 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 13; latest checked `logs.json.txt` reached `global_step=17756`, and the tmux pane showed
  about `1346/5359` batches through epoch 13. This is a true same-epoch A/B against DSNT-mix
  epoch 12 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0012-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_045349_epoch=0012-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0012_vs_nll_epoch0012\B_nll_ckpt_preview\20260616_045349_epoch=0012-val_loss=0.015`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0012`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0012_vs_nll_epoch0012\B_nll_val_preview\epoch_0012`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_045403_epoch=0012-val_loss=0.015_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0012_vs_nll_epoch0012\B_nll_episode86\20260616_045403_epoch=0012-val_loss=0.015_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_081625_epoch=0012-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0012`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_081732_epoch=0012-val_loss=0.013_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0012_vs_nll_epoch0012\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0012_vs_nll_epoch0012/episode86_A_dsnt_vs_B_nll_epoch0012.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0012_vs_nll_epoch0012\comparisons\episode86_A_dsnt_vs_B_nll_epoch0012.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples are the same selected validation indices as A: `6600, 1341, 9843, 11637`.
  Episode 86 rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 12 is still not a B-side win. A has the lower scalar checkpoint
  naming (`0.013` vs B `0.015`). B checkpoint samples are sometimes compact and close, but the
  validation contact sheet is worse in the target failure mode: B adds off-target table/object
  blobs and split modes across all four samples, with sample 2 still especially object/scene
  biased. Episode 86 frames 0/60/120/180 remain essentially tied and clean, with no checkerboard
  regression, but no clear tabletop/object-lock reduction. Net: point-NLL at weight `0.001` remains
  stable but continues to underperform the DSNT `XY` anchor in same-epoch A/B.

- 2026-06-16 04:08 +08 check: B-side `mix-LNLL` epoch 11 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 12; latest checked `logs.json.txt` reached `global_step=16237`, and the tmux pane showed
  about `632/5359` batches through epoch 12. This is a true same-epoch A/B against DSNT-mix
  epoch 11 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0011-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_035238_epoch=0011-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0011_vs_nll_epoch0011\B_nll_ckpt_preview\20260616_035238_epoch=0011-val_loss=0.015`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0011`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0011_vs_nll_epoch0011\B_nll_val_preview\epoch_0011`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_035244_epoch=0011-val_loss=0.015_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0011_vs_nll_epoch0011\B_nll_episode86\20260616_035244_epoch=0011-val_loss=0.015_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_071515_epoch=0011-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0011`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_071720_epoch=0011-val_loss=0.014_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0011_vs_nll_epoch0011\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0011_vs_nll_epoch0011/episode86_A_dsnt_vs_B_nll_epoch0011.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0011_vs_nll_epoch0011\comparisons\episode86_A_dsnt_vs_B_nll_epoch0011.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples are the same selected validation indices as A: `6600, 1341, 9843, 11637`.
  Episode 86 rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 11 remains stable but still favors DSNT-mix A overall. A has the
  lower scalar checkpoint naming (`0.014` vs B `0.015`). Checkpoint samples are mixed rather than
  a clean B regression: B is close on samples 0/1/3, but sample 2 is still a separated vertical
  multi-lobe structure around the carton/scene edge. Validation remains the deciding negative
  signal for B: all four B samples are more split/fragmented with extra table/object-biased lobes,
  while A is more compact around its dominant mode. Episode 86 frames 0/60/120/180 are essentially
  tied and clean, with no checkerboard regression, but also no clear tabletop/object-lock
  reduction. Net: point-NLL at weight `0.001` is stable but still not outperforming the DSNT `XY`
  anchor.

- 2026-06-16 03:30 +08 check: B-side `mix-LNLL` epoch 10 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 11; latest checked `logs.json.txt` reached `global_step=15656`, and the tmux pane showed
  about `3062/5359` batches through epoch 11. This is a true same-epoch A/B against DSNT-mix
  epoch 10 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0010-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_025129_epoch=0010-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0010_vs_nll_epoch0010\B_nll_ckpt_preview\20260616_025129_epoch=0010-val_loss=0.015`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0010`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0010_vs_nll_epoch0010\B_nll_val_preview\epoch_0010`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_025118_epoch=0010-val_loss=0.015_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0010_vs_nll_epoch0010\B_nll_episode86\20260616_025118_epoch=0010-val_loss=0.015_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_061703_epoch=0010-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0010`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_061617_epoch=0010-val_loss=0.013_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0010_vs_nll_epoch0010\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0010_vs_nll_epoch0010/episode86_A_dsnt_vs_B_nll_epoch0010.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0010_vs_nll_epoch0010\comparisons\episode86_A_dsnt_vs_B_nll_epoch0010.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples are the same selected validation indices as A: `6600, 1341, 9843, 11637`.
  Episode 86 rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 10 is a stronger DSNT-mix A win than epoch 9. A has the lower
  scalar checkpoint naming (`0.013` vs B `0.015`). B remains stable and has no checkerboard
  regression, but checkpoint sample 0 adds extra side/upper modes, sample 2 still has a separated
  vertical multi-lobe structure, and sample 3 picks up a side blob. Validation is clearly more
  fragmented on B: samples 0/1/2/3 all split into multiple object/table-biased lobes, while A is
  more compact near its dominant mode. Episode 86 frames 0/60/120/180 are essentially tied and
  clean, but still do not show a clear tabletop/object-lock reduction. Net: point-NLL is stable,
  but at weight `0.001` it is not outperforming the DSNT `XY` anchor.

- 2026-06-16 02:21 +08 check: B-side `mix-LNLL` epoch 9 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 10; latest checked `logs.json.txt` reached `global_step=14144`, and the tmux pane showed
  about `2984/5359` batches through epoch 10. This is a true same-epoch A/B against DSNT-mix
  epoch 9 outputs, not a fallback to the preserved epoch 19 reference.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0009-val_loss=0.015.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_015018_epoch=0009-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0009_vs_nll_epoch0009\B_nll_ckpt_preview\20260616_015018_epoch=0009-val_loss=0.015`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0009`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0009_vs_nll_epoch0009\B_nll_val_preview\epoch_0009`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_014955_epoch=0009-val_loss=0.015_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0009_vs_nll_epoch0009\B_nll_episode86\20260616_014955_epoch=0009-val_loss=0.015_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_051552_epoch=0009-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0009`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_051501_epoch=0009-val_loss=0.014_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0009_vs_nll_epoch0009\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0009_vs_nll_epoch0009/episode86_A_dsnt_vs_B_nll_epoch0009.mp4`
- Local direct A/B episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0009_vs_nll_epoch0009\comparisons\episode86_A_dsnt_vs_B_nll_epoch0009.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples are the same selected validation indices as A: `6600, 1341, 9843, 11637`.
  Episode 86 rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 9 still favors DSNT-mix A overall. A has the lower scalar
  checkpoint naming (`0.014` vs B `0.015`). B is stable and does not show a checkerboard
  regression, but the checkpoint sheet shows extra side/double modes on samples 0 and 2 and no
  clear cleanup on samples 1/3. Validation is the stronger negative signal for B: the B-side
  predictions are more fragmented or split across samples 0/1/2/3, while A stays more compact
  around its dominant mode. Episode 86 frames 0/60/120/180 are essentially tied and clean, with no
  obvious new artifact, but also no clear tabletop/object-lock reduction. Net: continue B only if
  later epochs are needed for completeness; current same-budget A/B still favors DSNT-mix.

- 2026-06-15 21:20 +08 check: B-side `mix-LNLL` epoch 4 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 5; latest checked `logs.json.txt` reached `global_step=7598`, and the tmux pane showed
  about `3598/5359` batches through epoch 5. This is a true same-epoch A/B against DSNT-mix
  epoch 4 outputs.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0004-val_loss=0.016.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_204423_epoch=0004-val_loss=0.016`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0004_vs_nll_epoch0004\B_nll_ckpt_preview\20260615_204423_epoch=0004-val_loss=0.016`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0004`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0004_vs_nll_epoch0004\B_nll_val_preview\epoch_0004`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_204514_epoch=0004-val_loss=0.016_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0004_vs_nll_epoch0004\B_nll_episode86\20260615_204514_epoch=0004-val_loss=0.016_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_001259_epoch=0004-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0004`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_001220_epoch=0004-val_loss=0.013_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0004_vs_nll_epoch0004\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0004_vs_nll_epoch0004/episode86_A_dsnt_vs_B_nll_epoch0004.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples are the same selected indices as A: `6600, 1341, 9843, 11637`. Episode 86
  rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 4 is still mixed and does not make `mix-LNLL` the winner. B has
  improved numerically from its own epoch 3 (`0.018 -> 0.016`), but same-epoch A remains lower
  (`0.013`). In checkpoint samples, B is slightly more favorable on the hard carton sample 2, but
  samples 0/1/3 are not cleaner than A. Validation views are the main concern: B shows more
  multi-blob fragmentation and vertical trailing on samples 0/1/3, while only sample 2 tightens.
  Episode 86 frames 0/60/120/180 remain essentially tied with no checkerboard regression, but also
  no clear tabletop/object-lock reduction. Net: continue B, but current A/B still favors DSNT-mix
  overall.

- 2026-06-15 19:40 +08 check: B-side `mix-LNLL` epoch 2 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 3; latest checked `logs.json.txt` reached `global_step=5106`, and the tmux pane showed
  about `4348/5359` batches through epoch 3. This is a true same-epoch A/B against DSNT-mix
  epoch 2 outputs.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.020.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_184200_epoch=0002-val_loss=0.020`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0002_vs_nll_epoch0002\B_nll_ckpt_preview\20260615_184200_epoch=0002-val_loss=0.020`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0002`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0002_vs_nll_epoch0002\B_nll_val_preview\epoch_0002`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_184226_epoch=0002-val_loss=0.020_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0002_vs_nll_epoch0002\B_nll_episode86\20260615_184226_epoch=0002-val_loss=0.020_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_221041_epoch=0002-val_loss=0.018`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0002`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_221300_epoch=0002-val_loss=0.018_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0002_vs_nll_epoch0002\comparisons`
- Server direct A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0002_vs_nll_epoch0002/episode86_A_dsnt_vs_B_nll_epoch0002.mp4`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Checkpoint samples are the same selected indices as A: `6600, 1341, 9843, 11637`. Episode 86
  rendered `3599/3599` frames at 15 FPS.
- Same-epoch visual read: epoch 2 still does not make `mix-LNLL` a clear winner. A has lower scalar
  validation naming (`0.018`) than B (`0.020`). In checkpoint samples, B is close to A but adds a
  weak upper/side scatter in sample 0, remains only slightly tighter on sample 1, does not fix the
  hard carton/scene-structure lock in sample 2, and is near-identical on sample 3. Validation
  samples show B as slightly more fragmented in several multi-blob cases. Episode 86 frames
  0/60/120/180 are essentially tied; there is no obvious checkerboard regression, but also no clear
  reduction of tabletop/object-lock bias. Net: keep B training, but current evidence favors
  DSNT-mix on scalar loss and marks LNLL as not-yet-proven.

- 2026-06-15 17:58 +08 check: B-side `mix-LNLL` epoch 1 completed with checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 2, latest checked `global_step=2913`, about `938/5359` batches into the epoch. This is a
  true same-epoch A/B against DSNT-mix epoch 1 outputs.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.025.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_174048_epoch=0001-val_loss=0.025`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0001_vs_nll_epoch0001\B_nll_ckpt_preview\20260615_174048_epoch=0001-val_loss=0.025`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0001`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0001_vs_nll_epoch0001\B_nll_val_preview\epoch_0001`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_174106_epoch=0001-val_loss=0.025_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0001_vs_nll_epoch0001\B_nll_episode86\20260615_174106_epoch=0001-val_loss=0.025_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_211230_epoch=0001-val_loss=0.024`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0001`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_211141_epoch=0001-val_loss=0.024_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0001_vs_nll_epoch0001\comparisons`
- Summary fields: B records `heatmap_xy_loss_weight=0.0`,
  `heatmap_point_nll_loss_weight=0.001`, `heatmap_js_loss_weight=0.10`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, no EMA, `num_inference_steps=8`.
  Episode 86 rendered `3599/3599` frames at 15 FPS in `552.1143245697021s`.
- Same-epoch visual read: epoch 1 remains mixed/tied. Scalar validation naming is slightly better
  for A (`0.024`) than B (`0.025`). In checkpoint samples, B is tighter/closer to gaze on samples
  0/1/3, but sample 2 is worse, with a more dispersed multi-peak response around the carton/scene
  structure. In validation samples, B is closer to the gaze area on samples 0/3, but has extra weak
  lower/right blobs in samples 1/2. Episode 86 frames 0/60/120/180 are essentially tied, with no
  clear object-lock improvement. Net: `mix-LNLL` is still healthy and plausible, but not yet a clear
  win over DSNT-mix; continue to later epochs.

- 2026-06-15 17:00 +08 check: B-side `mix-LNLL` epoch 0 completed and has full checkpoint,
  validation, checkpoint-preview, and episode 86 preview evidence. Training remains healthy in
  epoch 1, latest checked `global_step=2037`, about `2792/5359` batches into the epoch. This is a
  true same-epoch A/B against the preserved DSNT-mix epoch 0 outputs, not a mismatched-budget
  fallback.
- B-side checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.031.ckpt`
- B-side checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_163925_epoch=0000-val_loss=0.031`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0000_vs_nll_epoch0000\B_nll_ckpt_preview\20260615_163925_epoch=0000-val_loss=0.031`
- B-side validation preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0000`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0000_vs_nll_epoch0000\B_nll_val_preview\epoch_0000`
- B-side episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_163925_epoch=0000-val_loss=0.031_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0000_vs_nll_epoch0000\B_nll_episode86\20260615_163925_epoch=0000-val_loss=0.031_episode86`
- A-side same-epoch reference:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_201120_epoch=0000-val_loss=0.031`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0000`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_201028_epoch=0000-val_loss=0.031_episode86`
- Local direct A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0000_vs_nll_epoch0000\comparisons`
- Summary fields: both A and B use `source=open`, `split=val`, no EMA, `episode=86`,
  `rendered_frames=3599`, temporal-window target
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`, `num_inference_steps=8`. B additionally
  records `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`, and
  `heatmap_js_loss_weight=0.10`.
- Same-epoch visual read: epoch 0 A/B is essentially tied and still early. B `mix-LNLL` does not
  show obvious collapse or checkerboard worsening, but it also does not yet break the DSNT-mix
  object/table locks. Checkpoint samples are nearly identical; B has a slightly stronger right-side
  small hotspot in sample 1 and a slightly stronger upper side peak in sample 3. Validation samples
  remain scene/object biased; B slightly weakens one left isolated hotspot in validation sample 2,
  but the main response is still off-target. Episode 86 frames 0/60/120/180 are visually near
  identical to A, with the same tabletop/object-lock pattern. Net: keep training to later epochs;
  epoch 0 only confirms the LNLL objective is healthy enough to continue.

- 2026-06-15 15:40 +08 check: switched from single-run monitoring to a real A/B setup. The
  DSNT-mix A-side run was stopped after preserving epoch 19 previews. The new `mix-LNLL` B-side
  run is active and healthy in epoch 0; `logs.json.txt` has reached `global_step=41`, with all
  8 GPUs busy. The first observed B-side metrics include `train_heatmap_loss` about `0.953`,
  `train_heatmap_point_nll_loss` about `3.80`, and `train_heatmap_js_loss` about `0.204`; with
  weights, the decoded final contribution is roughly `0.001*3.80 + 0.10*0.204`, and no XY term is
  applied. B-side checkpoint and episode watchers are active and currently waiting for the first
  checkpoint directory to appear.
- Active B-side output:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_`
- Active B-side tmux sessions:
  `gaze_wam_open_cosmos_temporal_mixed_nll_8gpu`,
  `gaze_wam_temporal_mixed_nll_ckpt_preview_watch`,
  `gaze_wam_temporal_mixed_nll_episode_preview_watch`.
- Automation `check-temporal-gaze-wam-checkpoint` now monitors B-side `mix-LNLL` checkpoints and,
  on every new B preview, pulls B outputs plus the matching A-side DSNT-mix epoch when available.
  If the same A epoch is missing, it falls back to the preserved A epoch 19 reference and marks the
  comparison as mismatched-budget.

- 2026-06-15 12:40 +08 check: mixed-loss epoch 16 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 17; the latest checked `logs.json.txt`
  reached `global_step=22902`. The new checkpoint preview is epoch 16 with `val_loss=0.013`, and
  the full episode 86 render completed successfully. Per user feedback, the comparison was upgraded
  from historical-reference A/B-like reading to a direct pulled-output A/B against the pure
  temporal-window epoch 28/latest checkpoint.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0016-val_loss=0.013.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0016`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0016_val_preview\epoch_0016`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_122112_epoch=0016-val_loss=0.013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0016_ckpt_preview\20260615_122112_epoch=0016-val_loss=0.013`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_121906_epoch=0016-val_loss=0.013_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0016_episode86\20260615_121906_epoch=0016-val_loss=0.013_episode86\comparison.mp4`
- Direct A/B artifacts:
  - pure temporal epoch 28/latest checkpoint preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_pure_temporal_epoch0028_latest_ckpt_preview\20260614_184212_latest`
  - pure temporal epoch 28 validation preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_pure_temporal_epoch0028_val_preview\epoch_0028`
  - pure temporal epoch 28 episode local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_pure_temporal_epoch0028_episode86\20260614_184351_latest_episode86\comparison.mp4`
  - A/B composites local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_temporal_pure_epoch0028_vs_mixed_epoch0016`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86 rendered
  `3599/3599` frames at 15 FPS in `550.8630373477936s`.
- Direct A/B visual read against pure temporal-window epoch 28/latest: epoch 16 recovers the scalar
  loss to `0.013`, but the pulled-output A/B does not show a mixed-loss win. In checkpoint samples,
  mixed epoch 16 is comparable on samples 0/1 and a little cleaner on sample 3, but sample 2 still
  forms a vertical multi-peak response along the carton/scene edge, matching the hard failure mode.
  In the validation samples, mixed epoch 16 is worse than the pure temporal epoch 28 reference:
  samples 2 and 3 show more object-related side peaks and left/tabletop bias. Episode 86 frame 0
  remains a tabletop/object lock in both A and B, and frames 60/120/180 are broadly comparable.
  Net: the strict pulled-output A/B supports the same conclusion as the historical comparison:
  mixed decoded final loss has not beaten pure temporal epoch 28 on hard scene-prior/object-lock
  cases.

- 2026-06-15 11:55 +08 check: mixed-loss epoch 15 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 16, about `2724/5359` batches at roughly
  `1.51-1.52 it/s`; `logs.json.txt` reached `global_step=22120`. The new checkpoint preview is
  epoch 15 with `val_loss=0.015`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0015-val_loss=0.015.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_val_preview\epoch_0015`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_111959_epoch=0015-val_loss=0.015`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_ckpt_preview\20260615_111959_epoch=0015-val_loss=0.015`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_111751_epoch=0015-val_loss=0.015_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_episode86\20260615_111751_epoch=0015-val_loss=0.015_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86 rendered
  `3599/3599` frames at 15 FPS in `545.8032233715057s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 15 regresses numerically
  from epoch 14's `0.013` to `0.015` and does not recover the hard cases. Checkpoint samples 0 and
  3 remain compact but sample 0 is slightly lower than target; sample 1 is close but still has
  object-cluster pull; sample 2 remains a hard failure, with decoded mass along the carton/scene
  edge instead of the gaze marker. The epoch 15 validation preview is strongly object-prior biased:
  all four selected validation samples shift toward left-side tabletop objects while the target
  remains at the gaze point. Episode 86 frame 0 still locks to the tabletop/object area, while
  frames 60/120/180 remain close. Net: epoch 15 confirms plateau/regression, not a hard-case
  breakthrough.

- 2026-06-15 11:05 +08 check: mixed-loss epoch 14 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 15, about `4420/5359` batches at roughly
  `1.51-1.52 it/s`; `logs.json.txt` reached `global_step=21204`. The new checkpoint preview is
  epoch 14 with `val_loss=0.013`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0014-val_loss=0.013.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0014`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_val_preview\epoch_0014`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_101847_epoch=0014-val_loss=0.013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_ckpt_preview\20260615_101847_epoch=0014-val_loss=0.013`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_101931_epoch=0014-val_loss=0.013_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_episode86\20260615_101931_epoch=0014-val_loss=0.013_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86 rendered
  `3599/3599` frames at 15 FPS in `552.4642882347107s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 14 improves the scalar
  checkpoint loss from epoch 13's `0.014` to `0.013`, but remains on the same visual plateau.
  Checkpoint samples 0 and 3 are compact and usable; sample 1 keeps an off-target left/lower drag;
  sample 2 still forms a vertical multi-peak strip along the carton/scene edge rather than
  collapsing onto the gaze marker. The epoch 14 validation preview is also object/scene-prior
  biased, with all four validation samples shifted toward left-side tabletop objects while the
  target stays on the gaze point. Episode 86 frame 0 still locks to the tabletop/object area, while
  frames 60/120/180 are close. Net: no checkerboard collapse, but still no hard-case breakthrough
  over the pure temporal epoch 28 evidence.

- 2026-06-15 09:30 +08 check: mixed-loss epoch 13 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 14, about `642/5359` batches at roughly
  `1.36-1.42 it/s`; `logs.json.txt` reached `global_step=18919`. Epoch 12 also completed during
  the gap, but epoch 13 is the newest complete preview and is the one pulled locally. The new
  checkpoint preview is epoch 13 with `val_loss=0.014`, and the full episode 86 render completed
  successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0013-val_loss=0.014.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_val_preview\epoch_0013`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_091736_epoch=0013-val_loss=0.014`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_ckpt_preview\20260615_091736_epoch=0013-val_loss=0.014`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_091853_epoch=0013-val_loss=0.014_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_episode86\20260615_091853_epoch=0013-val_loss=0.014_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86 rendered
  `3599/3599` frames at 15 FPS in `508.36743450164795s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 13 remains on the same
  plateau (`val_loss=0.014`). Sample 0 and sample 3 stay compact and usable, but sample 2 regresses
  to a stronger vertical multi-peak strip along the carton/scene edge instead of the gaze marker.
  Episode 86 frame 0 still locks to the tabletop/object area rather than the left-up gaze target.
  Net: no checkerboard collapse, but the decoded final distribution term still has not beaten the
  pure temporal epoch 28 hard-case evidence.

- 2026-06-15 07:36 +08 check: mixed-loss epoch 11 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 12, about `2044/5359` batches at roughly
  `1.50-1.51 it/s`; `logs.json.txt` reached `global_step=16590`. The new checkpoint preview is
  epoch 11 with `val_loss=0.014`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0011-val_loss=0.014.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0011`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_val_preview\epoch_0011`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_071515_epoch=0011-val_loss=0.014`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_ckpt_preview\20260615_071515_epoch=0011-val_loss=0.014`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_071720_epoch=0011-val_loss=0.014_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_episode86\20260615_071720_epoch=0011-val_loss=0.014_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86 rendered
  `3599/3599` frames at 15 FPS in `483.65689849853516s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 11 regresses back to
  `0.014`, and the easy cases stay compact and usable. Sample 2 is still the key hard failure,
  locking to the carton/scene edge rather than the gaze marker, and episode 86 frame 0 still locks
  to the tabletop/object area. Net: no checkerboard collapse, but the decoded final distribution
  term still has not beaten the pure temporal epoch 28 hard-case evidence.

- 2026-06-15 06:31 +08 check: mixed-loss epoch 10 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 11, about `1472/5359` batches at roughly
  `1.49-1.50 it/s`; `logs.json.txt` reached `global_step=15107`. The new checkpoint preview is
  epoch 10 with `val_loss=0.013`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0010-val_loss=0.013.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0010`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_val_preview\epoch_0010`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_061703_epoch=0010-val_loss=0.013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_ckpt_preview\20260615_061703_epoch=0010-val_loss=0.013`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_061617_epoch=0010-val_loss=0.013_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_episode86\20260615_061617_epoch=0010-val_loss=0.013_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86 rendered
  `3599/3599` frames at 15 FPS in `533.8579666614532s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 10 recovers the scalar
  trend to `0.013` after epoch 9's `0.014`, and the easy cases stay compact and usable. Sample 2 is
  still the key hard failure, locking to the carton/scene edge rather than the gaze marker, and
  episode 86 frame 0 still locks to the tabletop/object area. Net: no checkerboard collapse, but
  the decoded final distribution term still has not beaten the pure temporal epoch 28 hard-case
  evidence.

- 2026-06-15 05:51 +08 check: mixed-loss epoch 9 validation completed and both watcher previews
  are now complete. Training remains healthy in epoch 10, about `3514/5359` batches at roughly
  `1.51-1.52 it/s`; `logs.json.txt` reached `global_step=14276`. The new checkpoint preview is
  epoch 9 with `val_loss=0.014`, and the full episode 86 render completed successfully after the
  earlier transient half-written state.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0009-val_loss=0.014.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0009`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_val_preview\epoch_0009`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_051552_epoch=0009-val_loss=0.014`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_ckpt_preview\20260615_051552_epoch=0009-val_loss=0.014`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_051501_epoch=0009-val_loss=0.014_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_episode86\20260615_051501_epoch=0009-val_loss=0.014_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86
  rendered `3599/3599` frames at 15 FPS in `547.384391784668s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 9 worsens the scalar trend
  again (`0.012 -> 0.013 -> 0.014`) and still does not recover the hard cases. Sample 0 is compact
  but slightly right/down biased with faint lower-left speckles. Sample 1 is compact but left/low of
  the marker and pulled toward the object cluster. Sample 2 remains the key failure: the predicted
  mass still locks to the carton/scene edge instead of the gaze marker, though the vertical strip is
  narrower than epoch 8. Sample 3 remains compact and near target with mild above/right bias.
  Episode 86 frames 60/120/180 remain close and stable; frame 0 still locks to the tabletop/object
  area. Net: no checkerboard collapse, but mixed decoded final loss continues to plateau/regress
  behind the pure temporal epoch 28 hard-case evidence.

- 2026-06-15 04:41 +08 check: mixed-loss epoch 8 validation completed and both watcher previews
  are available. Training remains healthy in epoch 9, about `2550/5359` batches at roughly
  `1.49-1.51 it/s`; `logs.json.txt` reached `global_step=12697`. The new checkpoint preview is
  epoch 8 with `val_loss=0.013`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0008-val_loss=0.013.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0008`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_val_preview\epoch_0008`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_041443_epoch=0008-val_loss=0.013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_ckpt_preview\20260615_041443_epoch=0008-val_loss=0.013`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_041640_epoch=0008-val_loss=0.013_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_episode86\20260615_041640_epoch=0008-val_loss=0.013_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86
  rendered `3599/3599` frames at 15 FPS in `551.8718445301056s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 8 regresses slightly in
  scalar validation loss (`0.012 -> 0.013`) and is not a hard-case breakthrough. Sample 0 is cleaner
  than epoch 7, with the earlier upper double mode mostly gone, but it still has faint lower-left
  speckles. Sample 1 remains compact and usable with a small left/low bias. Sample 2 remains the
  clearest failure: the prediction is still a vertical multi-lobe strip along the carton/scene edge
  rather than centered on the gaze marker. Sample 3 is compact and near target with a mild
  above/right bias. Episode 86 frames 60/120/180 are close and stable; frame 0 still locks to the
  tabletop/object area. Net: no checkerboard collapse, but epoch 8 further supports the view that
  the mixed decoded final loss is plateauing behind the pure temporal epoch 28 hard-case evidence.

- 2026-06-15 03:31 +08 check: mixed-loss epoch 7 validation completed and both watcher previews
  are available. Training remains healthy in epoch 8, about `1682/5359` batches at roughly
  `1.49-1.50 it/s`; `logs.json.txt` reached `global_step=11140`. The new checkpoint preview is
  epoch 7 with `val_loss=0.012`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0007-val_loss=0.012.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0007`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_val_preview\epoch_0007`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_031631_epoch=0007-val_loss=0.012`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_ckpt_preview\20260615_031631_epoch=0007-val_loss=0.012`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_031523_epoch=0007-val_loss=0.012_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_episode86\20260615_031523_epoch=0007-val_loss=0.012_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `heatmap_distribution_mode=intensity_softplus`,
  `num_inference_steps=8`, and selected sample indices `6600,1341,9843,11637`. Episode 86
  rendered `3599/3599` frames at 15 FPS in `548.4395904541016s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 7 remains at the
  `val_loss=0.012` plateau. Sample 1 and sample 3 are usable and compact; sample 1 is a touch
  left/low and sample 3 remains slightly above/right. Sample 0 is mostly target-centered but keeps
  a small upper double mode and persistent lower-left speckles. Sample 2 is still the clearest hard
  failure: the prediction follows the carton edge as a vertical multi-lobe structure rather than
  collapsing onto the gaze marker. Episode 86 frames 60/120/180 are close and stable, with only
  mild secondary blobs on frames 60 and 180; frame 0 still locks to the tabletop/object area.
  Net: no checkerboard collapse, but epoch 7 still has not shown that the decoded final loss
  removes object-lock/scene-prior artifacts better than the pure temporal epoch 28 evidence.

- 2026-06-15 02:21 +08 check: mixed-loss epoch 6 validation completed and both watcher previews
  are available. Training remains healthy in epoch 7, about `842/5359` batches at roughly
  `1.41-1.46 it/s`; `logs.json.txt` reached `global_step=9589`. The new checkpoint preview is
  epoch 6 with `val_loss=0.012`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0006-val_loss=0.012.ckpt`
- Validation heatmap preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0006`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_val_preview\epoch_0006`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_021521_epoch=0006-val_loss=0.012`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_ckpt_preview\20260615_021521_epoch=0006-val_loss=0.012`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_021423_epoch=0006-val_loss=0.012_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_episode86\20260615_021423_epoch=0006-val_loss=0.012_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `531.585426568985s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 6 is mostly a plateau at
  `val_loss=0.012`. Sample 0 remains compact but slightly vertically stretched/right-biased near
  the table target, with faint lower-left speckles. Sample 1 is compact and near the target but
  still left/low. Sample 2 remains the clearest failure: the prediction is multi-lobed and follows
  the carton/scene texture instead of centering on the gaze marker, though it is less vertically
  stretched than epoch 5. Sample 3 stays usable and compact with a small above/right bias. Episode
  86 frames 60/120/180 remain close and stable; frame 0 still locks to the tabletop/object area.
  Net: no new checkerboard collapse, but the mixed decoded final loss still has not removed the
  hard object-lock/scene-prior artifact.

- 2026-06-15 01:11 +08 check: mixed-loss epoch 5 validation completed and both watcher previews
  are available. Training remains healthy in epoch 6, about `1346/5359` batches at roughly
  `1.49-1.51 it/s`; `logs.json.txt` reached `global_step=8375`. The new checkpoint preview is
  epoch 5 with `val_loss=0.012`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0005-val_loss=0.012.ckpt`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_011409_epoch=0005-val_loss=0.012`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_ckpt_preview\20260615_011409_epoch=0005-val_loss=0.012`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_011306_epoch=0005-val_loss=0.012_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_episode86\20260615_011306_epoch=0005-val_loss=0.012_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `548.1814699172974s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 5 lowers scalar loss again
  (`0.013 -> 0.012`) and keeps the easy frames usable, but still does not resolve the hard
  object-lock cases. Sample 0 is compact and close to the target with faint lower-left speckles.
  Sample 1 improves versus epoch 4 because its main mass moves nearer the target, though it remains
  slightly left/low. Sample 2 is still the strongest failure: a tall multi-lobe prediction follows
  carton/scene texture instead of collapsing onto the target. Sample 3 is compact and usable, with
  a mild above-target/right bias. Episode 86 frames 60/120/180 remain close and stable; frame 0
  still locks to the tabletop/object area. Net: mixed loss is numerically improving, but the
  decoded final distribution term has not yet beaten pure temporal epoch 28 on hard scene-prior
  artifacts.

- 2026-06-15 00:36 +08 check: mixed-loss epoch 4 validation completed and both watcher previews
  are available. Training remains healthy in epoch 5, about `2360/5359` batches at roughly
  `1.50-1.52 it/s`; `logs.json.txt` reached about `global_step=7289`. The new checkpoint preview
  is epoch 4 with `val_loss=0.013`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0004-val_loss=0.013.ckpt`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_001259_epoch=0004-val_loss=0.013`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_ckpt_preview\20260615_001259_epoch=0004-val_loss=0.013`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_001220_epoch=0004-val_loss=0.013_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_episode86\20260615_001220_epoch=0004-val_loss=0.013_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `518.236878156662s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 4 lowers the scalar loss
  (`0.016 -> 0.013`) but is not a dramatic visual breakthrough. Sample 0 remains compact and near
  target with slight right/up object/table bias and faint lower-left speckles. Sample 1 is compact
  but shifted left/below the target toward the cup/carton/object area. Sample 2 regresses relative
  to epoch 3: it is compact, but its main mass is below/right on carton/scene texture instead of
  target-centered. Sample 3 is usable and near target, with the main mode above/slightly right of
  the marker. In episode 86, frame 0 remains the strongest failure and still locks to the
  tabletop/object area; frames 60/120/180 are close and stable, though frame 180 remains a little
  right-biased. Overall, mixed loss keeps easy frames usable and improves the scalar metric, but
  epoch 4 still does not clearly beat pure temporal epoch 28 on the hard object-lock cases.

- 2026-06-14 23:26 +08 check: mixed-loss epoch 3 validation completed and both watcher previews
  are available. Training remains healthy in epoch 4, about `1452/5359` batches at roughly
  `1.48-1.51 it/s`; `logs.json.txt` reached `global_step=5728`. The new checkpoint preview is
  epoch 3 with `val_loss=0.016`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0003-val_loss=0.016.ckpt`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_231150_epoch=0003-val_loss=0.016`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0003_ckpt_preview\20260614_231150_epoch=0003-val_loss=0.016`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_231119_epoch=0003-val_loss=0.016_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0003_episode86\20260614_231119_epoch=0003-val_loss=0.016_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `533.656818151474s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 3 is a modest but real
  step up from epoch 2. Sample 0 is compact and near the target with only a mild object/table bias
  and much less stray texture. Sample 1 is tighter and still usable, though a small side bias
  remains. Sample 2 is still the hard case, but the prediction is more compact than epoch 2 and is
  better organized around the carton/scene geometry rather than exploding into a broad multi-lobe.
  Sample 3 is the cleanest sampled case, tight and near target. Episode 86 frames 60/120/180 are
  close to target and more stable than epoch 2; frame 0 still locks to the tabletop/object area.
  Compared with pure temporal epoch 28, mixed epoch 3 is clearly tightening on easier frames but
  still is not a clean overall win because the hardest frame remains object/scene-geometry biased.

- 2026-06-14 22:16 +08 check: mixed-loss epoch 2 validation completed and both watcher previews
  are available. Training remains healthy in epoch 3, about `766/5359` batches at roughly
  `1.37-1.41 it/s`; `logs.json.txt` reached `global_step=4210`. The new checkpoint preview is
  epoch 2 with `val_loss=0.018`, and the full episode 86 render completed successfully.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.018.ckpt`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_221041_epoch=0002-val_loss=0.018`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0002_ckpt_preview\20260614_221041_epoch=0002-val_loss=0.018`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_221300_epoch=0002-val_loss=0.018_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0002_episode86\20260614_221300_epoch=0002-val_loss=0.018_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `551.144s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 2 is a useful step up from
  epoch 1 and is now approaching the pure temporal baseline on several stills, but it is not yet
  clearly better overall. Sample 0 becomes compact and near the target with a mild object/table
  bias; sample 1 is more compact but still carries a small side speck/edge bias; sample 2 remains
  the hardest case, still leaning below/right toward scene geometry and carton texture instead of a
  clean gaze-centered peak; sample 3 is tighter than epoch 1 but still a little broad. Episode 86
  frames 60/120/180 are close to target and more usable than epoch 1, but frame 0 still locks to
  the tabletop/object area instead of the left-up gaze target. Pure temporal epoch 28 still has the
  cleaner compactness on the hardest frame, so the mixed objective is promising but not yet the
  winner.

- 2026-06-14 21:41 +08 check: mixed-loss epoch 1 validation completed and both watcher previews
  are available. Training remains healthy in epoch 2, about `2992/5359` batches at roughly
  `1.49-1.52 it/s`; `logs.json.txt` reached `global_step=3427`. The new checkpoint preview is
  epoch 1 with `val_loss=0.024`.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.024.ckpt`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_211230_epoch=0001-val_loss=0.024`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_ckpt_preview\20260614_211230_epoch=0001-val_loss=0.024`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_211141_epoch=0001-val_loss=0.024_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_episode86\20260614_211141_epoch=0001-val_loss=0.024_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `550.66s`.
- Visual read against the pure temporal-window epoch 28 evidence: epoch 1 is tighter than epoch 0
  and the broad blob has shrunk, but it is still not as clean as pure temporal epoch 28. Sample 0
  is now a compact two-lobe prediction around the table/object area; sample 1 remains object-biased
  around the cup/carton scene; sample 2 is still the hard failure with a strong lower blob and side
  lobe tied to carton/edge/scene geometry; sample 3 is somewhat tighter but still broad. Episode 86
  frames 0/60/120/180 likewise get a bit tighter than epoch 0 but still trail the pure temporal
  epoch 28 stills in compactness and object-lock suppression.

- 2026-06-14 21:00 +08 check: mixed-loss epoch 0 validation completed and both watcher previews
  are available. Training remains healthy in epoch 1, about `4926/5359` batches at roughly
  `1.49-1.51 it/s`; `logs.json.txt` reached `global_step=2570`. Recent train rows still match the
  mixed objective with latent diffusion heatmap loss plus nonzero decoded `XY`/`JS` terms.
- Saved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.031.ckpt`
- Checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_201120_epoch=0000-val_loss=0.031`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_ckpt_preview\20260614_201120_epoch=0000-val_loss=0.031`
- Full episode 86 preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_201028_epoch=0000-val_loss=0.031_episode86`
  - local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_episode86\20260614_201028_epoch=0000-val_loss=0.031_episode86\comparison.mp4`
- Summary fields: `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `diffusion_final_heatmap_loss=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional/window_radius=30/beta=10.0/sigma_px=6.0/current_weight=2.0`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, and selected sample
  indices `6600,1341,9843,11637`. Episode 86 rendered `3599/3599` frames at 15 FPS in
  `545.52s`.
- Visual read against the pure temporal-window epoch 28 evidence: mixed-loss epoch 0 is a valid
  early preview but is not visually better yet. Sample 0 is broad and multi-lobed around the
  table/object area; sample 1 has multiple peaks around the cup/carton/object region; sample 2 is
  still a hard failure that follows the carton/edge/scene geometry instead of the compact target;
  sample 3 is closer but remains broad and partly offset. Episode 86 frames 0/60/120/180 show the
  same pattern: target heatmaps are compact temporal labels, while predictions are still larger,
  multi-peak, and object-biased. By contrast, the pure temporal epoch 28 episode stills are much
  tighter on these same frames. Treat this as very early mixed-loss evidence; the important next
  decision point is whether epoch 1+ reduces the broad object-locking behavior.

- 2026-06-14 19:54 +08 check: mixed-loss run remains healthy in epoch 0, about
  `4157/5359` batches at roughly `1.47-1.50 it/s`; `logs.json.txt` reached `global_step=1038`.
  Training is still using the mixed objective and all tmux sessions are alive. The checkpoint
  directory has still not been created, so the first mixed-loss checkpoint preview and episode 86
  video are not available yet. The recurring heartbeat automation was updated from the old pure
  temporal-window run path to this active mixed-loss run path so future checks follow the current
  experiment.

- 2026-06-14 19:43 +08 check: mixed-loss run is still healthy in epoch 0, about
  `3128/5359` batches at roughly `1.50-1.53 it/s`; `logs.json.txt` reached `global_step=781`.
  Training continues to use all eight GPUs. The checkpoint directory still does not exist, and the
  checkpoint/episode watcher sessions are alive and polling, so no mixed checkpoint preview or
  full-episode video is available yet.

- 2026-06-14 19:38 +08 check: mixed-loss run remains healthy in epoch 0, about
  `2726/5359` batches at roughly `1.50-1.53 it/s`; `logs.json.txt` reached `global_step=680`.
  The checkpoint directory still has not been created, so the first mixed-loss checkpoint preview
  and full-episode preview are still unavailable. Training tmux plus checkpoint and episode watcher
  tmux sessions remain alive and polling.

- 2026-06-14 19:35 +08 check: mixed-loss run remains healthy in epoch 0, about
  `2398/5359` batches at roughly `1.50-1.53 it/s`; `logs.json.txt` reached `global_step=599`.
  Recent log rows continue to match the active mixed objective, with nonzero decoded `XY` and `JS`
  terms added to the latent diffusion heatmap loss. The checkpoint directory still has not been
  created, so no checkpoint preview or full-episode preview is available yet. Both watcher sessions
  remain alive and polling.

- 2026-06-14 19:32 +08 check: mixed-loss run is still healthy in epoch 0, about
  `2120/5359` batches at roughly `1.50-1.51 it/s`; `logs.json.txt` reached `global_step=527`.
  The latest train rows remain consistent with the mixed objective (`latent diffusion heatmap loss`
  plus weighted decoded `XY` and `JS` losses). The checkpoint directory has still not been created,
  so there are no mixed-loss checkpoint or full-episode preview artifacts yet. The checkpoint and
  episode watcher tmux sessions are alive and polling.

- 2026-06-14 19:28 +08 check: mixed-loss run continues normally in epoch 0, about
  `1772/5359` batches at roughly `1.49-1.52 it/s`; `logs.json.txt` reached `global_step=442`.
  Latest rows still have nonzero decoded `XY`/`JS` loss terms on top of latent diffusion MSE.
  The checkpoint directory still does not exist, so there is no mixed checkpoint preview or
  full-episode preview to pull yet. Both watcher sessions remain alive and are polling for the
  first checkpoint.

- 2026-06-14 19:23 +08 check: mixed-loss run is still healthy in epoch 0, about
  `1382/5359` batches at roughly `1.49-1.52 it/s`; `logs.json.txt` reached `global_step=343`.
  Recent rows continue to show nonzero decoded `XY`/`JS` losses contributing to `train_loss`.
  There is still no checkpoint directory, so epoch 0 validation and watcher preview generation have
  not happened yet. Training tmux, checkpoint watcher tmux, and episode watcher tmux are all alive;
  all eight GPUs are occupied by the run.

- 2026-06-14 19:20 +08 check: mixed-loss run is healthy and still in epoch 0, about
  `1098/5359` batches at roughly `1.48-1.50 it/s`; `logs.json.txt` reached `global_step=274`.
  Recent rows confirm the decoded final heatmap terms are active, for example
  `train_loss=0.1114936601370573`, `train_heatmap_loss=0.09101053886115551`,
  `train_heatmap_xy_loss=0.04693415854126215`, and
  `train_heatmap_js_loss=0.18136414512991905`, matching the weighted mixed objective. The
  checkpoint directory does not exist yet because epoch 0 validation has not completed. Training
  tmux plus checkpoint and episode watcher tmux sessions are all alive; no mixed checkpoint or
  full-episode preview exists yet.

- 2026-06-14 19:13 +08 check: switched from the pure temporal-window latent-MSE run to the new
  temporal-window mixed-loss ablation. The previous pure-MSE run was stopped after epoch 28
  checkpoint and watcher previews were complete. The new mixed run is live:
  - training tmux: `gaze_wam_open_cosmos_temporal_mixed_loss_8gpu`
  - checkpoint watcher tmux: `gaze_wam_temporal_mixed_loss_ckpt_preview_watch`
  - episode watcher tmux: `gaze_wam_temporal_mixed_loss_episode_preview_watch`
  - output:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_`
- Mixed run contract is verified in `training_contract.json`:
  `diffusion_final_heatmap_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_supervision=latent_diffusion_mse_plus_decoded_final_heatmap_loss`,
  `heatmap_xy_loss_weight=0.05`, and `heatmap_js_loss_weight=0.1`.
- Startup health: training reached epoch 0 around `436/5359` with `global_step=108`. Recent
  log rows show the mixed objective is active, e.g.
  `train_loss=0.5017528533935547`, `train_heatmap_loss=0.4723067507147789`,
  `train_heatmap_xy_loss=0.07683922629803419`, and
  `train_heatmap_js_loss=0.25604138895869255`. This matches
  `heatmap_loss + 0.05 * xy + 0.10 * js`.

- 2026-06-14 18:52 +08 check: epoch 28 validation completed and the active run continued into
  epoch 29, about `19%` (`1004/5359`) with `logs.json.txt` reaching `global_step=39110`.
  Training tmux and both watcher tmux sessions remain alive. The epoch 28 validation row reports
  `global_step=38859`, `val_loss=0.005265417438931763`,
  `val_heatmap_loss=0.005265417438931763`, `val_heatmap_preview_saved=1`, and the same
  non-mixed objective (`train/val heatmap_xy/js/token_kl_loss=0`).
  - checkpoint preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_184212_latest`
  - checkpoint preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_184212_ckpt_preview\20260614_184212_latest`
  - episode preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_184351_latest_episode86`
  - episode preview local video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_184351_episode86\20260614_184351_latest_episode86\comparison.mp4`
- Epoch 28/latest visual read: sample 1 remains usable; sample 0 is close but still has a
  secondary blob near the plate/object region; sample 2 remains the hard failure with a multi-peak
  prediction tied to scene texture/light/box geometry; sample 3 is usable with mild offset. Full
  episode 86 still fails frame 0 by locking to the tabletop/object region, while frames
  60/120/180 align well enough for qualitative use. Net: epoch 28 is not better than epoch 27
  numerically and does not remove the scene-texture failure mode.
- Mixed-loss implementation has been prepared locally for the next ablation. The new path keeps
  `heatmap_objective=diffusion` and the latent noise/target MSE, then optionally decodes the
  reconstructed clean heatmap latent and adds final `XY`/`JS` heatmap losses when
  `policy.heatmap_diffusion_final_loss_enabled=true`. The proposed config is
  `diffusion_policy/config/train_gaze_wam_open_only_cosmos_temporal_mixed_loss_workspace.yaml`
  with temporal-window labels unchanged, `heatmap_xy_loss_weight=0.05`,
  `heatmap_js_loss_weight=0.10`, and
  `heatmap_final_loss_timestep_weighting=alpha_cumprod` so final decoded supervision is strongest
  at low-noise denoising steps.

- 2026-06-14 17:55 +08 check: epoch 27 validation completed and the active run continued into
  epoch 28. Training tmux and both watcher tmux sessions remain alive. The epoch 27 validation row
  reports `global_step=37519`, `val_loss=0.004825264634564519`,
  `val_heatmap_preview_saved=1`; a follow-up tail found epoch 28 at `global_step=37576`.
  - checkpoint preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_174118_latest`
  - checkpoint preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174118_ckpt_preview\20260614_174118_latest`
  - episode preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_174245_latest_episode86`
  - episode preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174245_episode86\20260614_174245_latest_episode86\comparison.mp4`
- Epoch 27/latest checkpoint summary keeps the same non-mixed contract:
  `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 render completed after a short delay: the server directory now
  contains `comparison.mp4`, `preview.log`, and `summary.json`, and the local copy has been
  refreshed accordingly.
- Visual read for epoch 27/latest: sample 1 and episode frames 60/120/180 are the best-looking
  part of this run and are usable, with only small offsets. Sample 0 regresses relative to epoch 26
  by showing a stronger off-target blob above/left of the gaze target. Sample 2 remains the same
  hard failure with multiple peaks following scene texture around the light/box. Sample 3 keeps a
  small right/up bias. Full episode frame 0 still locks to the tabletop/object region. Net: epoch
  27 is visually mixed, not a monotonic improvement. The temporal-window clean label remains
  useful, but pure latent diffusion/noise MSE still needs an additional decoded final-distribution
  loss.

- 2026-06-14 17:05 +08 check: epoch 26 validation completed and the active run continued into
  epoch 27. Training tmux and both watcher tmux sessions are alive. `logs.json.txt` reached
  `global_step=36650` in epoch 27 after the epoch 26 validation row
  (`global_step=36179`, `val_loss=0.004669780842959881`).
  - checkpoint preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_164022_latest`
  - checkpoint preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164022_ckpt_preview\20260614_164022_latest`
  - episode preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_164137_latest_episode86`
  - episode preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164137_episode86\20260614_164137_latest_episode86\comparison.mp4`
- Epoch 26/latest summary fields remain the intended non-mixed contract:
  `source=open`, `split=val`, `use_ema=false`, `checkpoint_has_ema=false`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `553.47s`.
- Visual read for epoch 26/latest: samples 0, 1, and 3 are still usable, but keep small spatial
  bias and the recurring lower-left speckles. Sample 2 is still the hard failure with multiple
  peaks around the light/box scene texture instead of a single target-centered blob. Full episode
  86 still has a bad first frame that locks onto the tabletop/object area, while frames
  60/120/180 are target-following and visually usable with minor offset/speckle. Net: epoch 26
  is slightly better numerically than epoch 25, but visually it is not a material breakthrough.
  The temporal-window label remains suitable as a DiT final clean label; the model objective still
  needs a real mixed-loss ablation to suppress decoded distribution artifacts.

- 2026-06-14 15:55 +08 check: epoch 25 checkpoint validation finished and a new checkpoint
  preview `20260614_154128_latest` plus a new full episode 86 preview
  `20260614_154057_latest_episode86` are now available. Training has advanced into epoch 26 and
  `logs.json.txt` reached `global_step=34839`. Training tmux and both watcher tmux sessions are
  alive.
  - checkpoint preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_154128_latest`
  - checkpoint preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154128_ckpt_preview\20260614_154128_latest`
  - episode preview server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_154057_latest_episode86`
  - episode preview local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154057_episode86\20260614_154057_latest_episode86\comparison.mp4`
- Summary fields confirm the same active contract:
  `source=open`, `split=val`, `use_ema=false`, `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  selected indices `6600,1341,9843,11637`. Epoch 25 validation row reports
  `val_loss=0.005157463566865772` at `global_step=34839`. Episode 86 rendered
  all `3599/3599` frames at 15 FPS in about `527.58s`.
- Visual read for epoch 25/latest: sample 0, 1, and 3 remain usable but still keep small offsets
  and lower-left speckles; sample 2 remains the hard failure and still locks onto the light/box
  texture as a multi-peak blob rather than a clean target-centered hotspot. Full episode 86 is
  also similar to epoch 24: frames 60/120/180 track the target reasonably well, while frame 0
  still locks onto the tabletop/object region. Net: the run is healthy but still not monotonic,
  and the new preview does not materially beat the epoch 24/latest result.

- 2026-06-14 14:55 +08 check: the latest checkpoint preview and full episode preview are now
  available after epoch 24. Training remains healthy in epoch 25 and `logs.json.txt` reached
  `global_step=33941`. Training tmux, checkpoint watcher, and episode watcher are all alive.
  Pulled checkpoint preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_144021_latest`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144021_ckpt_preview\20260614_144021_latest`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_144022_latest_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144022_episode86\20260614_144022_latest_episode86\comparison.mp4`
- Summary fields confirm the intended non-mixed objective:
  `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `511.54s`.
- Visual read for the latest checkpoint: sample 0, 1, and 3 are usable with only mild bias;
  sample 2 is still the hard case and remains multi-peak, with attraction to scene texture around
  the light/box region. The episode stills are more reassuring: frames 60/120/180 stay mostly on
  the temporal target with small offsets, but frame 0 still locks to the tabletop/object region.
  Net: the temporal-window label remains a good DiT final target, but pure frozen-Cosmos latent
  diffusion/noise MSE is still not a mixed-loss setup and remains non-monotonic on hard cases.

- 2026-06-14 13:52 +08 check: epoch 23 checkpoint and both watcher previews are now
  available. Training remains healthy and has continued into epoch 24; `logs.json.txt` reached
  `global_step=32366`. Training tmux, checkpoint watcher, and episode watcher are all alive.
- Epoch 23 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0023-val_loss=0.005.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_133927_epoch=0023-val_loss=0.005`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0023_ckpt_preview\20260614_133927_epoch=0023-val_loss=0.005`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_133929_epoch=0023-val_loss=0.005_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0023_episode86\20260614_133929_epoch=0023-val_loss=0.005_episode86\comparison.mp4`
- Epoch 23 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `541.66s`.
- Visual read for epoch 23: this checkpoint is slightly worse than epoch 22. Sample 0 keeps the
  mild right/up bias and lower-left speckles. Sample 1 is still usable but not clearly improved.
  Sample 2 remains the hard failure and is even more multi-peak, with stronger attraction to the
  light/box scene texture rather than the target-centered blob. Sample 3 remains usable with small
  bias. The episode stills stay similar to epoch 22: frames 60/120/180 are decent, but frame 0 still
  locks to the tabletop/object area. Net: temporal-window labels remain useful as a DiT final clean
  label, but pure frozen-Cosmos latent diffusion/noise MSE still carries scene-prior failures and
  has not stabilized into a clear monotonic improvement.

- 2026-06-14 13:37 +08 check: no new checkpoint or watcher preview beyond epoch 22 yet.
  Training remains healthy in epoch 23 at about `74%` (`~3992/5359`) and `logs.json.txt`
  reached `global_step=32136`. Checkpoints and both watcher preview directories still show
  epoch 22 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 13:21 +08 check: no new checkpoint or watcher preview beyond epoch 22 yet.
  Training remains healthy in epoch 23 at about `74%` (`~3992/5359`) and `logs.json.txt`
  reached `global_step=31817`. Checkpoints and both watcher preview directories still show
  epoch 22 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 13:06 +08 check: no new checkpoint or watcher preview beyond epoch 22 yet.
  Training remains healthy in epoch 23 at about `49%` (`~2636/5359`) and `logs.json.txt`
  reached `global_step=31478`. Checkpoints and both watcher preview directories still show
  epoch 22 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 12:45 +08 check: epoch 22 checkpoint and both watcher previews are now
  available. Training remains healthy and has continued into epoch 23; `logs.json.txt` reached
  `global_step=31006`. Training tmux, checkpoint watcher, and episode watcher are all alive.
- Epoch 22 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0022-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_123833_epoch=0022-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0022_ckpt_preview\20260614_123833_epoch=0022-val_loss=0.004`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_123831_epoch=0022-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0022_episode86\20260614_123831_epoch=0022-val_loss=0.004_episode86\comparison.mp4`
- Epoch 22 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `540.16s`.
- Visual read for epoch 22: this checkpoint is slightly better than epoch 21, but it still is not a
  clean step-change. Sample 0 now lands near the target with a mild right/up bias and keeps the
  lower-left speckles. Sample 1 is usable and tracks the cup/object region more cleanly. Sample 2
  remains the hard failure: the decoded heatmap still forms multiple peaks and is pulled by the
  bright light/box scene texture instead of the target-centered blob. Sample 3 is also usable with a
  small bias. The full episode video is decent on frames 60/120/180, but frame 0 still locks to the
  tabletop/object area. Net: temporal-window labels are usable as a DiT final clean label, but pure
  frozen-Cosmos latent diffusion/noise MSE still leaves a scene-prior failure mode.
- 2026-06-14 11:45 +08 check: epoch 21 checkpoint and both watcher previews are now
  available. Training remains healthy and has continued into epoch 22; `logs.json.txt` reached
  `global_step=29671`. Training tmux, checkpoint watcher, and episode watcher are all alive.
- Epoch 21 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0021-val_loss=0.005.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_113738_epoch=0021-val_loss=0.005`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0021_ckpt_preview\20260614_113738_epoch=0021-val_loss=0.005`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_113737_epoch=0021-val_loss=0.005_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0021_episode86\20260614_113737_epoch=0021-val_loss=0.005_episode86\comparison.mp4`
- Epoch 21 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `540.16s`.
- Visual read for epoch 21: this checkpoint is not an improvement over epoch 20. Samples 0, 1, and
  3 remain usable but keep the same mild offsets and lower-left speckles. Sample 2 is worse than
  the already-hard epoch 20 case: the decoded heatmap has stronger multi-peak mass along the upper
  light/box region plus a downward trailing mode, while target-centered mass is weak. Episode 86
  remains usable in frames 60/120/180 but frame 0 still locks to the tabletop/object area. The run
  remains plateaued.

- 2026-06-14 11:32 +08 check: no new checkpoint or watcher preview beyond epoch 20 yet.
  Training remains healthy in epoch 21 at about `94%` (`~5032/5359`) and `logs.json.txt`
  reached `global_step=29397`. Checkpoints and both watcher preview directories still show
  epoch 20 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 11:23 +08 check: no new checkpoint or watcher preview beyond epoch 20 yet.
  Training remains healthy in epoch 21 at about `78%` (`~4174/5359`) and `logs.json.txt`
  reached `global_step=29184`. Checkpoints and both watcher preview directories still show
  epoch 20 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 11:14 +08 check: no new checkpoint or watcher preview beyond epoch 20 yet.
  Training remains healthy in epoch 21 at about `63%` (`~3384/5359`) and `logs.json.txt`
  reached `global_step=28984`. Checkpoints and both watcher preview directories still show
  epoch 20 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 10:53 +08 check: epoch 20 checkpoint and both watcher previews are now
  available. Training remains healthy and has continued into epoch 21; `logs.json.txt` reached
  `global_step=28517`. Training tmux, checkpoint watcher, and episode watcher are all alive.
- Epoch 20 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0020-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_103643_epoch=0020-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0020_ckpt_preview\20260614_103643_epoch=0020-val_loss=0.004`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_103635_epoch=0020-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0020_episode86\20260614_103635_epoch=0020-val_loss=0.004_episode86\comparison.mp4`
- Epoch 20 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `548.67s`.
- Visual read for epoch 20: still no meaningful improvement over epochs 17-19. Sample 0 remains
  close to target but slightly right/up with the fixed lower-left speckles. Sample 1 is usable but
  left of the gaze marker toward the cup/object region. Sample 2 is still the clearest hard
  failure, with multi-peak mass attaching to bright box/light scene texture instead of the
  temporal target. Sample 3 remains usable but above/right biased with a weak trailing mode. In
  full episode 86, frames 60/120/180 remain near-target and usable, while frame 0 remains a
  tabletop/object lock. The plateau conclusion remains unchanged.

- 2026-06-14 10:35 +08 check: no new checkpoint or watcher preview beyond epoch 19 yet.
  Training remains healthy in epoch 20 and `logs.json.txt` reached `global_step=28135`.
  Checkpoints and both watcher preview directories still show epoch 19 as the latest completed
  visual evidence. Training tmux, checkpoint watcher, and episode watcher are all alive.

- 2026-06-14 10:30 +08 check: no new checkpoint or watcher preview beyond epoch 19 yet.
  Training remains healthy in epoch 20 and `logs.json.txt` reached `global_step=28020`.
  Checkpoints and both watcher preview directories still show epoch 19 as the latest completed
  visual evidence. Training tmux, checkpoint watcher, and episode watcher are all alive.

- 2026-06-14 10:14 +08 check: no new checkpoint or watcher preview beyond epoch 19 yet.
  Training remains healthy in epoch 20 at about `~27750/5359` and continuing; `logs.json.txt`
  reached `global_step=27750`. Checkpoints and both watcher preview directories still show
  epoch 19 as the latest completed visual evidence. Training tmux, checkpoint watcher, and episode
  watcher are all alive.

- 2026-06-14 09:49 +08 check: epoch 19 checkpoint and both watcher previews are now available.
  Training remains healthy and has continued into epoch 20; `logs.json.txt` reached
  `global_step=26840`. Training tmux, checkpoint watcher, and episode watcher are all alive.
- Epoch 19 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0019-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_093549_epoch=0019-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_093835_epoch=0019-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_episode86\comparison.mp4`
- Epoch 19 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `545.09s`.
- Visual read for epoch 19: no meaningful improvement over epochs 17-18. Sample 0 stays near the
  target but remains slightly right/up with the fixed lower-left speckles. Sample 1 is usable but
  left-biased toward the cup/object region. Sample 2 is still the clearest failure, with multiple
  off-target modes attached to box/highlight scene texture rather than the temporal target. Sample
  3 remains usable but above/right biased with a horizontal trail. In full episode 86, frames
  60/120/180 remain near-target and usable, while frame 0 still locks to the tabletop/object region
  instead of the left-up target. The run therefore remains visually plateaued: temporal-window
  labels are good DiT final-label candidates, but pure latent diffusion/noise MSE still leaves
  scene-prior attraction and speckles.

- 2026-06-14 09:18 +08 check: no new checkpoint or watcher preview beyond epoch 18 yet.
  Training remains healthy in epoch 19 at about `65%+` and continuing; `logs.json.txt` reached
  `global_step=26426`. Checkpoints and both watcher preview directories still show epoch 18 as the
  latest completed visual evidence. Training tmux, checkpoint watcher, and episode watcher are all
  alive.

- 2026-06-14 09:13 +08 check: no new checkpoint or watcher preview beyond epoch 18 yet.
  Training remains healthy in epoch 19 at about `65%` (`~3508/5359`) and roughly
  `1.49-1.50 it/s`; `logs.json.txt` reached `global_step=26317`. Checkpoints and both watcher
  preview directories still show epoch 18 as the latest completed visual evidence. Training tmux,
  checkpoint watcher, and episode watcher are all alive.

- 2026-06-14 09:08 +08 check: no new checkpoint or watcher preview beyond epoch 18 yet.
  Training remains healthy in epoch 19 at about `57%` (`~3036/5359`) and roughly
  `1.49-1.51 it/s`; `logs.json.txt` reached `global_step=26208`. Checkpoints and both watcher
  preview directories still show epoch 18 as the latest completed visual evidence. Training tmux,
  checkpoint watcher, and episode watcher are all alive.

- 2026-06-14 09:01 +08 check: no new checkpoint or watcher preview beyond epoch 18 yet.
  Training remains healthy in epoch 19 at about `48%` (`~2562/5359`) and roughly
  `1.47-1.49 it/s`; `logs.json.txt` reached `global_step=26045`. Checkpoints and both watcher
  preview directories still show epoch 18 as the latest completed visual evidence. Training tmux,
  checkpoint watcher, and episode watcher are all alive.

- 2026-06-14 08:45 +08 check: epoch 18 checkpoint and both watcher previews are now available.
  Training remains healthy and has continued into epoch 19 at about `19%` (`~1024/5359`) with
  roughly `1.38-1.44 it/s`; `logs.json.txt` reached `global_step=25706`. Training tmux,
  checkpoint watcher, and episode watcher are all alive.
- Epoch 18 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0018-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_083455_epoch=0018-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_083728_epoch=0018-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_episode86\comparison.mp4`
- Epoch 18 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`.
- Visual read for epoch 18: no clear improvement over epoch 17. Sample 0 is near-target but biased
  slightly right/up and still has the persistent lower-left speckles. Sample 1 is usable but shifted
  left of the target, with the strongest mass around the cup/object region. Sample 2 is still the
  hard failure: multi-peak output with the strongest mode on box/highlight scene texture instead of
  the temporal target. Sample 3 is usable but above/right biased with a horizontal trailing mode.
  In full episode 86, frame 0 remains a tabletop/object lock, while frames 60, 120, and 180 remain
  generally near-target and usable. This keeps the same conclusion: the temporal-window clean label
  is a good DiT final-label candidate, but pure latent diffusion/noise MSE still leaves scene-prior
  attraction, multi-peak hard cases, and lower-left speckles.

- 2026-06-14 08:31 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `96%` (`~5132/5359`) and roughly
  `1.50-1.52 it/s`; `logs.json.txt` reached `global_step=25408`. Checkpoint watcher and episode
  watcher are alive and still show epoch 17 as the latest completed preview. Latest fully verified
  visual evidence remains epoch 17.

- 2026-06-14 08:27 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `90%` (`~4814/5359`) and roughly
  `1.49-1.52 it/s`; `logs.json.txt` reached `global_step=25322`. Checkpoint watcher and episode
  watcher are alive and still show epoch 17 as the latest completed preview. Latest fully verified
  visual evidence remains epoch 17.

- 2026-06-14 08:24 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `85%` (`~4532/5359`) and roughly
  `1.46-1.51 it/s`; `logs.json.txt` reached `global_step=25252`. Checkpoint watcher and episode
  watcher are alive and still show epoch 17 as the latest completed preview. Latest fully verified
  visual evidence remains epoch 17.

- 2026-06-14 08:21 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `79%` (`~4228/5359`) and roughly
  `1.49-1.52 it/s`; `logs.json.txt` reached `global_step=25175`. Checkpoint watcher and episode
  watcher are alive and still show epoch 17 as the latest completed preview. Latest fully verified
  visual evidence remains epoch 17.

- 2026-06-14 08:17 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `73%` (`~3902/5359`) and roughly
  `1.49-1.52 it/s`; `logs.json.txt` reached `global_step=25094`. Checkpoint and episode watcher
  logs still show the epoch 17 preview as the latest completed item. Latest fully verified visual
  evidence remains epoch 17. One tmux-session status SSH command timed out, but subsequent
  training-pane, directory-listing, and watcher-log checks succeeded.

- 2026-06-14 08:13 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `66%` (`~3512/5359`) and roughly
  `1.51-1.55 it/s`; `logs.json.txt` reached `global_step=24996`. Checkpoint watcher and episode
  watcher are alive. Latest fully verified visual evidence remains epoch 17.

- 2026-06-14 08:09 +08 check: no new checkpoint or watcher preview beyond epoch 17 yet.
  Training remains healthy in epoch 18 at about `59%` (`~3146/5359`) and roughly
  `1.50-1.53 it/s`; `logs.json.txt` reached `global_step=24906`. Checkpoint watcher and episode
  watcher are alive. Latest fully verified visual evidence remains epoch 17.

- 2026-06-14 07:51 +08 check: epoch 17 is the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 18 at about `28%` (`~1514/5359`) and roughly `1.49-1.51 it/s`; `logs.json.txt`
  reached `global_step=24497`. Checkpoint watcher and episode watcher are alive.
- Epoch 17 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0017-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_073600_epoch=0017-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_073641_epoch=0017-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_episode86\comparison.mp4`
- Epoch 17 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `535.28s`.
- Visual read for epoch 17: still no meaningful improvement over epoch 16. Sample 0 is near-target
  but splits vertically and keeps the fixed lower-left speckles. Sample 1 shifts left toward the
  cup/hand region and is less centered on the target. Sample 2 remains the hard failure with
  multi-peak scene/box/light texture lock and weak target mass. Sample 3 remains usable but
  above/right biased. In full episode 86, frames 60/120/180 remain near-target and usable, while
  frame 0 still locks to the tabletop/object region instead of the left-up gaze target. This
  strengthens the plateau read: temporal-window labels are useful as clean labels, but pure latent
  diffusion/noise MSE is not removing scene-prior attraction or speckles.

- 2026-06-14 07:29 +08 check: no new checkpoint or watcher preview beyond epoch 16 yet.
  Training remains healthy in epoch 17 at about `92%` (`~4950/5359`) and roughly `1.48-1.50 it/s`;
  `logs.json.txt` reached `global_step=24016` with recent `train_heatmap_loss=0.00350`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 16.

- 2026-06-14 07:26 +08 check: no new checkpoint or watcher preview beyond epoch 16 yet.
  Training remains healthy in epoch 17 at about `88%` (`~4720/5359`) and roughly `1.48-1.51 it/s`;
  `logs.json.txt` reached `global_step=23958` with recent `train_heatmap_loss=0.00361`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 16.

- 2026-06-14 07:22 +08 check: no new checkpoint or watcher preview beyond epoch 16 yet.
  Training remains healthy in epoch 17 at about `81%` (`~4336/5359`) and roughly `1.48-1.50 it/s`;
  `logs.json.txt` reached `global_step=23862` with recent `train_heatmap_loss=0.00297`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 16.

- 2026-06-14 07:03 +08 check: no new checkpoint or watcher preview beyond epoch 16 yet.
  Training remains healthy in epoch 17 at about `50%` (`~2684/5359`) and roughly `1.46-1.51 it/s`;
  `logs.json.txt` reached `global_step=23450` with recent `train_heatmap_loss=0.00340`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 16.

- 2026-06-14 07:00 +08 check: no new checkpoint or watcher preview beyond epoch 16 yet.
  Training remains healthy in epoch 17 at about `44%` (`~2354/5359`) and roughly `1.49-1.51 it/s`;
  `logs.json.txt` reached `global_step=23367` with recent `train_heatmap_loss=0.00322`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 16.

- 2026-06-14 06:48 +08 check: epoch 16 is the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 17 at about `23%` (`~1248/5359`) and roughly `1.48-1.50 it/s`; checkpoint watcher
  and episode watcher are alive.
- Epoch 16 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0016-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_063509_epoch=0016-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_063620_epoch=0016-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_episode86\comparison.mp4`
- Epoch 16 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `506.93s`.
- Visual read for epoch 16: no meaningful improvement over epoch 15. Samples 0, 1, and 3 remain
  near-target and usable with mild offsets and lower-left speckles. Sample 2 is worse than the
  already-failing epoch 15 view: it becomes a multi-peak/vertical scene-texture lock around the
  box/light region rather than centering on the gaze target. In full episode 86, frames 60/120/180
  remain near-target and usable; frame 0 remains the stable tabletop/object failure. This continues
  to support temporal-window labels as clean labels, while showing that pure latent MSE alone does
  not remove scene-prior attraction.

- 2026-06-14 06:19 +08 check: no new checkpoint or preview beyond epoch 15 yet. Training remains
  healthy in epoch 16 at about `78%` (`~4176/5359`) and roughly `1.49-1.50 it/s`;
  `logs.json.txt` reached `global_step=22483` with recent `train_heatmap_loss=0.00375`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 15.

- 2026-06-14 05:50 +08 check: epoch 15 is the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 16 at about `28%` (`~1528/5359`) and roughly `1.51 it/s`; checkpoint watcher and
  episode watcher are alive.
- Epoch 15 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0015-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_053413_epoch=0015-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_053508_epoch=0015-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_episode86\comparison.mp4`
- Epoch 15 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `558.58s`.
- Visual read for epoch 15: visually close to epoch 14, without a decisive improvement. Samples 0,
  1, and 3 remain usable with their main decoded mass near the temporal target, but they keep mild
  local bias and the persistent lower-left speckle cluster. Sample 2 remains the main hard failure:
  the strongest decoded mass attaches to the box/light and nearby scene texture instead of the
  gaze-centered target. In full episode 86, frames 60/120/180 remain near-target and usable; frame 0
  still locks to the tabletop/object area. Conclusion remains that temporal-window labels are useful
  as DiT clean labels, while pure latent diffusion/noise MSE still leaves scene-prior attraction.

- 2026-06-14 05:14 +08 check: no new checkpoint or preview beyond epoch 14 yet. Training remains
  healthy in epoch 15 at about `69%` (`~3704/5359`) and roughly `1.46-1.50 it/s`;
  `logs.json.txt` reached `global_step=21025` with recent `train_heatmap_loss=0.00387`.
  Checkpoint watcher and episode watcher are alive. Latest fully verified visual evidence remains
  epoch 14.

- 2026-06-14 05:05 +08 check: no new checkpoint or preview beyond epoch 14 yet. Training remains
  healthy in epoch 15 at about `55%` (`~2924/5359`) and roughly `1.50 it/s`; `logs.json.txt`
  reached `global_step=20829` with recent `train_heatmap_loss=0.00329`. Checkpoint watcher and
  episode watcher are alive. Latest fully verified visual evidence remains epoch 14.

- 2026-06-14 04:54 +08 check: epoch 14 is the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 15 at about `36%` (`~1956/5359`) and roughly `1.50 it/s`; `logs.json.txt`
  reached `global_step=20588` with `epoch=15`. Checkpoint watcher and episode watcher are alive.
- Epoch 14 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0014-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_043317_epoch=0014-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_043401_epoch=0014-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_episode86\comparison.mp4`
- Epoch 14 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `553.67s`.
- Visual read for epoch 14: sample 0, sample 1, and sample 3 are usable and put their main decoded
  mass close to the gaze/temporal target, but still keep small local biases and the persistent
  lower-left speckle cluster. Sample 2 remains the hard failure: its strongest modes follow the
  bright box/light/scene-texture region above the gaze marker instead of making the target the main
  mode. In full episode 86, frames 60/120/180 remain near-target and visually usable; frame 0 still
  locks to the tabletop/object area instead of the left-up gaze target. Conclusion remains:
  temporal-window labels are a good DiT clean-label candidate, but pure latent diffusion/noise MSE
  still leaves scene-prior attraction and speckle artifacts.

- 2026-06-14 04:09 +08 check: epoch 13 is the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 14 at about `41%` (`~2212/5359`) and roughly `1.5 it/s`; checkpoint watcher and
  episode watcher are alive.
- Epoch 13 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0013-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_033223_epoch=0013-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_033306_epoch=0013-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_episode86\comparison.mp4`
- Epoch 13 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `552.01s`.
- Visual read for epoch 13: visually close to epoch 12 and not a decisive improvement. Sample 0
  remains near-target but right/up biased with lower-left speckles and a faint upper residual.
  Sample 1 stays usable but left-biased. Sample 2 remains the hard failure: the decoded heatmap is
  multi-peak around the box/highlight region and the target is not the main mode. Sample 3 improves
  slightly versus epoch 12 because the right-side false peak is weaker and the main peak is more
  compact, but it is still offset above the target. In full episode 86, frames 60/120/180 are
  near-target and usable; frame 0 remains a stable tabletop/object lock. Conclusion remains:
  temporal-window labels help the clean target semantics, but pure latent diffusion/noise MSE still
  leaves scene-prior and multi-peak artifacts.

- 2026-06-14 02:57 +08 check: epoch 12 was the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 13 at about `28%` (`~1482/5359`) and roughly `1.48 it/s`; checkpoint watcher and
  episode watcher are alive.
- Epoch 12 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0012-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_023128_epoch=0012-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_023203_epoch=0012-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_episode86\comparison.mp4`
- Epoch 12 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `550.34s`.
- Visual read for epoch 12: the lower scalar validation loss (`0.003`) is not a monotonic visual
  improvement over epoch 11. Sample 0 remains near-target but slightly right/up with persistent
  lower-left speckles and a faint upper residual. Sample 1 is much better than the old baseline
  failure, but still left-biased and wider than the target. Sample 2 is still the main failure:
  the single object-texture lock becomes a multi-peak prediction with only weak mass near the gaze
  target. Sample 3 remains near-target but adds a clear right-side false peak. In full episode 86,
  frames 60/120/180 remain near-target and usable, while frame 0 still locks to the tabletop/object
  area instead of the left-up gaze target. The temporal-window clean label remains useful, but
  pure latent diffusion/noise MSE still leaves scene-prior attraction, multi-peak artifacts, and
  lower-left speckles.

- 2026-06-14 01:47 +08 check: epoch 11 was the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive in epoch 12 at about `17%` (`~920/5359`) and roughly `1.4 it/s`; checkpoint watcher and
  episode watcher are alive.
- Epoch 11 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0011-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_013035_epoch=0011-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_013117_epoch=0011-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_episode86\comparison.mp4`
- Epoch 11 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `535.07s`.
- Visual read for epoch 11: sample 0, sample 1, and sample 3 place their main decoded mass near
  the target, although sample 0 still has the fixed lower-left speckles and sample 1 is slightly
  left-biased. Sample 2 remains the key failure: the strongest mass locks to the lower box/object
  texture and a secondary false peak appears above, while the temporal target is centered near the
  gaze marker. In the full episode 86 preview, frames 60/120/180 are near-target and usable; frame
  0 still fails by placing the main mass on the tabletop/object area instead of the left-up gaze
  target. The temporal-window label remains a strong DiT clean-label candidate, but pure latent
  diffusion/noise MSE still leaves scene-prior attraction and texture speckles.

- 2026-06-13 22:45 +08 check: epoch 8 was the latest fully verified checkpoint with both
  checkpoint sample preview and full episode 86 preview generated successfully. Training is still
  alive and had advanced beyond this point; checkpoint watcher and episode watcher are alive.
- Epoch 8 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0008-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_222749_epoch=0008-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_222811_epoch=0008-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_episode86\comparison.mp4`
- Epoch 8 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `548.19s`.
- Visual read for epoch 8: validation loss remains low (`0.003`). Sample 1 keeps the improvement
  first seen around epoch 5: the main mass is near the target, and the old right-side object mode is
  only a secondary peak. Sample 3 returns near the target with a right tail. Sample 0 keeps a small
  right/up offset plus the persistent lower-left speckle cluster. Sample 2 is still a major failure:
  the main mode remains locked to the lower box/object texture instead of the gaze point. In full
  episode 86, frames 60/120/180 are near-target and visually usable, while frame 0 remains a stable
  failure that locks to the tabletop/object area. Latest evidence supports the temporal-window
  clean label as useful for DiT labels, but pure latent MSE alone has not removed all scene-prior
  failures.

- 2026-06-13 20:44 +08 check: epoch 6 completed, the run advanced to `epoch=7`, and both
  checkpoint sample preview plus full episode 86 preview were generated successfully. Training is
  still alive in `epoch=7`, about `76%`; `logs.json.txt` reached `global_step=10394`.
  Checkpoint watcher and episode watcher are alive.
- Epoch 6 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0006-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_202600_epoch=0006-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_202558_epoch=0006-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_episode86\comparison.mp4`
- Epoch 6 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `553.94s`.
- Visual read for epoch 6: validation loss remains low (`0.003`) and sample 1 keeps the epoch 5
  improvement: the main mass stays near the gaze/hand region and the old right-side object false
  mode is much weaker than epoch 4. However, sample 2 regresses with its main mass again locked
  onto the lower box/object texture, and sample 3 is horizontally stretched and shifted above the
  target compared with the cleaner epoch 4 version. Episode frames 60/120/180 remain near-target
  and usable, while frame 0 still fails by locking to the tabletop/object area. Latest evidence
  therefore supports temporal-window labels, but also shows that pure latent MSE has persistent
  scene-prior failures and non-monotonic visual quality despite low validation loss.

- 2026-06-13 19:39 +08 check: epoch 5 completed, the run advanced to `epoch=6`, and both
  checkpoint sample preview plus full episode 86 preview were generated successfully. Training is
  still alive in `epoch=6`, about `51%`; `logs.json.txt` reached `global_step=8719`.
  Checkpoint watcher and episode watcher are alive.
- Epoch 5 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0005-val_loss=0.003.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_192505_epoch=0005-val_loss=0.003`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_192449_epoch=0005-val_loss=0.003_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_episode86\comparison.mp4`
- Epoch 5 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `555.46s`.
- Visual read for epoch 5: validation loss improved again (`0.018 -> 0.011 -> 0.008 -> 0.007
  -> 0.004 -> 0.003`). Sample 1 improves materially compared with epoch 4: its strongest mass
  moves back near the gaze/hand region, although a right-side object false mode remains as a
  secondary peak. Sample 0 remains close but slightly right/up and still shows the fixed lower-left
  speckles. Sample 2 remains vertically elongated and biased by the box/object texture. Sample 3
  regresses relative to epoch 4, with the main mode shifted above the gaze point. In episode 86,
  frames 120 and 180 stay near-target and usable, frame 60 remains near-target with a small left
  bias, and frame 0 still fails by locking onto the tabletop/object area. Epoch 5 is numerically
  best and fixes one hard sample, but visually it is not a clean monotonic improvement over epoch 4.

- 2026-06-13 18:34 +08 check: epoch 4 completed, the run advanced to `epoch=5`, and both
  checkpoint sample preview plus full episode 86 preview were generated successfully. Training is
  still alive in `epoch=5`, about `34%`; `logs.json.txt` reached `global_step=7150`.
  Checkpoint watcher and episode watcher are alive.
- Epoch 4 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0004-val_loss=0.004.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_182410_epoch=0004-val_loss=0.004`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_182344_epoch=0004-val_loss=0.004_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_episode86\comparison.mp4`
- Epoch 4 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `551.64s`.
- Visual read for epoch 4: validation loss improved sharply (`0.018 -> 0.011 -> 0.008 -> 0.007
  -> 0.004`) and this is the best temporal-window checkpoint so far. Sample 3 is now much cleaner,
  with the main mode near the gaze point and a weaker right tail. Sample 0 remains close but still
  slightly right/up of the gaze point and still shows the fixed lower-left speckle cluster. Sample
  2 is somewhat more compact than epoch 3 but still activates on the box/object texture below/right
  of the target. Sample 1 remains the hardest sampled failure: a strong off-target mode persists on
  the right-side bottle/hand/object region. In the full episode 86 preview, frames 60/120/180 are
  near-target and visually usable; frame 0 still fails by placing the main mass on the tabletop
  object area instead of the left-up gaze target. Temporal-window labels remain the preferred DiT
  final clean label candidate, but pure latent MSE has not fully removed scene-prior/object-lock
  artifacts.

- 2026-06-13 17:39 +08 check: epoch 3 completed, the run advanced to `epoch=4`, and both
  checkpoint sample preview plus full episode 86 preview were generated successfully. Training is
  still alive in `epoch=4`, about `19%`; `logs.json.txt` reached `global_step=5654`.
  Checkpoint watcher and episode watcher are alive.
- Epoch 3 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0003-val_loss=0.007.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_172313_epoch=0003-val_loss=0.007`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_172533_epoch=0003-val_loss=0.007_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_episode86\comparison.mp4`
- Epoch 3 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `557.58s`.
- Visual read for epoch 3: validation loss improved again (`0.018 -> 0.011 -> 0.008 -> 0.007`).
  Sample 0 has its strongest mass close to the gaze point but retains the fixed lower-left speckle
  cluster. Sample 1 still puts a strong off-target mode on the right-side bottle/hand/object
  region instead of the gaze target. Sample 2 remains displaced onto the box/object texture below
  and right of the target. Sample 3 is closer than earlier checkpoints but still drags a rightward
  tail. In the full episode 86 preview, frames 120 and 180 are near-target and look usable as early
  evidence, while frames 0 and 60 still show scene-object attraction and spurious peaks. The
  temporal-window label remains promising, but epoch 3 is still not visually converged.

- 2026-06-13 16:47 +08 check: epoch 2 completed, the run advanced to `epoch=3`, and both
  checkpoint sample preview plus full episode 86 preview were generated successfully. Training is
  still alive in `epoch=3`, about `23%`; checkpoint watcher and episode watcher are alive.
- Epoch 2 checkpoint and previews:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.008.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_162219_epoch=0002-val_loss=0.008`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_162158_epoch=0002-val_loss=0.008_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_episode86\comparison.mp4`
- Epoch 2 summary fields: `source=open`, `split=val`, `use_ema=false`, no EMA in checkpoint,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`,
  temporal params `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all `3599/3599` frames at 15 FPS in about
  `522.43s`.
- Visual read for epoch 2: validation loss improved again (`0.018 -> 0.011 -> 0.008`). Samples 0
  and 3 now put their main mass close to the gaze/target, although both retain right/up tails or
  secondary peaks plus the persistent lower-left speckle cluster. Sample 1 still locks to a
  right-side object/edge region instead of the hand/gaze target, and sample 2 remains a clear
  off-target miss below the target on object texture. Episode frames 120 and 180 show better
  near-target mass than epoch 1, but frames 0/60 and the sampled failures show remaining
  semantic/texture locks and spurious peaks. This is an encouraging trend, not final convergence.

- 2026-06-13 16:15 +08 check: temporal-window training remains healthy in `epoch=2`, about
  `89%` (`4796/5359`) at roughly `1.47-1.49 it/s`. `logs.json.txt` reached
  `global_step=3878`; recent `train_heatmap_loss` rows are around `0.0122-0.0206`, with no action
  loss, `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, and 100% open rows. Training,
  checkpoint watcher, and episode watcher are all alive.
- No epoch 2 checkpoint or watcher preview exists yet. The newest available sample/episode visual
  evidence remains epoch 1.

- 2026-06-13 16:12 +08 check: temporal-window training remains healthy in `epoch=2`, about
  `83%` (`4472/5359`) at roughly `1.48-1.49 it/s`. `logs.json.txt` reached
  `global_step=3797`; recent `train_heatmap_loss` rows are around `0.0156-0.0183`, with no action
  loss, `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, and 100% open rows. Training,
  checkpoint watcher, and episode watcher are all alive.
- No epoch 2 checkpoint or watcher preview exists yet. The newest available sample/episode visual
  evidence remains epoch 1.

- 2026-06-13 16:07 +08 check: temporal-window training remains healthy in `epoch=2`, about
  `76%` (`4064/5359`) at roughly `1.46-1.49 it/s`. `logs.json.txt` reached
  `global_step=3695`; latest recent `train_heatmap_loss` rows are around `0.0156-0.0202`, with
  no action loss, `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, and 100% open rows.
  Training, checkpoint watcher, and episode watcher are all alive.
- No epoch 2 checkpoint or watcher preview exists yet. The newest available sample/episode visual
  evidence remains epoch 1.

- 2026-06-13 16:03 +08 check: temporal-window training remains healthy in `epoch=2`, about
  `69%` (`3680/5359`) at roughly `1.47-1.48 it/s`. `logs.json.txt` reached
  `global_step=3599`; the latest row reports `train_heatmap_loss=0.01375`, no action loss,
  `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, and 100% open rows. Training,
  checkpoint watcher, and episode watcher are all alive.
- No epoch 2 checkpoint or watcher preview exists yet. The newest available sample/episode visual
  evidence remains epoch 1.

- 2026-06-13 15:57 +08 check: temporal-window training remains healthy in `epoch=2`, about
  `59%` (`3148/5359`) at roughly `1.48-1.49 it/s`. `logs.json.txt` reached
  `global_step=3466`; recent `train_heatmap_loss` remains in the expected early-training range
  around `0.0155-0.0211`, with `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, no
  action loss, and 100% open rows. Training, checkpoint watcher, and episode watcher are all
  alive.
- No epoch 2 checkpoint or watcher preview exists yet. The newest available sample/episode visual
  evidence remains epoch 1.

- 2026-06-13 15:51 +08 check: temporal-window training is still healthy in `epoch=2`, about
  `50%` (`2668/5359`) at roughly `1.50 it/s`. The latest observed `logs.json.txt` reached
  `global_step=3346`; recent rows still match the intended open-only, no-action, no-gaze-condition,
  pure latent diffusion/noise MSE contract with `train_heatmap_xy_loss=0.0` and
  `train_heatmap_js_loss=0.0`. The training, checkpoint-preview watcher, and episode-preview
  watcher tmux sessions are all alive.
- No epoch 2 checkpoint, checkpoint preview, validation heatmap preview, or full-episode preview
  exists yet. The newest available visual evidence remains epoch 1.

- 2026-06-13 15:41 +08 check: temporal-window training is healthy and still running in
  `epoch=2`, about `31%` (`1680/5359`) at roughly `1.48-1.50 it/s`. The latest observed
  `logs.json.txt` reached `global_step=3099`, with recent `train_heatmap_loss` around
  `0.0148-0.0221`. The training, checkpoint-preview watcher, and episode-preview watcher tmux
  sessions are all alive.
- No epoch 2 checkpoint or watcher preview exists yet. The newest available checkpoint/preview is
  still epoch 1, so there is no new full-episode video beyond the epoch 1 episode86 render.
- Epoch 1 checkpoint and previews are available:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.011.ckpt`
  - checkpoint sample preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_152126_epoch=0001-val_loss=0.011`
  - local sample preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_ckpt_preview`
  - full episode 86 preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_152120_epoch=0001-val_loss=0.011_episode86`
  - local episode video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_episode86\comparison.mp4`
- Epoch 1 summary fields match the intended contract: `source=open`, `split=val`,
  `use_ema=false`, `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode 86 rendered all 3599 frames at 15 FPS.
- Visual read for epoch 1: validation loss improved from epoch 0 (`0.018 -> 0.011`), but the
  decoded predictions are still visually close to epoch 0. Some frames/samples put mass near the
  gaze point, especially sample 3 and episode frame 180, but sample 1/2 still show clear off-target
  object or texture locks, with persistent lower-left speckles and occasional vertical/border
  residuals. This is an encouraging numeric trend, not yet a convergence result.

- 2026-06-13 15:00 +08 check: temporal-window latent MSE training is still healthy. The three
  tmux sessions are alive: training, checkpoint-preview watcher, and episode-preview watcher.
  The training pane was in `epoch=1` at about `35%` (`1870/5359`) and roughly `1.48 it/s`.
  `logs.json.txt` reached `global_step=1869`; recent `train_heatmap_loss` rows were around
  `0.020-0.025`. Routing still confirms open-only training, no action loss, no gaze condition,
  and no DSNT/JS/xy auxiliary loss.
- First temporal-window checkpoint exists:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.018.ckpt`.
  No later named checkpoint exists yet; `latest.ckpt` currently mirrors epoch 0.
- First checkpoint watcher preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_142032_epoch=0000-val_loss=0.018`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_ckpt_preview`
- First full-episode watcher preview:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_142153_epoch=0000-val_loss=0.018_episode86`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_episode86`
  - rendered video:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_episode86\comparison.mp4`
- Epoch 0 preview summary fields: `source=open`, `split=val`, `use_ema=false`
  because the checkpoint has no EMA, `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`, temporal params
  `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, selected indices
  `6600,1341,9843,11637`. Episode preview rendered validation episode 86 with 3599 frames at
  15 FPS.
- Visual read: the temporal target panels are clean and match the intended label shape: current
  gaze is brightest, with a short temporally weighted neighborhood/tail. Epoch 0 model predictions
  are still early and not converged: some frames hit near the gaze, but there are clear offsets,
  object/texture locks, border activations, and lower-left speckle noise. This is a useful first
  sanity check for the target/video pipeline, not yet evidence of final convergence.

- 2026-06-13 13:46 +08 check: temporal-window latent MSE run is alive in tmux and training on
  8x H20. The training pane was in `epoch=0` at about `48%` (`2571/5359`) and roughly
  `1.5 it/s`. The latest `logs.json.txt` rows reached `global_step=642`, with recent
  `train_heatmap_loss` around `0.0340-0.0462`; startup rows were around `1.3`, so the new
  label path is learning normally.
- Training contract confirms:
  `name=train_gaze_wam_open_only_cosmos_temporal_delta_noise`,
  `open_train_samples=685918`, `open_val_samples=15039`,
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_label_source=temporal_window_dense_heatmap`, and temporal params
  `bidirectional / radius 30 / beta 10 / sigma 6 / current weight 2`.
- Current routing remains 100% open data: no action loss, no DSNT/JS/xy loss, no gaze condition;
  `train_heatmap_mask_count=512` per optimizer step.
- Watcher is alive and waiting for the first checkpoint under the new output directory. No
  checkpoint preview exists yet because epoch 0 has not completed.
- Episode watcher is also alive and will render validation episode 86 with
  `diffusion_policy.scripts.preview_gaze_wam_episode` when the first checkpoint appears.
- A first accidental launch was stopped because it was not attached to tmux; its partial output was
  archived at
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp__notmux_20260613_051802`.
  The active run is the clean tmux-backed directory listed above.

- 2026-06-11 12:57 +08 heartbeat check: training stayed alive in `epoch=195`, about `65%`
  (`3464/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=262169`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.46e-4` to `4.14e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New validation previews and watcher reload previews were available after epoch 191:
  - `latest.ckpt` watcher previews:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_122554_latest` and
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_123905_latest`
  - epoch 194 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_125217_epoch=0194-val_loss=0.002`
  - epoch 194 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0194`
  - local copies:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260611_122554_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260611_123905_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0194_cosmos_delta_noise_mse`, and
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_val_epoch0194_cosmos_delta_noise_mse`.
- The two `latest.ckpt` reload previews and epoch 194 confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 194 remains visually redundant with epochs 176-191 and is still plateaued.
  Sample 0 is close but biased right/up; sample 1 still locks onto the stable off-target left-side
  cup/object region; sample 2 remains a clear below/right miss; sample 3 is close but biased
  right/up. The lower-left cyan speckle cluster, faint right-edge texture, and blue background wash
  persist, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- 2026-06-11 12:19 +08 heartbeat check: training stayed alive in `epoch=192`, about `73%`
  (`3900/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=258254`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.56e-4` to `5.40e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New validation previews and watcher reload previews were available for epochs 190 and 191:
  - epoch 190 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_115930_epoch=0190-val_loss=0.002`
  - epoch 191 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_121241_epoch=0191-val_loss=0.002`
  - validation previews were present through:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0191`
  - local copies:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0190_cosmos_delta_noise_mse` and
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0191_cosmos_delta_noise_mse`.
- Epoch 190/191 summaries confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 191 remains visually redundant with epochs 176-189 and is still plateaued.
  Sample 0 is close but biased right/up; sample 1 still locks onto the stable off-target left-side
  cup/object region; sample 2 remains a clear below/right miss; sample 3 is close but biased
  right/up. The lower-left cyan speckle cluster, faint right-edge texture, and blue background wash
  persist, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- 2026-06-11 11:54 +08 heartbeat check: training stayed alive in `epoch=190`, about `73%`
  (`3924/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=255582`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.51e-4` to `4.05e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New validation previews and watcher reload previews were available after epoch 184:
  - epoch 185 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_105630_epoch=0185-val_loss=0.002`
  - `latest.ckpt` watcher previews:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_110942_latest` and
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_111954_latest`
  - epoch 188 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_113306_epoch=0188-val_loss=0.002`
  - epoch 189 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_114618_epoch=0189-val_loss=0.002`
  - validation previews were present through:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0189`
  - local copies:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0185_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260611_110942_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260611_111954_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0188_cosmos_delta_noise_mse`, and
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0189_cosmos_delta_noise_mse`.
- Epoch 185, the two `latest.ckpt` reload previews, and epochs 188/189 all confirm the same active
  preview contract: `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 189 remains visually redundant with epochs 176-184 and is still plateaued.
  Sample 0 is close but biased right/up; sample 1 still locks onto the stable off-target left-side
  cup/object region; sample 2 remains a clear below/right miss; sample 3 is close but biased
  right/up. The lower-left cyan speckle cluster, faint right-edge texture, and blue background wash
  persist, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- 2026-06-11 10:43 +08 heartbeat check: training stayed alive in `epoch=185`, about `4%`
  (`215/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=247952`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.66e-4` to `3.83e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New validation previews and watcher reload previews were available for epochs 181, 182, 183, and
  184:
  - epoch 181 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_100342_epoch=0181-val_loss=0.002`
  - epoch 182 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_101654_epoch=0182-val_loss=0.002`
  - epoch 183 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_103006_epoch=0183-val_loss=0.002`
  - epoch 184 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_104318_epoch=0184-val_loss=0.002`
  - epoch 184 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0184`
  - local copies:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0181_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0182_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0183_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0184_cosmos_delta_noise_mse`, and
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_val_epoch0184_cosmos_delta_noise_mse`.
- Epoch 181/182/183/184 summaries confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 184 remains visually redundant with epochs 176-180 and is still plateaued.
  Sample 0 is close but biased right/up; sample 1 still locks onto the stable off-target left-side
  cup/object region; sample 2 remains a clear below/right miss; sample 3 is close but biased
  right/up. The lower-left cyan speckle cluster, faint right-edge texture, and blue background wash
  persist, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- 2026-06-11 09:56 +08 heartbeat check: training stayed alive in `epoch=181`, about `61%`
  (`3264/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=243356`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.93e-4` to `3.42e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New validation previews and watcher reload previews were available for epochs 178, 179, and 180:
  - epoch 178 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_092407_epoch=0178-val_loss=0.002`
  - epoch 179 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_093718_epoch=0179-val_loss=0.002`
  - epoch 180 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_095030_epoch=0180-val_loss=0.002`
  - epoch 178-180 validation previews:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0178`,
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0179`, and
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0180`
  - local copies:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0178_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0179_cosmos_delta_noise_mse`, and
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0180_cosmos_delta_noise_mse`.
- Epoch 178/179/180 summaries confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 180 is still visually redundant with epochs 176-177 and remains plateaued.
  Sample 0 is close but biased right/up; sample 1 still locks onto the stable off-target left-side
  cup/object region; sample 2 remains a clear below/right miss; sample 3 is close but biased
  right/up. Spot checks across epochs 178-179 show the same pattern. The lower-left cyan speckle
  cluster, faint right-edge texture, and blue background wash persist, so pure Cosmos latent
  denoising still has not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- 2026-06-11 09:15 +08 heartbeat check: training stayed alive in `epoch=178`, about `43%`
  (`2296/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=239096`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.21e-4` to `4.82e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New checkpoint, validation preview, and watcher reload preview were available for epoch 177:
  - epoch 177 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0177-val_loss=0.002.ckpt`
  - epoch 177 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0177`
  - epoch 177 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_091355_epoch=0177-val_loss=0.002`
  - epoch 177 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0177_cosmos_delta_noise_mse`
- Epoch 177 summary confirms the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 177 is visually redundant with epoch 176 and remains plateaued. Sample 0 is
  close but biased right/up; sample 1 still locks onto the stable off-target left-side cup/object
  region; sample 2 is a clear below/right miss; sample 3 is close but biased right/up. The
  lower-left cyan speckle cluster, faint right-edge texture, and blue background wash persist, so
  pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- 2026-06-11 09:07 +08 heartbeat check: training stayed alive in `epoch=177`, about `76%`
  (`4063/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=238243`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.65e-4` to `3.82e-4`.
  GPUs were normal at about `12.6-13.0GB/97.9GB` with high utilization.
- New watcher reload previews were available beyond the previous documented epoch 155. Complete
  watcher summaries exist through epoch 176:
  - epoch 156:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_044248_epoch=0156-val_loss=0.002`
  - epoch 157:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_045600_epoch=0157-val_loss=0.002`
  - epoch 158-166, `latest` reload at `20260611_070458_latest`, and epoch 168-176 were also
    present under the watcher preview root.
  - latest pulled and inspected epoch 176 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_090044_epoch=0176-val_loss=0.002`
  - latest epoch 176 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0176-val_loss=0.002.ckpt`
  - latest epoch 176 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0176`
  - local copies checked this heartbeat:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0156_cosmos_delta_noise_mse`,
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0157_cosmos_delta_noise_mse`, and
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0176_cosmos_delta_noise_mse`.
- Server-side summaries for the new watched previews from epoch 156 through epoch 176 all continue
  to report the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`,
  `num_inference_steps=8`, `heatmap_dim=16`, frozen Cosmos CI16x16,
  `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping, no gaze condition,
  no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read of the latest pulled epoch 176 preview: the run is still visually plateaued rather
  than cleaner. Sample 0 is close but biased right/up of the gaze marker; sample 1 still locks onto
  the stable off-target left-side cup/object region; sample 2 is a clear below/right miss; sample 3
  is near the marker but still slightly high. The lower-left cyan speckle cluster, faint right-edge
  texture, and blue background wash persist, so pure Cosmos latent denoising still has not
  meaningfully reduced the scene-texture/checkerboard-like artifacts.
- 2026-06-11 04:35 +08 heartbeat check: training stayed alive in `epoch=156`, about `62%`
  (`3348/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=209875`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.8e-4` to `4.5e-4`.
- New checkpoint, validation preview, and watcher reload previews were available for epochs 153,
  154, and 155:
  - epoch 153 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0153-val_loss=0.002.ckpt`
  - epoch 153 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0153`
  - epoch 153 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_040313_epoch=0153-val_loss=0.002`
  - epoch 153 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0153_cosmos_delta_noise_mse`
  - epoch 154 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0154-val_loss=0.002.ckpt`
  - epoch 154 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0154`
  - epoch 154 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_041624_epoch=0154-val_loss=0.002`
  - epoch 154 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0154_cosmos_delta_noise_mse`
  - epoch 155 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0155-val_loss=0.002.ckpt`
  - epoch 155 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0155`
  - epoch 155 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_042936_epoch=0155-val_loss=0.002`
  - epoch 155 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0155_cosmos_delta_noise_mse`
- Epoch 153/154/155 summaries confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epochs 153-155 remain plateaued. Sample 0 is close but still offset slightly
  right/up of the gaze marker; sample 1 still locks onto the fixed off-target left-side
  cup/object region; sample 2 is above/right at epoch 153 and below/right by epochs 154-155;
  sample 3 keeps a small right/up offset. The persistent lower-left speckle cluster, faint
  right-edge texture, and blue background wash remain, so pure Cosmos latent denoising still has
  not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- GPU memory stayed normal: all GPUs were around `12.6-13.0GB/98GB` with high utilization, with
  only normal training-rank allocations visible. The old `[Not Found]` PID `1748481` holding about
  `74.6GB` on every GPU did not appear.
- 2026-06-11 03:54 +08 heartbeat check: training stayed alive in `epoch=153`, about `44%`
  (`2334/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=205601`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `3.0e-4` to `3.9e-4`.
- New checkpoint, validation preview, and watcher reload previews were available for epochs 150,
  151, and 152:
  - epoch 150 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0150-val_loss=0.002.ckpt`
  - epoch 150 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0150`
  - epoch 150 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_032637_epoch=0150-val_loss=0.002`
  - epoch 150 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0150_cosmos_delta_noise_mse`
  - epoch 151 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0151-val_loss=0.002.ckpt`
  - epoch 151 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0151`
  - epoch 151 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_033648_epoch=0151-val_loss=0.002`
  - epoch 151 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0151_cosmos_delta_noise_mse`
  - epoch 152 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0152-val_loss=0.002.ckpt`
  - epoch 152 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0152`
  - epoch 152 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_035000_epoch=0152-val_loss=0.002`
  - epoch 152 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0152_cosmos_delta_noise_mse`
- Epoch 150/151/152 summaries confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epochs 150-152 remain plateaued. Sample 0 is close but still offset slightly
  right/up of the gaze marker; sample 1 still locks onto the fixed off-target left-side
  cup/object region; sample 2 is above/right for epochs 150-151 and flips back to a below/right
  miss at epoch 152; sample 3 keeps a small right/up offset. The persistent lower-left speckle
  cluster, faint right-edge texture, and blue background wash remain, so pure Cosmos latent
  denoising still has not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- GPU memory stayed normal: all GPUs were around `12.6-13.0GB/98GB` with high utilization, with
  only normal training-rank allocations visible. The old `[Not Found]` PID `1748481` holding about
  `74.6GB` on every GPU did not appear.
- 2026-06-11 03:14 +08 heartbeat check: training stayed alive in `epoch=150`, about `34%`
  (`1820/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=201456`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.6e-4` to `4.4e-4`.
- The checkpoint, validation preview, and watcher reload preview were available for epoch 149:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0149-val_loss=0.002.ckpt`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0149`
  - server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_031325_epoch=0149-val_loss=0.002`
  - local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0149_cosmos_delta_noise_mse`
- Epoch 149 summary confirms the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 149 remains plateaued and is visually redundant with epoch 148. Sample 0 is
  close but still offset slightly right/up of the gaze marker; sample 1 still locks onto the fixed
  off-target left-side cup/object region; sample 2 remains above/right of the marker; sample 3
  keeps a small right/up offset. The persistent lower-left speckle cluster, faint right-edge
  texture, and blue background wash remain, so pure Cosmos latent denoising still has not
  meaningfully reduced the scene-texture/checkerboard-like artifacts.
- GPU memory stayed normal: all GPUs were around `12.6-13.0GB/98GB` with high utilization, with
  only normal training-rank allocations visible. The old `[Not Found]` PID `1748481` holding about
  `74.6GB` on every GPU did not appear.
- 2026-06-11 03:02 +08 heartbeat check: training stayed alive in `epoch=149`, about `28%`
  (`1521/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=200039`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `4.8e-4`.
- New checkpoint, validation preview, and watcher reload previews were available for epochs 145,
  146, 147, and 148:
  - epoch 145 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0145-val_loss=0.002.ckpt`
  - epoch 145 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0145`
  - epoch 145 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_022038_epoch=0145-val_loss=0.002`
  - epoch 145 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0145_cosmos_delta_noise_mse`
  - epoch 146 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0146-val_loss=0.002.ckpt`
  - epoch 146 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0146`
  - epoch 146 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_023349_epoch=0146-val_loss=0.002`
  - epoch 146 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0146_cosmos_delta_noise_mse`
  - epoch 147 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0147-val_loss=0.002.ckpt`
  - epoch 147 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0147`
  - epoch 147 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_024701_epoch=0147-val_loss=0.002`
  - epoch 147 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0147_cosmos_delta_noise_mse`
  - epoch 148 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0148-val_loss=0.002.ckpt`
  - epoch 148 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0148`
  - epoch 148 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_030013_epoch=0148-val_loss=0.002`
  - epoch 148 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0148_cosmos_delta_noise_mse`
- Epoch 145/146/147/148 summaries confirm the same active preview contract:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`, `heatmap_dim=16`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epochs 145-148 remain plateaued. Sample 0 is close but still offset slightly
  right/up of the gaze marker; sample 1 still locks onto the fixed off-target left-side
  cup/object region; sample 2 jitters by checkpoint and is clearly above/right of the marker at
  epoch 148 after the epoch 147 below/right miss; sample 3 keeps a small right/up offset. The
  persistent lower-left speckle cluster, faint right-edge texture, and blue background wash remain,
  so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The previous GPU-memory anomaly was absent again: all GPUs were around `12.6-13.0GB/98GB` with
  high utilization, and only normal training-rank allocations were visible. The old `[Not Found]`
  PID `1748481` holding about `74.6GB` on every GPU did not appear.
- 2026-06-11 02:11 +08 heartbeat check: training stayed alive in `epoch=145`, about `38%`
  (`2053/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=194815`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.8e-4` to `4.5e-4`.
- New checkpoint, validation preview, and watcher reload previews were available for epochs 142,
  143, and 144:
  - epoch 142 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0142-val_loss=0.002.ckpt`
  - epoch 142 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0142`
  - epoch 142 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_013501_epoch=0142-val_loss=0.002`
  - epoch 142 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0142_cosmos_delta_noise_mse`
  - epoch 143 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0143-val_loss=0.002.ckpt`
  - epoch 143 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0143`
  - epoch 143 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_015414_epoch=0143-val_loss=0.002`
  - epoch 143 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0143_cosmos_delta_noise_mse`
  - epoch 144 checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0144-val_loss=0.002.ckpt`
  - epoch 144 validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0144`
  - epoch 144 server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_020725_epoch=0144-val_loss=0.002`
  - epoch 144 local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0144_cosmos_delta_noise_mse`
- Epoch 142/143/144 summaries confirm the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epochs 142-144 remain plateaued. Sample 0 is close but still offset slightly
  right/up of the gaze marker; sample 1 still locks onto the fixed off-target left-side
  cup/hand/object region; sample 2 was closer in epochs 142-143 but regressed at epoch 144 to the
  older below/right miss; sample 3 keeps a small right/up offset. The persistent lower-left speckle
  cluster, faint right-edge texture, and blue background wash remain, so pure Cosmos latent
  denoising still has not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- The previous GPU-memory anomaly was not present in this check: all GPUs were around
  `12.6-13.0GB/98GB` and `92-96%` utilization, with only the normal training-rank allocations
  visible. The old `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU did not appear.
- 2026-06-11 00:54 +08 heartbeat check: training stayed alive in `epoch=142`, about `16%`
  (`862/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=190494`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.3e-4` to `4.3e-4`.
- The checkpoint, validation preview, and watcher reload preview were available for epoch 141:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0141-val_loss=0.002.ckpt`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0141`
  - server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_004847_epoch=0141-val_loss=0.002`
  - local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0141_cosmos_delta_noise_mse`
- Epoch 141 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 141 remains mostly plateaued. Sample 0 is close but still offset slightly
  right/up of the gaze marker; sample 1 still locks onto the fixed off-target left-side
  cup/hand/object region; sample 2 moved closer than epoch 140 but remains upper/right of the
  marker; sample 3 keeps a small right/up offset. The persistent lower-left speckle cluster,
  faint right-edge texture, and blue background wash remain, so pure Cosmos latent denoising still
  has not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-11 00:23 +08 heartbeat check: training stayed alive in `epoch=141`, about `42%`
  (`2242/5359`) at roughly `1.9-2.1 it/s`; `logs.json.txt` reached `global_step=189499`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `5.2e-4`.
- The checkpoint, validation preview, and watcher reload preview were available for epoch 140:
  - checkpoint:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0140-val_loss=0.002.ckpt`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/val_heatmap/epoch_0140`
  - server preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260611_000234_epoch=0140-val_loss=0.002`
  - local preview:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0140_cosmos_delta_noise_mse`
- Epoch 140 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 140 remains plateaued. Sample 0 is close but still offset slightly right/up
  of the gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object
  region; sample 2 remains clearly displaced below/right of the marker; sample 3 keeps a small
  right/up offset. The persistent lower-left speckle cluster, faint right-edge texture, and blue
  background wash remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 23:44 +08 heartbeat check: training stayed alive in `epoch=140`, about `61%`
  (`3265/5359`) at roughly `1.8-2.1 it/s`; `logs.json.txt` reached `global_step=188414`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.1e-4` to `5.3e-4`.
- The checkpoint-preview watcher produced the epoch 139 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_231918_epoch=0139-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0139_cosmos_delta_noise_mse`.
- Epoch 139 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 139 remains plateaued. Sample 0 is close but still offset slightly right/up
  of the gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object
  region; sample 2 remains clearly displaced below/right of the marker; sample 3 keeps a small
  right/up offset. The persistent lower-left speckle cluster, faint right-edge texture, and blue
  background wash remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 22:39 +08 heartbeat check: training stayed alive in `epoch=139`, about `12%`
  (`664/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=186425`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `4.6e-4`.
- The epoch 138 checkpoint and validation preview appeared first; the watcher was not broken. It
  logged checkpoint detection at `2026-06-10T22:35:19+08:00`, waited 45 seconds for file stability,
  then produced the epoch 138 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_223604_epoch=0138-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0138_cosmos_delta_noise_mse`.
- Epoch 138 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 138 remains plateaued. Sample 0 is close but still offset slightly right/up
  of the gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object
  region; sample 2 remains clearly displaced below/right of the marker; sample 3 keeps a small
  right/up offset. The persistent lower-left speckle cluster, faint right-edge texture, and blue
  background wash remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 21:59 +08 heartbeat check: training stayed alive in `epoch=138`, about `26%`
  (`1372/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=185263`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.0e-4` to `5.3e-4`.
- The checkpoint-preview watcher produced the epoch 137 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_214950_epoch=0137-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0137_cosmos_delta_noise_mse`.
- Epoch 137 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 137 remains plateaued. Sample 0 is close but still offset slightly right/up
  of the gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object
  region; sample 2 remains displaced below/right of the marker; sample 3 keeps a small right/up
  offset. The persistent lower-left speckle cluster, faint right-edge texture, and blue background
  wash remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 21:24 +08 heartbeat check: training stayed alive in `epoch=137`, about `48%`
  (`2547/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=184215`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `1.8e-4` to `5.6e-4`.
- The checkpoint-preview watcher produced the epoch 136 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_210637_epoch=0136-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0136_cosmos_delta_noise_mse`.
- Epoch 136 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 136 remains plateaued. Sample 0 is close but still offset right/up of the
  gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object region;
  sample 2 remains displaced below/right of the marker; sample 3 keeps a small right/up offset.
  The persistent lower-left speckle cluster, faint right-edge texture, and blue background wash
  remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 20:49 +08 heartbeat check: training stayed alive in `epoch=136`, about `69%`
  (`3708/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=183166`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.6e-4` to `4.9e-4`.
- The checkpoint-preview watcher produced the epoch 135 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_202033_epoch=0135-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0135_cosmos_delta_noise_mse`.
- Epoch 135 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 135 remains plateaued. Sample 0 is close but still offset right/up of the
  gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object region;
  sample 2 remains clearly below/right of the marker; sample 3 keeps a small right/up offset.
  The persistent lower-left speckle cluster, faint right-edge texture, and blue background wash
  remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 19:39 +08 heartbeat check: training stayed alive in `epoch=135`, about `11%`
  (`601/5359`) at roughly `1.9 it/s`; `logs.json.txt` reached `global_step=181049`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.3e-4` to `5.1e-4`.
- The checkpoint-preview watcher produced the epoch 134 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_193720_epoch=0134-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0134_cosmos_delta_noise_mse`.
- Epoch 134 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 134 is still plateaued. Sample 0 remains close but slightly right/up of the
  gaze marker; sample 1 still locks onto the fixed off-target left-side cup/hand/object region;
  sample 2 moves clearly below/right of the marker again; sample 3 keeps the small upper/right
  offset. The persistent lower-left speckle cluster, faint right-edge texture, and blue background
  wash remain, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 19:04 +08 heartbeat check: training stayed alive in `epoch=134`, about `33%`
  (`1793/5359`) at roughly `1.9 it/s`; `logs.json.txt` reached `global_step=180007`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.4e-4` to `4.1e-4`.
- The checkpoint-preview watcher produced the epoch 133 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_185106_epoch=0133-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0133_cosmos_delta_noise_mse`.
- Epoch 133 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 133 is still plateaued. Sample 0 remains close but slightly right/up of the
  gaze marker; sample 1 still locks onto the off-target left-side cup/hand/object region; sample 2
  has shifted back above/right of the marker after epoch 132's below/right position; sample 3 keeps
  the small upper/right offset. The persistent lower-left speckle cluster, faint right-edge
  texture, and blue background wash remain, so pure Cosmos latent denoising still has not
  meaningfully reduced the scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 18:34 +08 heartbeat check: training stayed alive in `epoch=133`, about `65%`
  (`3462/5359`) at roughly `1.9-2.1 it/s`; `logs.json.txt` reached `global_step=179083`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `4.9e-4`.
- The checkpoint-preview watcher produced the epoch 132 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_180753_epoch=0132-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0132_cosmos_delta_noise_mse`.
- Epoch 132 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 132 remains in the same plateau. Sample 0 is still slightly right/up of the
  gaze marker; sample 1 keeps the fixed off-target peak left of the marker around the
  cup/hand/object region; sample 2 moved back below/right after the epoch 131 above/right
  regression; sample 3 keeps a small upper/right offset. The persistent lower-left speckle cluster
  and blue background wash remain, so pure Cosmos latent denoising still has not meaningfully
  reduced the scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 99-100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 17:56 +08 heartbeat check: training stayed alive in `epoch=132`, about `78%`
  (`4170/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=177922`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `4.7e-4`.
- The checkpoint-preview watcher produced the epoch 131 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_172139_epoch=0131-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0131_cosmos_delta_noise_mse`.
- Epoch 131 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 131 still does not show a meaningful recovery. Sample 0 remains only
  marginally closer to the gaze marker than the older previews; sample 1 remains fixed off-target
  left of the gaze marker around the cup/hand/object region; sample 3 keeps the small upper/right
  offset. Sample 2 regressed relative to the recent below/right pattern and now places the peak
  above/right of the marker. The fixed lower-left speckle cluster and blue background wash remain
  unchanged, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 16:46 +08 heartbeat check: training stayed alive in `epoch=131`, about `21%`
  (`1108/5359`) at roughly `1.8-1.9 it/s`; `logs.json.txt` reached `global_step=175816`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.1e-4` to `4.2e-4`.
- The checkpoint-preview watcher produced the epoch 130 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_163823_epoch=0130-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0130_cosmos_delta_noise_mse`.
- Epoch 130 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 130 remains effectively unchanged from epochs 128-129. Sample 0 stays
  marginally closer to the gaze marker than the older previews, but sample 1 is still fixed
  off-target left of the gaze marker around the cup/hand/object region, sample 2 remains
  below/right, and sample 3 keeps a small upper/right offset. The fixed lower-left speckle cluster
  and blue background wash remain unchanged, so pure Cosmos latent denoising still has not
  meaningfully reduced the scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 16:12 +08 heartbeat check: training stayed alive in `epoch=130`, about `44%`
  (`2380/5359`) at roughly `1.8-1.9 it/s`; `logs.json.txt` reached `global_step=174794`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `4.9e-4`.
- The checkpoint-preview watcher produced the epoch 129 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_155210_epoch=0129-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0129_cosmos_delta_noise_mse`.
- Epoch 129 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 129 preserves the same plateau as epoch 128. Sample 0 remains marginally
  closer to the gaze marker than the older previews, but sample 1 is still fixed off-target left
  of the gaze marker around the cup/hand/object region, sample 2 remains below/right, and sample 3
  keeps a small upper/right offset. The fixed lower-left speckle cluster and blue background wash
  remain unchanged, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 15:36 +08 heartbeat check: training stayed alive in `epoch=129`, about `61%`
  (`3293/5359`) at roughly `1.8-2.0 it/s`; `logs.json.txt` reached `global_step=173681`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `5.2e-4`.
- The checkpoint-preview watcher produced the epoch 128 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_150856_epoch=0128-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0128_cosmos_delta_noise_mse`.
- Epoch 128 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 128 is still in the same late-epoch plateau. Sample 0 is marginally closer to
  the gaze marker than the immediately prior previews, but sample 1 remains fixed off-target left
  of the gaze marker around the cup/hand/object region, sample 2 remains below/right, and sample 3
  keeps a small upper/right offset. The fixed lower-left speckle cluster and blue background wash
  remain unchanged, so pure Cosmos latent denoising still has not meaningfully reduced the
  scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 14:30 +08 heartbeat check: training stayed alive in `epoch=128`, about `19%`
  (`1027/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=171757`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.6e-4` to `4.8e-4`.
- The checkpoint-preview watcher produced the epoch 127 reload preview after its normal 45-second
  file-stability wait:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_142543_epoch=0127-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0127_cosmos_delta_noise_mse`.
- Epoch 127 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 127 is again visually redundant with the late-epoch plateau. Sample 0 and
  sample 3 keep small local upper/right offsets; sample 1 remains fixed off-target left of the gaze
  marker around the cup/hand/object region; sample 2 remains displaced below/right of the target.
  The fixed lower-left speckle cluster and blue background wash remain unchanged, so pure Cosmos
  latent denoising still has not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 13:48 +08 heartbeat check: training stayed alive in `epoch=127`, about `26%`
  (`1413/5359`) at roughly `1.8-2.0 it/s`; `logs.json.txt` reached `global_step=170532`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `5.4e-4`.
- The checkpoint-preview watcher produced the epoch 126 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_133930_epoch=0126-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0126_cosmos_delta_noise_mse`.
- Epoch 126 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 126 remains visually redundant with epochs 124-125 and the late-epoch plateau.
  Sample 0 and sample 3 keep small local upper/right offsets; sample 1 remains fixed off-target
  left of the gaze marker around the cup/hand/object region; sample 2 remains displaced below/right
  of the target. The fixed lower-left speckle cluster and blue background wash remain unchanged, so
  pure Cosmos latent denoising still has not meaningfully reduced the scene-texture/checkerboard-like
  artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 13:13 +08 heartbeat check: training stayed alive in `epoch=126`, about `48%`
  (`2582/5359`) at roughly `1.9-2.1 it/s`; `logs.json.txt` reached `global_step=169485`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.2e-4` to `5.9e-4`.
- The checkpoint-preview watcher produced the epoch 125 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_125316_epoch=0125-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0125_cosmos_delta_noise_mse`.
- Epoch 125 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 125 is visually redundant with epoch 124 and the late-epoch plateau. Sample 0
  and sample 3 keep small local upper/right offsets; sample 1 remains fixed off-target left of the
  gaze marker around the cup/hand/object region; sample 2 remains displaced below/right of the
  target. The fixed lower-left speckle cluster and blue background wash remain unchanged, so pure
  Cosmos latent denoising still has not meaningfully reduced the scene-texture/checkerboard-like
  artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 12:38 +08 heartbeat check: training stayed alive in `epoch=125`, about `75%`
  (`4029/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=168507`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.1e-4` to `4.8e-4`.
- The checkpoint-preview watcher produced the epoch 124 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_121003_epoch=0124-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0124_cosmos_delta_noise_mse`.
- Epoch 124 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`,
  scheduler clipping, no gaze condition, no EMA, and selected indices `6600,1341,9843,11637`.
- Visual read: epoch 124 remains essentially unchanged from the late-epoch plateau. Sample 0 and
  sample 3 keep small local upper/right offsets; sample 1 remains fixed off-target left of the gaze
  marker around the cup/hand/object region; sample 2 remains displaced below/right of the target.
  The fixed lower-left speckle cluster and blue background wash remain unchanged, so pure Cosmos
  latent denoising has not meaningfully reduced the scene-texture/checkerboard-like artifacts.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 11:28 +08 heartbeat check: training stayed alive in `epoch=124`, about `15%`
  (`815/5359`) at roughly `1.8-2.0 it/s`; `logs.json.txt` reached `global_step=166363`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `5.2e-4`.
- The checkpoint-preview watcher produced the epoch 123 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_112350_epoch=0123-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0123_cosmos_delta_noise_mse`.
- Epoch 123 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 123 remains essentially unchanged from the late-epoch plateau. Sample 0 and
  sample 3 keep small local offsets; sample 1 remains fixed off-target to the left of the gaze
  marker around the cup/hand/object region; sample 2 remains displaced below/right of the target.
  The fixed lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 10:53 +08 heartbeat check: training stayed alive in `epoch=123`, about `38%`
  (`2042/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=165329`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.5e-4` to `5.2e-4`.
- The checkpoint-preview watcher produced the epoch 122 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_104032_epoch=0122-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0122_cosmos_delta_noise_mse`.
- Epoch 122 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 122 remains essentially unchanged from the late-epoch plateau. Sample 0 and
  sample 3 keep small local offsets; sample 1 remains fixed off-target to the left of the gaze
  marker around the cup/hand/object region; sample 2 remains displaced below/right of the target.
  The fixed lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 10:18 +08 heartbeat check: training stayed alive in `epoch=122`, about `59%`
  (`3186/5359`) at roughly `1.9-2.1 it/s`; `logs.json.txt` reached `global_step=164276`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.9e-4` to `4.7e-4`.
- The checkpoint-preview watcher produced the epoch 121 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_095419_epoch=0121-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0121_cosmos_delta_noise_mse`.
- Epoch 121 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 121 remains essentially unchanged from the late-epoch plateau. Sample 0 and
  sample 3 keep small local offsets; sample 1 remains fixed off-target to the left of the gaze
  marker around the cup/hand/object region; sample 2 remains displaced below/right of the target.
  The fixed lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 09:08-09:11 +08 heartbeat check: training stayed alive in `epoch=121`,
  about `3%` (`165/5359`) at roughly `2.0 it/s`; `logs.json.txt` reached
  `global_step=162309`. Recent rows still matched the intended contract: 100% open data,
  `use_gaze_condition=false`, no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE
  around `3.2e-4` to `5.3e-4`.
- The epoch 120 checkpoint and validation preview appeared before the watcher preview. The watcher
  was not broken: it logged checkpoint detection at `2026-06-10T09:10:20+08:00`, waited 45 seconds
  for file stability, then produced the epoch 120 reload preview at `09:11:05`.
- The checkpoint-preview watcher produced the epoch 120 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_091105_epoch=0120-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0120_cosmos_delta_noise_mse`.
- Epoch 120 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 120 remains essentially unchanged from epochs 117-119. Sample 0 and sample 3
  keep small local offsets; sample 1 remains fixed off-target to the left of the gaze marker around
  the cup/hand/object region; sample 2 remains displaced below/right of the target. The fixed
  lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 08:33 +08 heartbeat check: training stayed alive in `epoch=120`, about `24%`
  (`1307/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=161126`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `3.1e-4` to `5.5e-4`.
- The checkpoint-preview watcher produced the epoch 119 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_082451_epoch=0119-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0119_cosmos_delta_noise_mse`.
- Epoch 119 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 119 remains essentially unchanged from epochs 117-118. Sample 0 and sample 3
  keep small local offsets; sample 1 remains fixed off-target to the left of the gaze marker around
  the cup/hand/object region; sample 2 remains displaced below/right of the target. The fixed
  lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 07:58 +08 heartbeat check: training stayed alive in `epoch=119`, about `46%`
  (`2472/5359`) at roughly `1.7-1.9 it/s`; `logs.json.txt` reached `global_step=160077`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.9e-4` to `6.0e-4`.
- The checkpoint-preview watcher produced the epoch 118 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_074136_epoch=0118-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0118_cosmos_delta_noise_mse`.
- Epoch 118 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 118 is essentially unchanged from epoch 117. Sample 0 and sample 3 keep
  small local offsets; sample 1 remains fixed off-target to the left of the gaze marker around
  the cup/hand/object region; sample 2 remains displaced below/right of the target. The fixed
  lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 07:23 +08 heartbeat check: training stayed alive in `epoch=118`, about `68%`
  (`3670/5359`) at roughly `1.8-1.9 it/s`; `logs.json.txt` reached `global_step=159036`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.9e-4` to `5.2e-4`.
- The checkpoint-preview watcher produced the epoch 117 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_065523_epoch=0117-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0117_cosmos_delta_noise_mse`.
- Epoch 117 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 117 is still not a qualitative improvement. Sample 0 and sample 3 keep
  small local offsets; sample 1 remains fixed off-target to the left of the gaze marker around
  the cup/hand/object region; sample 2 remains displaced below/right of the target. The fixed
  lower-left speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 06:13 +08 heartbeat check: training stayed alive in `epoch=117`, about `14%`
  (`767/5359`) at roughly `2.0 it/s`; `logs.json.txt` reached `global_step=156971`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE only.
- The checkpoint-preview watcher produced the epoch 116 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_060908_epoch=0116-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0116_cosmos_delta_noise_mse`.
- Epoch 116 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 116 is still not a qualitative improvement. Sample 0 stays recovered from
  the epoch 114 far upper-left regression back to a small local offset; sample 1 remains fixed
  off-target to the left of the gaze marker around the cup/hand/object region; sample 2 remains
  displaced below/right of the target; sample 3 keeps a small local offset. The fixed lower-left
  speckle cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 05:38 +08 heartbeat check: training stayed alive in `epoch=116`, about `36%`
  (`1939/5359`) at roughly `1.9-2.1 it/s`; `logs.json.txt` reached `global_step=155924`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.9e-4` to `4.2e-4`.
- The checkpoint-preview watcher produced the epoch 115 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_052553_epoch=0115-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0115_cosmos_delta_noise_mse`.
- Epoch 115 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 115 recovers sample 0 from the epoch 114 far upper-left regression back to a
  small local offset, but it is still not a qualitative improvement. Sample 1 remains fixed
  off-target to the left of the gaze marker, sample 2 remains displaced below/right of the target,
  and sample 3 keeps a small local offset. The fixed lower-left speckle cluster and blue background
  wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 05:03 +08 heartbeat check: training stayed alive in `epoch=115`, about `58%`
  (`3130/5359`) at roughly `1.9-2.1 it/s`; `logs.json.txt` reached `global_step=154881`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `4.1e-4` to `4.9e-4`.
- The checkpoint-preview watcher produced the epoch 114 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_043940_epoch=0114-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0114_cosmos_delta_noise_mse`.
- Epoch 114 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 114 is not an improvement. Sample 0 regressed to a far upper-left off-target
  peak; sample 1 remains fixed off-target to the left of the gaze marker; sample 2 remains
  strongly displaced below/right of the target; sample 3 keeps the same small local offset. The
  fixed lower-left speckle cluster and blue background wash remain unchanged. The late-epoch
  previews are non-monotonic but stay within the same frozen Cosmos heatmap-codec failure mode.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 03:53 +08 heartbeat check: training stayed alive in `epoch=114`, about `3%`
  (`169/5359`) at roughly `1.9 it/s`; `logs.json.txt` reached `global_step=152801`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `3.4e-4` to `4.9e-4`.
- The checkpoint-preview watcher produced the epoch 113 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_035326_epoch=0113-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0113_cosmos_delta_noise_mse`.
- Epoch 113 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 113 is unchanged from epoch 112. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target to the left of the gaze marker around the cup/hand/object region, and
  sample 2 remains strongly displaced below/right of the target. The fixed lower-left speckle
  cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 03:18 +08 heartbeat check: training stayed alive in `epoch=113`, about `25%`
  (`1367/5359`) at roughly `1.8-2.0 it/s`; `logs.json.txt` reached `global_step=151761`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.8e-4` to `4.5e-4`.
- The checkpoint-preview watcher produced the epoch 112 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_031011_epoch=0112-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0112_cosmos_delta_noise_mse`.
- Epoch 112 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 112 is unchanged from epoch 111. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target to the left of the gaze marker around the cup/hand/object region, and
  sample 2 remains strongly displaced below/right of the target. The fixed lower-left speckle
  cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 02:43 +08 heartbeat check: training stayed alive in `epoch=112`, about `47%`
  (`2539/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=150714`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `2.8e-4` to `4.5e-4`.
- The checkpoint-preview watcher produced the epoch 111 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_022658_epoch=0111-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0111_cosmos_delta_noise_mse`.
- Epoch 111 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 111 is unchanged from epoch 110. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target to the left of the gaze marker around the cup/hand/object region, and
  sample 2 remains strongly displaced below/right of the target. The fixed lower-left speckle
  cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 02:08 +08 heartbeat check: training stayed alive in `epoch=111`, about `69%`
  (`3709/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=149666`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `3e-4` to `5e-4`.
- The checkpoint-preview watcher produced the epoch 110 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_014044_epoch=0110-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0110_cosmos_delta_noise_mse`.
- Epoch 110 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 110 is unchanged from epoch 109. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target to the left of the gaze marker around the cup/hand/object region, and
  sample 2 remains strongly displaced below/right of the target. The fixed lower-left speckle
  cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 00:58 +08 heartbeat check: training stayed alive in `epoch=110`, about `14%`
  (`728/5359`) at roughly `1.8-2.0 it/s`; `logs.json.txt` reached `global_step=147581`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `3e-4` to `4e-4`.
- The checkpoint-preview watcher produced the epoch 109 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_005430_epoch=0109-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0109_cosmos_delta_noise_mse`.
- Epoch 109 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 109 is unchanged from epoch 108. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target to the left of the gaze marker around the cup/hand/object region, and
  sample 2 remains strongly displaced below/right of the target. The fixed lower-left speckle
  cluster and blue background wash remain unchanged.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-10 00:23 +08 heartbeat check: training stayed alive in `epoch=109`, about `51%`
  (`2751/5359`) at roughly `1.9-2.0 it/s`; `logs.json.txt` reached `global_step=146810`.
  Recent rows still matched the intended contract: 100% open data, `use_gaze_condition=false`,
  no action loss, no DSNT/JS/xy loss, and pure latent heatmap MSE around `3e-4` to `5e-4`.
- The checkpoint-preview watcher produced the epoch 108 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260610_001116_epoch=0108-val_loss=0.002`.
  Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0108_cosmos_delta_noise_mse`.
- Epoch 108 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  frozen Cosmos CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping,
  no gaze condition, and no EMA.
- Visual read: epoch 108 is unchanged from epoch 107. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target around the cup/hand/object region, and sample 2 remains strongly
  down-shifted. The fixed lower-left speckle cluster and blue background wash remain unchanged.
  The late-epoch plateau continues to point at the heatmap representation/codec path rather than
  insufficient training time.
- The GPU-memory anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training-rank allocations around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU.
  No process was killed.
- 2026-06-08 18:53 +08: stopped the DSNT/JS intensity-softplus run and watcher:
  - `gaze_wam_open_cosmos_intensity_8gpu`
  - `gaze_wam_cosmos_intensity_ckpt_preview_watch`
  The previous output directory is preserved for comparison:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_intensity_softplus_latent_8gpu_amp_20260608_072949`.
- 2026-06-08 18:53 +08: launched the from-scratch delta-noise MSE ablation in tmux
  `gaze_wam_open_cosmos_delta_noise_8gpu`.
  The intended timestamp suffix was swallowed by local PowerShell command interpolation, so the
  live output directory is the untimestamped:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_`.
  Training is healthy, so the run was not restarted just to rename the directory.
- 2026-06-08 18:59 +08: launched watcher tmux
  `gaze_wam_cosmos_delta_noise_ckpt_preview_watch` for the delta-noise output directory.
- 2026-06-09 13:05 +08 health check: training is healthy in `epoch=85`, about
  `42%` (`2264/5359`) at about `7.0 it/s`; `logs.json.txt` reached `global_step=114466`.
- Latest observed log row had `train_loss=0.00061`, `train_heatmap_loss=0.00061`,
  `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`,
  `train_heatmap_token_kl_loss=0.0`, and `train_action_loss=0.0`.
- Runtime contract confirms:
  - `heatmap_objective=diffusion`
  - `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`
  - `latent_mse_loss=true`
  - `heatmap_xy_loss_weight=0.0`
  - `heatmap_js_loss_weight=0.0`
  - `mixed_precision=bf16`
  - `num_processes=8`
  - `effective_train_batch_size=512`
  - `open_ratio=1.0`
  - `robot_ratio=0.0`
  - `heatmap_dim=16`
  - `heatmap_spatial_decoder=cosmos_tokenizer`
- First delta-noise checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.017.ckpt`.
- Watcher reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260608_190859_epoch=0000-val_loss=0.017`.
- Local pulled preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0000_cosmos_delta_noise_mse`.
- Epoch 0 preview summary confirms `heatmap_prediction_mode=iterative_denoise`,
  `num_inference_steps=8`, `heatmap_distribution_mode=intensity_softplus`,
  `latent_mse_loss=true`, `heatmap_supervision=latent_diffusion_mse_against_frozen_cosmos_target`,
  `use_gaze_condition=false`, and `use_ema=false`.
- Visual finding: pure latent delta-noise MSE removes the DSNT/JS timestep conflict from the
  objective and produces some localized high-response blobs, but epoch 0 is not reliably gaze
  aligned. Sample 0/1 put a strong blob near the target region with extra background speckles;
  sample 2/3 show clear off-target peaks. The decoded maps still contain Cosmos-codec texture,
  border, and blue-background artifacts, so this ablation currently points to codec/target-space
  mismatch as a remaining issue rather than DSNT/JS being the only failure source.
- Epoch 1/2 follow-up checkpoints:
  - `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.010.ckpt`
  - `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.006.ckpt`
- Epoch 1/2 watcher reload previews:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260608_192214_epoch=0001-val_loss=0.010`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0001_cosmos_delta_noise_mse`
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260608_193528_epoch=0002-val_loss=0.006`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0002_cosmos_delta_noise_mse`
- Epoch 1/2 visual finding: validation loss improves quickly and the predicted maps become more
  localized. Sample 0 is consistently near the gaze point by epoch 2, but sample 1/2/3 still show
  off-target peaks or secondary responses. Persistent left-bottom speckles, faint blue background,
  and border/texture artifacts remain visible, so DSNT/JS was a conflict term but the frozen
  Cosmos CI16x16 codec is still not a clean single-channel heatmap representation.
- Epoch 3/4 follow-up checkpoints:
  - `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0003-val_loss=0.005.ckpt`
  - `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0004-val_loss=0.004.ckpt`
- Epoch 3/4 watcher reload previews:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260608_194841_epoch=0003-val_loss=0.005`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0003_cosmos_delta_noise_mse`
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260608_200155_epoch=0004-val_loss=0.004`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0004_cosmos_delta_noise_mse`
- Epoch 4 visual finding: the loss continues to improve, but the qualitative failure is now
  clearer. Sample 0 and sample 3 are close to the gaze point; sample 1 remains displaced to the
  right and sample 2 remains biased toward the lower object region. The fixed left-bottom speckles,
  blue background wash, and faint border artifacts persist despite lower latent MSE, which suggests
  the remaining issue is codec/decoded-intensity calibration rather than the DiT failing to learn
  the latent denoising target.
- Epoch 5 follow-up checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0005-val_loss=0.002.ckpt`
- Epoch 5 watcher reload preview:
  server:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260608_201509_epoch=0005-val_loss=0.002`
  local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0005_cosmos_delta_noise_mse`
- Epoch 5 visual finding: sample 0 and sample 3 remain close to the gaze point, sample 1 is still
  right-shifted, and sample 2 is now mostly suppressed rather than clearly aligned. The same fixed
  left-bottom speckles and blue background wash persist even at `val_loss=0.002`, strengthening the
  diagnosis that pure latent MSE is optimizing the Cosmos latent target while the decoded heatmap
  distribution still carries codec/calibration artifacts.
- Overnight follow-up: the run continued through many epochs. Named retained checkpoints include
  epoch 6/7/8/10/12/13/16/17 at about `val_loss=0.002`, later retained checkpoints around
  `val_loss=0.003`, and the newest named retained checkpoint
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0062-val_loss=0.003.ckpt`.
- Latest watcher reload preview pulled locally:
  - server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_114844_latest`
  - local:
    `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260609_114844_cosmos_delta_noise_mse`
- Latest visual finding: by the `latest.ckpt` preview around epoch 79, all four inspected samples
  put the main predicted peak near the gaze point, which is a real improvement over epoch 0-5.
  However, the fixed left-bottom speckle cluster, blue background wash, and faint border/texture
  artifacts remain visible. The current diagnosis is therefore stronger: latent delta-noise MSE
  successfully trains the DiT stream, but frozen Cosmos CI16x16 still leaves decoded single-channel
  heatmap artifacts and is likely the remaining bottleneck.
- Epoch 81/83 and latest follow-up previews:
  - epoch 81 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_122523_epoch=0081-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0081_cosmos_delta_noise_mse`
  - epoch 83 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_125146_epoch=0083-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0083_cosmos_delta_noise_mse`
  - latest server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_130457_latest`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260609_130457_cosmos_delta_noise_mse`
- Latest 20260609_130457 visual finding: this is still an 8-step iterative denoise preview.
  Sample 0 and sample 3 are close to gaze; sample 1 remains slightly left/right ambiguous around
  the hand/object region; sample 2 is visibly above the gaze point. The same left-bottom speckles
  and blue background wash remain. Training longer has not removed the decoded artifacts, and the
  sample-level alignment is not monotonic despite very low latent MSE.
- Heartbeat check on 2026-06-09 13:53 +08: training stayed healthy and advanced to epoch 89;
  `logs.json.txt` reached `global_step=119307`. Recent log rows kept the intended open-only
  contract (`train_action_loss=0.0`, `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`,
  `train_routing_open_rows=512`, `train_routing_use_gaze_condition_rows=0`) with heatmap latent
  MSE fluctuating around `3e-4` to `7e-4`.
- New watcher previews pulled locally:
  - epoch 85 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_131808_epoch=0085-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0085_cosmos_delta_noise_mse`
  - latest server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_134432_latest`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260609_134432_cosmos_delta_noise_mse`
- Latest 20260609_134432 visual finding: all four inspected samples show a compact main predicted
  peak near the gaze marker, so the 8-step latent denoising behavior is still broadly aligned.
  However, samples 1/2/3 remain visibly offset by a small local amount, and the same fixed
  lower-left speckle cluster plus blue background wash persists. This reinforces the current
  diagnosis that longer training is not removing the frozen Cosmos CI16x16 decoded-intensity
  artifacts.
- First-step denoise diagnostic on 2026-06-09 13:43 +08:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/diagnostics/first_step_latest_20260609_`
  was pulled to
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\first_step_latest_20260609_cosmos_delta_noise_mse`.
  It confirms that the first update of the 8-step schedule (`t=42` in
  `[42,36,30,24,18,12,6,0]`) still decodes almost like the random initial Cosmos latent; the
  localized gaze peak forms by the final step rather than appearing after the first update.
- Heartbeat check on 2026-06-09 14:28 +08: training stayed healthy in epoch 91, about `71%`
  (`3800/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=122951`.
  Recent rows still match the active ablation contract: 100% open rows, no action loss, no
  DSNT/JS/xy loss, and pure heatmap latent MSE around `3e-4` to `8e-4`.
- New watcher previews pulled locally:
  - epoch 88 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_135443_epoch=0088-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0088_cosmos_delta_noise_mse`
  - epoch 89 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_140754_epoch=0089-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0089_cosmos_delta_noise_mse`
  - epoch 90 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_142110_epoch=0090-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0090_cosmos_delta_noise_mse`
- Epoch 89 reached the lowest scalar validation loss in this group (`val_loss=0.002`), but the
  qualitative result is not materially different from epoch 90. The four inspected samples keep
  compact predicted peaks near the gaze point; sample 1 and sample 2 remain noticeably offset, and
  sample 0/3 have smaller offsets. The fixed lower-left speckles and blue background wash persist.
  Longer training is therefore continuing to optimize latent MSE without removing the frozen
  Cosmos decoded-intensity artifacts.
- Heartbeat check on 2026-06-09 15:03 +08: training stayed healthy in epoch 94, about `44%`
  (`2336/5359`) at roughly `7.0 it/s`; `logs.json.txt` reached `global_step=126772`.
  Recent rows still show pure open-data latent MSE training with `train_action_loss=0.0`,
  `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, and
  `train_routing_use_gaze_condition_rows=0`.
- New watcher previews pulled locally:
  - latest server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_143424_latest`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_latest_20260609_143424_cosmos_delta_noise_mse`
  - epoch 92 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_144736_epoch=0092-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0092_cosmos_delta_noise_mse`
  - epoch 93 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_150048_epoch=0093-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0093_cosmos_delta_noise_mse`
- Epoch 93 visual finding: scalar `val_loss=0.002` matches the better retained checkpoints, but
  the qualitative pattern is unchanged. The predicted peaks remain compact and near the gaze
  region, but all four inspected samples show some local offset, with sample 1/2/3 especially
  right/down shifted. The lower-left speckle cluster and blue wash are still fixed, so this run has
  not escaped the frozen Cosmos CI16x16 decoded-intensity artifact pattern.
- Heartbeat check on 2026-06-09 15:38 +08: training stayed healthy in epoch 97, about `27%`
  (`1440/5359`) at roughly `6.95 it/s`; `logs.json.txt` reached `global_step=130428`.
  Recent rows remain pure open-data latent MSE training with `train_action_loss=0.0`,
  `train_heatmap_xy_loss=0.0`, `train_heatmap_js_loss=0.0`, and
  `train_routing_use_gaze_condition_rows=0`.
- New watcher previews pulled locally:
  - epoch 94 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_151404_epoch=0094-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0094_cosmos_delta_noise_mse`
  - epoch 95 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_152718_epoch=0095-val_loss=0.003`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0095_cosmos_delta_noise_mse`
  - epoch 96 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_153736_epoch=0096-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0096_cosmos_delta_noise_mse`
- Epoch 96 visual finding: summary still confirms 8-step iterative denoise, frozen Cosmos
  CI16x16, `intensity_softplus`, `heatmap_latent_scale=0.25`, scheduler clipping, no EMA, and
  no gaze condition. Qualitatively it is nearly identical to epoch 93: compact peaks, no collapse,
  but persistent local offsets on all four samples and especially sample 1/2/3. The lower-left
  speckle cluster and blue background wash are unchanged.
- Heartbeat check on 2026-06-09 16:13 +08: training stayed alive but slowed sharply. The tmux pane
  showed epoch 98 at about `47%` (`2540/5359`) and only about `1.9 it/s`; `logs.json.txt` reached
  `global_step=131982`. Recent rows still show the intended pure open-data latent MSE contract.
- Resource anomaly: `nvidia-smi` reported all GPUs at about `87GB/98GB` and 100% utilization.
  The compute-app list showed normal training-rank allocations around `12.6-13.0GB`, plus a
  `[Not Found]` PID `1748481` holding about `74.6GB` on every GPU. No corrective action was taken
  because training and the watcher are still alive, but this likely explains the speed drop from
  the earlier roughly `7 it/s`.
- New watcher preview pulled locally:
  - epoch 97 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_155354_epoch=0097-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0097_cosmos_delta_noise_mse`
- Epoch 97 visual finding: summary remains the same active contract (`iterative_denoise`,
  `num_inference_steps=8`, `intensity_softplus`, `latent_mse_loss=true`, `use_gaze_condition=false`,
  `use_ema=false`). The images remain qualitatively unchanged from epoch 93/96: compact peaks near
  gaze, persistent offsets on all four samples, and fixed lower-left speckles/blue background wash.
- Heartbeat check on 2026-06-09 16:48 +08: training stayed alive but remained slow. The tmux pane
  showed epoch 99 at about `27%` (`1470/5359`) and about `1.9 it/s`; `logs.json.txt` reached
  `global_step=133056`. Recent rows still match the pure open-data latent MSE contract.
- The resource anomaly persisted: all GPUs stayed around `87GB/98GB` and 100% utilization.
  `nvidia-smi` again showed normal training-rank allocations around `12.6-13.0GB`, plus the
  `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was taken.
- New watcher preview pulled locally:
  - epoch 98 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_164015_epoch=0098-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0098_cosmos_delta_noise_mse`
- Epoch 98 visual finding: same active contract and same qualitative behavior as epoch 97. The
  predicted peaks remain compact and near gaze, but sample 1/2 are still visibly offset and
  sample 0/3 retain smaller offsets. The lower-left speckle cluster and blue background wash are
  unchanged.
- Heartbeat check on 2026-06-09 17:28 +08: training stayed alive but remained slow. The tmux pane
  showed epoch 100 at about `26%` (`1376/5359`) and about `1.8-2.0 it/s`.
- The resource anomaly persisted: all GPUs were around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` showed normal training ranks around `12.6-13.0GB`, plus the
  `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was taken.
- New watcher preview pulled locally:
  - epoch 99 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_172636_epoch=0099-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0099_cosmos_delta_noise_mse`
- Epoch 99 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `num_inference_steps=8`, `heatmap_dim=16`,
  `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 99 visual finding: same qualitative pattern as epoch 98. Predicted peaks are compact and
  usually near the gaze region, but sample 1/2 remain visibly offset while sample 0/3 have smaller
  offsets. The fixed lower-left speckle cluster and blue background wash are unchanged, so this
  still points to frozen Cosmos CI16x16 decoded-intensity artifacts rather than a simple
  undertraining problem.
- Heartbeat check on 2026-06-09 18:33 +08: training stayed alive and advanced to epoch 101. The
  tmux pane showed epoch 101 at about `56%` (`3018/5359`) and about `1.9-2.0 it/s`;
  `logs.json.txt` reached `global_step=136126`. Recent rows still match the pure open-data latent
  MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` again showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 100 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_180953_epoch=0100-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0100_cosmos_delta_noise_mse`
- Epoch 100 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 100 visual finding: no qualitative improvement over epoch 99. The predictions remain
  compact and non-collapsed, but sample 1/2 are still visibly offset and sample 0/3 have smaller
  local offsets. The fixed lower-left speckle cluster and blue background wash persist unchanged,
  further supporting a frozen Cosmos CI16x16 heatmap codec/decoded-intensity bottleneck.
- Heartbeat check on 2026-06-09 19:08 +08: training stayed alive and advanced to epoch 102. The
  tmux pane showed epoch 102 at about `35%` (`1878/5359`) and about `1.8-1.9 it/s`;
  `logs.json.txt` reached `global_step=137148`. Recent rows still match the pure open-data latent
  MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` again showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 101 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_185616_epoch=0101-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0101_cosmos_delta_noise_mse`
- Epoch 101 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 101 visual finding: still no qualitative improvement over epoch 99/100. The heatmap peaks
  remain compact and non-collapsed, but sample 1/2 are visibly offset and sample 0/3 retain smaller
  local offsets. The lower-left speckle cluster and blue wash remain fixed, so the current
  late-epoch evidence continues to favor changing the heatmap codec/target space rather than
  training this frozen Cosmos CI16x16 objective longer.
- Heartbeat check on 2026-06-09 19:43 +08: training stayed alive and advanced to epoch 103. The
  tmux pane showed epoch 103 at about `14%` (`751/5359`) and about `1.8-1.9 it/s`;
  `logs.json.txt` reached `global_step=138206`. Recent rows still match the pure open-data latent
  MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` again showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 102 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_193952_epoch=0102-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0102_cosmos_delta_noise_mse`
- Epoch 102 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 102 visual finding: unchanged from epoch 100/101. Sample 0 and sample 3 keep small local
  offsets; sample 1 remains offset toward the lower-left object region; sample 2 remains strongly
  down-shifted. The fixed lower-left speckle cluster and blue background wash persist, reinforcing
  that this late-stage run is stable but bottlenecked by the frozen Cosmos heatmap representation.
- Heartbeat check on 2026-06-09 20:53 +08: training stayed alive and advanced to epoch 104. The
  tmux pane showed epoch 104 at about `68%` (`3664/5359`) and about `1.9-2.0 it/s`;
  `logs.json.txt` reached `global_step=140275`. Recent rows still match the pure open-data latent
  MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` again showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 103 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_202606_epoch=0103-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0103_cosmos_delta_noise_mse`
- Epoch 103 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 103 visual finding: unchanged from epoch 102. Sample 0 and sample 3 retain small offsets;
  sample 1 remains locked onto the lower-left cup/hand region rather than the gaze marker; sample 2
  remains strongly down-shifted. The fixed lower-left speckles and blue wash persist, so additional
  epochs are not changing the qualitative failure mode.
- Heartbeat check on 2026-06-09 21:28 +08: training stayed alive and advanced to epoch 105. The
  tmux pane showed epoch 105 at about `46%` (`2447/5359`) and about `1.9 it/s`;
  `logs.json.txt` reached `global_step=141311`. Recent rows still match the pure open-data latent
  MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` again showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 104 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_210919_epoch=0104-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0104_cosmos_delta_noise_mse`
- Epoch 104 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 104 visual finding: unchanged from epoch 103. Sample 0/3 keep small local offsets; sample 1
  stays locked to the lower-left cup/hand region; sample 2 stays strongly down-shifted. The fixed
  lower-left speckles and blue background wash remain unchanged, so the late-epoch qualitative
  behavior has plateaued.
- Heartbeat check on 2026-06-09 22:03 +08: training stayed alive and advanced to epoch 106. The
  tmux pane showed epoch 106 at about `24%` (`1308/5359`), and speed recovered to roughly
  `5-6 it/s`; `logs.json.txt` reached `global_step=142366`. Recent rows still match the pure
  open-data latent MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB`, with utilization around
  `93-97%`. `nvidia-smi --query-compute-apps` still showed normal training ranks around
  `12.6-13.0GB`, plus the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU.
  No corrective action was taken.
- New watcher preview pulled locally:
  - epoch 105 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_215533_epoch=0105-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0105_cosmos_delta_noise_mse`
- Epoch 105 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 105 visual finding: unchanged from epoch 104. Sample 0/3 keep small local offsets; sample 1
  remains locked to the lower-left cup/hand region; sample 2 remains strongly down-shifted. The
  fixed lower-left speckles and blue wash persist, so the recovered training speed did not change
  the qualitative plateau.
- Heartbeat check on 2026-06-09 22:38 +08: training stayed alive and advanced to epoch 107. The
  tmux pane showed epoch 107 at about `5%` (`242/5359`) and about `1.9 it/s`;
  `logs.json.txt` reached `global_step=143439`. Recent rows still match the pure open-data latent
  MSE contract.
- The watcher initially lagged behind the already-saved `epoch=0106` checkpoint, then produced the
  epoch 106 preview after a short wait. No watcher failure was observed.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 106 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_224147_epoch=0106-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0106_cosmos_delta_noise_mse`
- Epoch 106 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 106 visual finding: unchanged from epoch 105. Sample 0/3 keep small local offsets; sample 1
  remains locked to the lower-left cup/hand region; sample 2 remains strongly down-shifted. The
  fixed lower-left speckles and blue background wash persist.
- Heartbeat check on 2026-06-09 23:48 +08: training stayed alive and advanced to epoch 108. The
  tmux pane showed epoch 108 at about `61%` (`3270/5359`) and about `2.0 it/s`;
  `logs.json.txt` reached `global_step=145537`. Recent rows still match the pure open-data latent
  MSE contract.
- The resource anomaly persisted: all GPUs were still around `87GB/98GB` and 100% utilization.
  `nvidia-smi --query-compute-apps` still showed normal training ranks around `12.6-13.0GB`, plus
  the `[Not Found]` PID `1748481` holding about `74.6GB` on each GPU. No corrective action was
  taken.
- New watcher preview pulled locally:
  - epoch 107 server:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260609_232502_epoch=0107-val_loss=0.002`
    local: `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0107_cosmos_delta_noise_mse`
- Epoch 107 summary confirms the same active preview contract: `heatmap_objective=diffusion`,
  `latent_mse_loss=true`, `heatmap_prediction_mode=iterative_denoise`, `num_inference_steps=8`,
  `heatmap_dim=16`, `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `heatmap_distribution_mode=intensity_softplus`, `use_gaze_condition=false`, and `use_ema=false`.
- Epoch 107 visual finding: unchanged from epoch 106. Sample 0/3 keep small local offsets, sample 1
  remains fixed off-target, and sample 2 remains strongly down-shifted. The lower-left speckles and
  blue background wash persist.

## Previous DSNT/JS Intensity-Softplus Run

- 2026-06-08 15:29 +08: stopped the old scaled/logits training tmux
  `gaze_wam_open_cosmos_scaled_8gpu` and watcher `gaze_wam_cosmos_scaled_ckpt_preview_watch`.
  Its output directory is preserved for comparison:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_scaled_latent_8gpu_amp_20260608_040418`.
- 2026-06-08 15:29 +08: launched a fresh from-scratch 8-GPU AMP run in tmux
  `gaze_wam_open_cosmos_intensity_8gpu`.
- 2026-06-08 15:30 +08: launched the checkpoint-preview watcher
  `gaze_wam_cosmos_intensity_ckpt_preview_watch` for the new output directory.
- 2026-06-08 15:34 +08 health check: training is healthy in `epoch=0`, about
  `17%` (`932/5359`) at about `5.18-5.20 it/s`.
- `logs.json.txt` reached `global_step=233`; latest observed row had
  `train_loss=0.4099`, `train_heatmap_xy_loss=0.0655`, `train_heatmap_js_loss=0.3444`,
  and `train_action_loss=0.0`.
- Early loss sanity: step 0 was `train_loss=0.7758`, while step 152 was `0.4150`;
  the last-50-step mean at that check was about `0.5112`.
- Routing confirms open-only training: `train_routing_open_rows=512`,
  `train_routing_robot_rows=0`, `train_routing_open_heatmap_loss_count=512`,
  `train_routing_has_gaze_label_rows=512`, and `train_routing_use_gaze_condition_rows=0`.
- GPU health: all 8 H20 GPUs were active at about `95-97%` utilization, with about
  `16.4-16.6 GB` memory per GPU.
- Runtime contract confirms:
  - `canonical_main_config_ok=true`
  - `heatmap_distribution_mode_intensity_softplus=true`
  - `heatmap_dsnt_temperature_0p1=true`
  - `heatmap_spatial_decoder_cosmos_tokenizer=true`
  - `heatmap_objective_dsnt_js=true`
  - `heatmap_scheduler_clip_sample_enabled=true`
  - `open_ratio=1.0`
  - `robot_ratio=0.0`
  - `heatmap_num_tokens=256`
  - `heatmap_dim=16`
- First checkpoint for the new run now exists:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_intensity_softplus_latent_8gpu_amp_20260608_072949/checkpoints/epoch=0000-val_loss=0.224.ckpt`.
- 2026-06-08 15:39 +08 follow-up: training remained healthy in `epoch=0`, about
  `50%` (`2690/5359`) at about `5.18-5.20 it/s`; `logs.json.txt` reached
  `global_step=671`. Latest observed row had `train_loss=0.3113`,
  `train_heatmap_xy_loss=0.0521`, `train_heatmap_js_loss=0.2592`, and
  `train_action_loss=0.0`. Watcher is still waiting for the first checkpoint.
- 2026-06-08 16:48 +08 heartbeat check: training is healthy in `epoch=2`, about
  `80%` (`4288/5359`) at about `5.21-5.23 it/s`; `logs.json.txt` reached
  `global_step=3750`. Latest observed row had `train_loss=0.2580`,
  `train_heatmap_xy_loss=0.0404`, `train_heatmap_js_loss=0.2176`, and
  `train_action_loss=0.0`.
- The run has saved epoch 0 and epoch 1 checkpoints:
  - `epoch=0000-val_loss=0.224.ckpt`
  - `epoch=0001-val_loss=0.198.ckpt`
- New watcher reload preview was produced for epoch 1:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_intensity_softplus_latent_8gpu_amp_20260608_072949/media/ckpt_heatmap/watched/20260608_163629_epoch=0001-val_loss=0.198`.
- Pulled local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\server_preview_epoch0001_cosmos_intensity_softplus`.
- Epoch 1 preview summary:
  - `heatmap_prediction_mode=iterative_denoise`
  - `num_inference_steps=8`
  - `heatmap_distribution_mode=intensity_softplus`
  - `heatmap_latent_scale=0.25`
  - `heatmap_scheduler_clip_sample=true`
  - `latent_mse_loss=false`
  - `use_gaze_condition=false`
  - `use_ema=false` because the checkpoint has no EMA weights
- Visual finding: epoch 1 loss improves over epoch 0, but predicted heatmaps still show strong
  scene-texture and block-like codec artifacts. The images support the current diagnosis that the
  failure is more likely codec/loss calibration than direct RGB leakage. The local preview renderer
  has been updated to split pure heatmap and RGB overlay panels, and the update was synced to the
  server for future previews.

## Previous Scaled/Logits Run

- 2026-06-08 14:23 +08 heartbeat check: active scaled-Cosmos training is healthy in
  `epoch=7`, about `96%` (`5170/5359`) at about `5.18-5.19 it/s`; `logs.json.txt`
  reached `global_step=10672`.
- Current checkpoints for the active scaled run:
  - `epoch=0000-val_loss=0.327.ckpt`
  - `epoch=0005-val_loss=0.256.ckpt`
  - `latest.ckpt`
- The checkpoint watcher produced the automatic epoch 5 reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_scaled_latent_8gpu_amp_20260608_040418/media/ckpt_heatmap/watched/20260608_135023_epoch=0005-val_loss=0.256`.
- Local pulled copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_scaled_watched_epoch0005_20260608_135023`.
- Epoch 5 watcher summary: `source=open`, `split=val`, `num_samples=4`,
  `num_inference_steps=8`, `heatmap_prediction_mode=iterative_denoise`,
  `heatmap_latent_scale=0.25`, `heatmap_scheduler_clip_sample=true`,
  `use_gaze_condition=false`, and `use_ema=false` because the checkpoint has no EMA weights.
- Visual finding for the active scaled run: the epoch 5 watcher preview matches the manual reload
  diagnosis. It no longer shows the old hard `16x16` patch-seam/checkerboard failure, but predicted
  heatmaps still contain vertical/scene-texture artifacts and broad salient-region activation.
  Several samples place energy near the gaze target, yet the maps are still too diffuse for a
  converged intent heatmap.
- Codec-only diagnostic from the same checkpoint window found a likely loss-calibration issue:
  frozen Cosmos E-D preserves heatmap peak location to roughly 1-2 px, but treating decoder output
  as logits with `spatial_softmax(temperature=1.0)` makes the distribution too flat. A temperature
  sweep improved round-trip loss substantially near `temperature=0.1`, so the next recommended
  ablation is a short scaled-Cosmos run with lower heatmap DSNT/JS temperature before abandoning the
  Cosmos codec path.
- The old pre-scale Cosmos run, the early `0.2857142857142857` scaled run, and the project-local
  heatmap-autoencoder run were stopped and their output directories were deleted from server1024.
- Training is running in tmux `gaze_wam_open_cosmos_scaled_8gpu`.
- Last observed progress: epoch 0, `14/5359` iterations, initial speed still warming up.
- Latest log rows show `train_action_loss=0.0`, `train_heatmap_loss鈮?.81`,
  `train_routing_open_rows=512`, `train_routing_robot_rows=0`.
- Runtime contract confirms:
  - `canonical_main_config_ok=true`
  - `num_processes=8`
  - `mixed_precision=bf16`
  - `require_amp=true`
  - `open_ratio=1.0`
  - `robot_ratio=0.0`
  - `heatmap_num_tokens=256`
  - `heatmap_dim=16`
  - `heatmap_spatial_decoder=cosmos_tokenizer`
  - `heatmap_latent_codec=frozen_cosmos_tokenizer`
  - `heatmap_latent_scale=0.25`
  - `heatmap_latent_offset=0.0`
  - `heatmap_scheduler_clip_sample=true`
- Watcher is waiting for the first checkpoint; no preview exists yet for this corrected run.

## Cosmos Epoch 0 Preview

- Checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/checkpoints/epoch=0000-val_loss=0.248.ckpt`
- Validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/val_heatmap/epoch_0000`
- Manual checkpoint reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/ckpt_heatmap/watched/manual_epoch0000_cosmos`
- Local copy:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_policy_ckpt_epoch0000`
- Summary:
  - `heatmap_prediction_mode=iterative_denoise`
  - `num_inference_steps=8`
  - `heatmap_dim=16`
  - `heatmap_image_size=[256,256]`
  - `use_gaze_condition=false`
  - selected validation indices `[6600,1341,9843,11637]`
- Visual finding: the Cosmos path produces decoded full-resolution heatmaps without the old hard
  patch seams. At epoch 0 the maps still show strong global scene/edge activation and fine texture
  noise, with some mass near the target gaze point. This looks like an early semantic/scale learning
  issue rather than the previous deterministic token-boundary artifact.

## Cosmos Epoch 5 Preview

- Checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/checkpoints/epoch=0005-val_loss=0.135.ckpt`
- Validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/val_heatmap/epoch_0005`
- Watcher reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/ckpt_heatmap/watched/20260608_020804_epoch=0005-val_loss=0.135`
- Manual reload preview after code fix:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/ckpt_heatmap/watched/manual_epoch0005_cosmos_fixed`
- Local copies:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_policy_val_epoch0005`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_policy_ckpt_epoch0005_manual_fixed`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_policy_ckpt_epoch0005_watched_20260608_020804`
- Summary:
  - `heatmap_prediction_mode=iterative_denoise`
  - `num_inference_steps=8`
  - `heatmap_dim=16`
  - `heatmap_image_size=[256,256]`
  - `use_gaze_condition=false`
  - selected validation indices `[6600,1341,9843,11637]`
  - `use_ema=false` because this run has no EMA weights
- Visual finding: the epoch 5 Cosmos path does not show the old hard `16x16` block seam pattern.
  It still shows strong vertical/scene-texture artifacts and large, diffuse high-response regions.
  This is a different residual artifact pattern from the previous deterministic patch-boundary
  issue, and more training is needed before claiming the heatmap branch has converged.

## Cosmos Epoch 10 Preview

- Checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/checkpoints/epoch=0010-val_loss=0.125.ckpt`
- Validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/val_heatmap/epoch_0010`
- Watcher reload preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/media/ckpt_heatmap/watched/20260608_032115_epoch=0010-val_loss=0.125`
- Local copies:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_policy_val_epoch0010`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\cosmos_policy_ckpt_epoch0010_watched_20260608_032115`
- Summary:
  - `heatmap_prediction_mode=iterative_denoise`
  - `num_inference_steps=8`
  - `heatmap_dim=16`
  - `heatmap_image_size=[256,256]`
  - `use_gaze_condition=false`
  - selected validation indices `[6600,1341,9843,11637]`
  - `use_ema=false` because this run has no EMA weights
  - epoch 10 validation loss improved to `0.125`, from epoch 5 `0.135`
- Visual finding: the epoch 10 Cosmos path still does not show the old hard `16x16` block seam or
  checkerboard-boundary failure. Compared with epoch 5, the watcher preview is more centered near
  the target gaze point. Residual issues remain: broad/diffuse activation, scene-texture imprinting,
  edge noise, and blue/green border artifacts are still visible in some samples. These residuals are
  not the same failure mode as the removed project-local ED path.

## Current Preview Watcher State

- The watcher is running and watching:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_tokenizer_8gpu_amp_20260607_160951/checkpoints`
- Latest check on 2026-06-08 02:18 +08: training is healthy in epoch 7.
  - tmux pane showed epoch 7 at about `34%` (`1841/5359`) and about `5.22-5.23 it/s`.
  - `logs.json.txt` reached `global_step=9935`, `epoch=7`.
  - Latest per-step losses were in the expected open-only heatmap range, e.g.
    `train_loss=0.095`, `train_heatmap_xy_loss=0.0146`, `train_heatmap_js_loss=0.0806`.
  - Routing confirms open-only training: `train_routing_open_rows=512`,
    `train_routing_robot_rows=0`, `train_routing_use_gaze_condition_rows=0`,
    `train_routing_dropped_gaze_condition_rows=512`.
  - No new named checkpoint beyond epoch 5 yet; the next required visual decision point remains
    the epoch 10 checkpoint/watcher preview.
- Follow-up check on 2026-06-08 02:20 +08: training was still healthy in epoch 7, now about
  `52%` (`2771/5359`) at `5.23-5.24 it/s`; `logs.json.txt` reached `global_step=10073`.
  Checkpoints and watcher previews were unchanged: epoch 0, epoch 5, and `latest.ckpt` only.
- Follow-up check on 2026-06-08 02:22 +08: training remained healthy in epoch 7, about
  `65%` (`3461/5359`) at `5.23-5.24 it/s`; `logs.json.txt` reached `global_step=10244`.
  Checkpoints and watcher previews were still unchanged, so there was no new comparison image
  to pull.
- Follow-up check on 2026-06-08 02:24 +08: `logs.json.txt` was still updating at
  `global_step=10418`, `epoch=7`. The tmux capture command was interrupted by the known SSH reset,
  but checkpoint and log inspection succeeded. Checkpoints remained epoch 0, epoch 5, and
  `latest.ckpt`; watcher summaries remained epoch 0/5 only.
- Follow-up check on 2026-06-08 02:26 +08: `logs.json.txt` reached `global_step=10551`,
  still in `epoch=7`. Checkpoints remained unchanged (`epoch=0000`, `epoch=0005`,
  `latest.ckpt`), and watcher summaries remained epoch 0/5 only.
- Follow-up check on 2026-06-08 02:28 +08: `logs.json.txt` reached `global_step=10683`,
  still in `epoch=7`. Checkpoints and watcher summaries remained unchanged, so no new
  comparison images were available.
- Follow-up check on 2026-06-08 02:30 +08: training entered `epoch=8` and `logs.json.txt`
  reached `global_step=10814`. Checkpoints remained epoch 0, epoch 5, and `latest.ckpt`;
  watcher summaries remained epoch 0/5 only.
- Follow-up check on 2026-06-08 02:32 +08: training continued in `epoch=8` and
  `logs.json.txt` reached `global_step=11013`. Checkpoints and watcher summaries remained
  unchanged; no new comparison images were available.
- Follow-up check on 2026-06-08 02:34 +08: training continued in `epoch=8` and
  `logs.json.txt` reached `global_step=11172`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:36 +08: training continued in `epoch=8` and
  `logs.json.txt` reached `global_step=11349`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:38 +08: training continued in `epoch=8` and
  `logs.json.txt` reached `global_step=11494`. Checkpoints remained epoch 0, epoch 5, and
  `latest.ckpt`; watcher summaries remained epoch 0/5.
- Follow-up check on 2026-06-08 02:40 +08: training continued in `epoch=8` and
  `logs.json.txt` reached `global_step=11637`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:43 +08: training continued in `epoch=8` and
  `logs.json.txt` reached `global_step=11883`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:46 +08: training entered `epoch=9` and `logs.json.txt`
  reached `global_step=12159`. Checkpoints remained epoch 0, epoch 5, and `latest.ckpt`;
  watcher summaries remained epoch 0/5. The next expected named checkpoint remains epoch 10.
- Follow-up check on 2026-06-08 02:50 +08: training continued in `epoch=9` and
  `logs.json.txt` reached `global_step=12473`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:53 +08: training continued in `epoch=9` and
  `logs.json.txt` reached `global_step=12662`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:55 +08: training continued in `epoch=9` and
  `logs.json.txt` reached `global_step=12834`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:57 +08: training continued in `epoch=9` and
  `logs.json.txt` reached `global_step=12964`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 02:59 +08: training continued in `epoch=9` and
  `logs.json.txt` reached `global_step=13108`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 03:01 +08: training continued in `epoch=9` and
  `logs.json.txt` reached `global_step=13256`. Checkpoints and watcher summaries remained
  unchanged.
- Follow-up check on 2026-06-08 03:03 +08: training entered `epoch=10`. The tmux pane showed
  about `5%` progress (`250/5359`) at about `5.23 it/s`, with roughly 16 minutes of epoch training
  remaining before validation/checkpoint save. Checkpoints and watcher summaries had not updated
  yet.
- Follow-up check on 2026-06-08 03:05 +08: `epoch=10` reached about `16%` (`838/5359`) at
  about `5.22-5.23 it/s`, with roughly 14 minutes of epoch training remaining. Checkpoints and
  watcher summaries had not updated yet.
- Follow-up check on 2026-06-08 03:07 +08: `epoch=10` reached about `28%` (`1480/5359`) at
  about `5.21-5.23 it/s`, with roughly 12 minutes of epoch training remaining. Checkpoints and
  watcher summaries had not updated yet.
- Follow-up check on 2026-06-08 03:09 +08: `epoch=10` reached about `39%` (`2116/5359`) at
  about `5.22-5.23 it/s`, with roughly 10 minutes of epoch training remaining. Checkpoints and
  watcher summaries had not updated yet.
- Follow-up check on 2026-06-08 03:11 +08: `epoch=10` reached about `53%` (`2818/5359`) at
  about `5.22-5.23 it/s`, with roughly 8 minutes of epoch training remaining. Checkpoints and
  watcher summaries had not updated yet.
- Follow-up check on 2026-06-08 03:13 +08: `epoch=10` reached about `63%` (`3352/5359`) at
  about `5.22-5.23 it/s`, with roughly 6 minutes of epoch training remaining. Checkpoints and
  watcher summaries had not updated yet.
- Follow-up check on 2026-06-08 03:15 +08: `epoch=10` reached about `73%` (`3892/5359`) at
  about `5.23 it/s`, with roughly 4-5 minutes of epoch training remaining. Checkpoints and watcher
  summaries had not updated yet.
- Follow-up check on 2026-06-08 03:17 +08: `epoch=10` reached about `83%` (`4438/5359`) at
  about `5.22-5.24 it/s`, with roughly 3 minutes of epoch training remaining. Checkpoints and
  watcher summaries had not updated yet.
- Follow-up check on 2026-06-08 03:25 +08: epoch 10 completed, the run advanced to epoch 11, and
  both validation and watcher reload previews for epoch 10 were pulled locally and inspected. The
  active decoder-retrain question is resolved: Cosmos mitigates the old hard block/checkerboard
  artifact, while leaving softer texture/border artifacts to improve in later training or codec
  ablations.
- The original watcher exited after a checkpoint reload failure caused by a deprecated policy kwarg
  leaking into `DDIMScheduler.step`.
- `GazeWamPolicy` now filters constructor `**kwargs` to scheduler-supported `step` kwargs only.
- After the fix, manual epoch 5 reload preview succeeded and a restarted watcher also produced
  the automatic epoch 5 reload preview above.

## Local Code Contract

- Active policy path only accepts `policy.heatmap_spatial_decoder=cosmos_tokenizer`.
- Project-local heatmap ED training entrypoint has been removed:
  `diffusion_policy/scripts/train_heatmap_autoencoder_codec.py` no longer exists.
- Main open-only configs and training scripts default to the frozen Cosmos tokenizer paths.
- Server verification passed after cleanup:
  - `py_compile` for `gaze_wam_policy.py`, `heatmap_decoder.py`, and
    `download_cosmos_tokenizer.py`.
  - filtered code search found no active self-trained heatmap ED route.

## Archived Comparisons

- Learned decoder run:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_learned_latent_fullres_8gpu_amp_20260607_003811`
- Project-local heatmap autoencoder run:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_frozen_heatmap_autoencoder_8gpu_amp_20260607_124838`
- Both are historical comparisons only. The current implementation contract removes the self-trained
  heatmap ED path from active policy training.





