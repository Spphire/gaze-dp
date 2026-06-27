# Gaze-WAM Temporal-Window Experiment Report

Date: 2026-06-15

## Summary

The previous single-point Gaussian clean-label baseline completed 200 epochs and was numerically
stable, but its decoded heatmaps plateaued with fixed spatial offsets plus persistent
scene-texture/checkerboard-like artifacts. The pure temporal-window label experiment has evidence
through the epoch 28 `latest.ckpt` preview that the clean label is usable as a DiT final target,
while the active mix-LNLL follow-up now has mismatched-budget trend evidence through
`latest/epoch48`. The pure frozen-Cosmos
latent diffusion/noise MSE objective still leaves a scene-prior failure mode. Evidence now extends
through the epoch 28 `latest.ckpt` preview: epoch 28 is numerically worse than epoch 27
(`val_loss=0.005265417438931763` vs `0.004825264634564519`) and still fails the same hard
scene-texture sample. The new experiment keeps the
same 8-GPU frozen-Cosmos latent diffusion training path, but replaces the clean heatmap label with a
temporal-window dense target:

```text
q_t(x) = normalize(sum_{tau=t-K}^{t+K} exp(-abs(tau-t)/beta) * Gaussian(x; gaze_tau, sigma))
```

The completed temporal-window latent-MSE run is not a mixed decoded-image loss experiment. The
only intended change in that run is the clean `H0` label. The heatmap stream still trains with
pure latent diffusion/noise MSE:
`heatmap_objective=diffusion`, `latent_mse_loss=true`, `heatmap_xy_loss_weight=0.0`, and
`heatmap_js_loss_weight=0.0`.

The first follow-up ablation used a true mixed objective:
`diffusion noise/latent MSE + decoded final heatmap XY/JS loss`. It keeps the same temporal-window
clean label, enables `policy.heatmap_diffusion_final_loss_enabled=true`, and uses small decoded
loss weights (`XY=0.05`, `JS=0.10`) with `alpha_cumprod` timestep weighting so the final
distribution constraint is strongest near low-noise denoising steps. Evidence now reaches epoch 16:
validation loss improved from `0.031` to `0.024` to `0.018` to `0.016` to `0.013`, then plateaued
around `0.012-0.015` for epochs 5-16. Checkpoint and full episode 86 previews are available, and
the predictions are clearly tighter than epoch 0. Mixed epoch 16 recovers scalar loss to `0.013`;
easier stills and episode frames remain usable, but the direct pulled-output A/B still does not
surpass the pure temporal epoch 28 evidence because sample 2 and episode 86 frame 0 still show
scene-geometry/object-lock behavior.

Visual evidence supports the label choice: the temporal target is clean, current-frame
weighted, and temporally smooth enough to serve as a DiT final clean label. Validation loss improves
quickly from `0.018` to `0.011` to `0.008` to `0.007` to `0.004`, then stays around
`0.003-0.006` through epoch 28. Epochs 5-28 fix one hard sampled failure, while other hard cases
remain non-monotonic. Model convergence is not yet proven because some frames still show
off-target object/texture locks; epoch 22 is slightly better than epoch 21, epoch 23 slightly
regresses, and the epoch 24-28 `latest.ckpt` previews improve the episode stills while still
failing the same hard sample 2. Mixed-loss epoch 16 did not resolve that hard sample/frame-0
failure. The direct A/B composites now show that the decoded final distribution loss is plateauing
behind the pure temporal evidence on hard object-lock cases.

Because the temporal-window label is intentionally multi-modal, the next A/B test replaces the
single-coordinate DSNT `XY` term with a point negative log likelihood on the decoded distribution.
The active B-side `mix-LNLL` objective is:

```text
L = L_diffusion_noise_mse + alpha_bar_t * (0.001 * L_point_NLL + 0.10 * L_JS)
L_point_NLL = -log p_decoded(pixel(gaze_xy))
L_XY = 0
```

This keeps the broad decoded distribution aligned to the temporal-window heatmap through `JS`,
while only asking that the observed gaze point have probability mass instead of forcing the whole
distribution into a single DSNT mean. The DSNT-mix run is now frozen as A-side reference at epoch
19, and the `mix-LNLL` run is active as B-side. Same-epoch A/B evidence now reaches B epoch 19,
which is the last preserved same-budget A-side checkpoint. `mix-LNLL` is stable and does not
introduce checkerboard collapse, but it still has not reduced the object/table lock pattern enough
to beat DSNT-mix. Epoch 19 is the current latest same-epoch read: A has the better scalar
checkpoint naming (`0.016` vs B `0.018`), B is not cleaner on the hard carton/box checkpoint
sample, and the validation row is again more fragmented under B with off-target table/cup/object
side modes across all four samples. Episode 86 remains essentially tied and clean, but with no
clear tabletop/object-lock reduction. Later B-side evidence now includes mismatched-budget fallbacks
through `latest/epoch48` against A epoch 19/latest. Those later reads are useful trend checks only: B remains stable and checkerboard-safe; latest/epoch43 was the cleanest validation sheet, but epochs 44-48 regress or stay below/off-target with renewed tabletop side modes. Episode 86 remains near-tied, and all post-19 comparisons remain mismatched-budget because A is frozen at epoch 19.

### Temporal Mix-LNLL Latest/Epoch 48 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced through validation `epoch_0048`; the checkpoint watcher preview
  `20260617_172059_latest`, validation preview `epoch_0048`, and episode 86 preview
  `20260617_171849_latest_episode86` are complete. This is `latest/epoch48` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 49; latest checked `logs.json.txt` reached
  `global_step=65924`, and the tmux pane showed about `1060/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a fallback comparison with B latest/epoch48 against A-side DSNT-mix
  epoch 19/latest. This is explicitly a mismatched-budget latest comparison, not a strict
  same-epoch A/B.
- Server epoch48 comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0048_mismatched_budget`.
- Local epoch48 comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0048_mismatched_budget`.
- Server epoch48 episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0048_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0048_mismatched_budget.mp4`.
- Local epoch48 episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0048_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0048_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `48`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch48 remains compact and checkerboard-safe, but not a clear improvement
  over the frozen DSNT-mix A-side or the cleaner B epoch43 sheet. Checkpoint sample 2 is a compact
  below-target object/texture lock, sample 3 remains laterally biased, and validation keeps
  vertical/table-cup side modes in samples 1-3. Episode 86 is still near-tied, with frame 0
  off-target on both sides and later frames compact.
### Temporal Mix-LNLL Latest/Epoch 47 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced through validation `epoch_0047`; the checkpoint watcher preview
  `20260617_161949_latest`, validation preview `epoch_0047`, and episode 86 preview
  `20260617_161730_latest_episode86` are complete. This is `latest/epoch47` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 48; latest checked `logs.json.txt` reached
  `global_step=64726`, and the tmux pane showed about `1630/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a fallback comparison with B latest/epoch47 against A-side DSNT-mix
  epoch 19/latest. This is explicitly a mismatched-budget latest comparison, not a strict
  same-epoch A/B.
- Server epoch47 comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0047_mismatched_budget`.
- Local epoch47 comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0047_mismatched_budget`.
- Server epoch47 episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0047_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0047_mismatched_budget.mp4`.
- Local epoch47 episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0047_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0047_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `47`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch47 remains compact and checkerboard-safe, but not a clear improvement
  over the frozen DSNT-mix A-side or the cleaner B epoch43 sheet. Checkpoint sample 2 is less
  vertically fragmented than A but keeps a below-target texture tail, sample 3 remains laterally
  biased, and validation keeps below/side table-cup modes. Episode 86 is still near-tied, with
  frame 0 off-target on both sides and later frames compact.

### Temporal Mix-LNLL Latest/Epoch 46 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced through validation `epoch_0046`; the checkpoint watcher preview
  `20260617_151837_latest`, validation preview `epoch_0046`, and episode 86 preview
  `20260617_151916_latest_episode86` are complete. This is `latest/epoch46` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 47; latest checked `logs.json.txt` reached
  `global_step=63496`, and the tmux pane showed about `2070/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a fallback comparison with B latest/epoch46 against A-side DSNT-mix
  epoch 19/latest. This is explicitly a mismatched-budget latest comparison, not a strict
  same-epoch A/B.
- Server epoch46 comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0046_mismatched_budget`.
- Local epoch46 comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0046_mismatched_budget`.
- Server epoch46 episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0046_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0046_mismatched_budget.mp4`.
- Local epoch46 episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0046_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0046_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `46`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch46 remains compact and checkerboard-safe but does not recover the
  cleaner epoch43 validation result. Checkpoint sample 2 locks below target on object texture,
  sample 3 remains sideways, and validation is consistently below/off-target with table/cup side
  modes. Episode 86 remains near-tied. This keeps the strict conclusion unchanged: replacing DSNT
  XY with point NLL has not yet produced reliable object/table-lock reduction.
### Temporal Mix-LNLL Latest/Epoch 45 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced through validation `epoch_0045`; the checkpoint watcher preview
  `20260617_141727_latest`, validation preview `epoch_0045`, and episode 86 preview
  `20260617_141802_latest_episode86` are complete. This is `latest/epoch45` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 46; latest checked `logs.json.txt` reached
  `global_step=62690`, and the tmux pane showed about `4204/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a fallback comparison with B latest/epoch45 against A-side DSNT-mix
  epoch 19/latest. This is explicitly a mismatched-budget latest comparison, not a strict
  same-epoch A/B.
- Server epoch45 comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0045_mismatched_budget`.
- Local epoch45 comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0045_mismatched_budget`.
- Server epoch45 episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0045_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0045_mismatched_budget.mp4`.
- Local epoch45 episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0045_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0045_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `45`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch45 regresses relative to epoch43. It remains compact and
  checkerboard-safe, but checkpoint sample 0 has a new off-target shelf blob, the hard carton
  sample has a stronger vertical chain, and validation fragments into multiple table/monitor modes.
  Episode 86 remains near-tied and does not show a placement win. This keeps the strict conclusion
  unchanged: replacing DSNT XY with point NLL has not yet produced reliable object/table-lock
  reduction.
### Temporal Mix-LNLL Latest/Epoch 43 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced through validation `epoch_0043`; the checkpoint watcher preview
  `20260617_121504_latest`, validation preview `epoch_0043`, and episode 86 preview
  `20260617_121515_latest_episode86` are complete. This is `latest/epoch43` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 44; latest checked `logs.json.txt` reached
  `global_step=59185`, and the tmux pane showed about `618/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled two fallback comparisons while catching up: B latest/epoch42 and
  B latest/epoch43 against A-side DSNT-mix epoch 19/latest. Both are explicitly
  mismatched-budget latest comparisons, not strict same-epoch A/B.
- Server epoch43 comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0043_mismatched_budget`.
- Local epoch43 comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0043_mismatched_budget`.
- Server epoch43 episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0043_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0043_mismatched_budget.mp4`.
- Local epoch43 episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0043_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0043_mismatched_budget.mp4`.
- Intermediate epoch42 comparison was also packaged on server at
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0042_mismatched_budget`
  and locally at
  `W:\实验室项目\gaze-wam\.codex_tmp\dsnt_epoch0019_vs_nll_latest_epoch0042_mismatched_budget`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `43`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch43 is the best mismatched-budget trend read so far. It remains
  compact and checkerboard-safe. Validation now places compact mass near the fixation dot on all
  four repeated monitor/table samples, with only small residual vertical tails rather than the
  strong table/monitor streaks seen at epoch41/42. Checkpoint sample 0 tightens toward the table
  target, and the hard carton/object sample has a weaker off-target lobe. Episode 86 remains
  near-tied against A epoch19, so this is not yet a strict object/table-lock win.

### Temporal Mix-LNLL Latest/Epoch 41 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced to validation `epoch_0041`; the checkpoint watcher preview
  `20260617_101541_latest`, validation preview `epoch_0041`, and episode 86 preview
  `20260617_101530_latest_episode86` are complete. This is `latest/epoch41` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 42; latest checked `logs.json.txt` reached
  `global_step=57358`, and the tmux pane showed about `2300/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a compact fallback comparison with B latest/epoch41 against A-side
  DSNT-mix epoch 19/latest, because the stopped A-side run has no epoch 41 output. This is
  explicitly a mismatched-budget latest comparison, not a strict same-epoch A/B.
- Server comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0041_mismatched_budget`.
- Local comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0041_mismatched_budget`.
- Server episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0041_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0041_mismatched_budget.mp4`.
- Local episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0041_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0041_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `41`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch41 remains compact and checkerboard-safe, but it still does not show a
  reliable win over A epoch19. The hard checkpoint carton/object sample keeps an off-target
  vertical streak, checkpoint sample 3 spreads sideways, validation regresses into tall vertical
  streaks around the table/cup/monitor structure, and episode 86 remains near-tied without clear
  object/table-lock reduction.

### Temporal Mix-LNLL Latest/Epoch 40 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced to validation `epoch_0040`; the checkpoint watcher preview
  `20260617_091433_latest`, validation preview `epoch_0040`, and episode 86 preview
  `20260617_091411_latest_episode86` are complete. This is `latest/epoch40` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 41; latest checked `logs.json.txt` reached
  `global_step=55440`, and the tmux pane showed about `1996/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a compact fallback comparison with B latest/epoch40 against A-side
  DSNT-mix epoch 19/latest, because the stopped A-side run has no epoch 40 output. This is
  explicitly a mismatched-budget latest comparison, not a strict same-epoch A/B.
- Server comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0040_mismatched_budget`.
- Local comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0040_mismatched_budget`.
- Server episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0040_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0040_mismatched_budget.mp4`.
- Local episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0040_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0040_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `40`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch40 remains checkerboard-safe and compact, but it still does not show a
  reliable win over A epoch19. Checkpoint sample 0 is cleaner than epoch39, but sample 2 keeps
  separated vertical/carton-object mass; validation remains below/left of the target with
  table/cup/object bias and a split sample 3; episode 86 remains near-tied without clear
  object/table-lock reduction.

### Temporal Mix-LNLL Latest/Epoch 39 Fallback Comparison

Details:

- Rechecked the active B-side `mix-LNLL` run on `root@106.14.2.243 -p 1024`.
- B-side `latest.ckpt` advanced to validation `epoch_0039`; the checkpoint watcher preview
  `20260617_081334_latest`, validation preview `epoch_0039`, and episode 86 preview
  `20260617_081248_latest_episode86` are complete. This is `latest/epoch39` evidence only, not a
  named checkpoint comparison.
- Training remains healthy in epoch 40; latest checked `logs.json.txt` reached
  `global_step=54246`, and the tmux pane showed about `2586/5359` batches through the epoch.
  Training, checkpoint watcher, and episode watcher tmux sessions are alive.
- Generated and pulled a compact fallback comparison with B latest/epoch39 against A-side
  DSNT-mix epoch 19/latest, because the stopped A-side run has no epoch 39 output. This is
  explicitly a mismatched-budget latest comparison, not a strict same-epoch A/B.
- Server comparison bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0039_mismatched_budget`.
- Local comparison root:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0039_mismatched_budget`.
- Server episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0039_mismatched_budget/episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0039_mismatched_budget.mp4`.
- Local episode video:
  `W:\实验室项目\gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_latest_epoch0039_mismatched_budget\episode86_A_dsnt_epoch0019_vs_B_nll_latest_epoch0039_mismatched_budget.mp4`.
- Summary fields confirm `comparison=mismatched-budget-latest`, A epoch `19`, B validation
  epoch `39`, B `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`, checkpoint selected indices `6600, 1341, 9843, 11637`,
  and matched validation selected batch indices `6, 3, 0, 7`. The generated A-vs-B episode
  video is valid at `2592x256`, 15 FPS, `3599` frames.
- Visual read: latest/epoch39 remains checkerboard-safe and compact, but it still does not show a
  reliable win over A epoch19. Checkpoint sample 0 adds an extra off-target mode, sample 2 stays
  vertically fragmented, validation is more fragmented with table/cup/object side modes, and
  episode 86 remains near-tied without clear object/table-lock reduction.

## Runs Compared

### Previous Baseline

- Output:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_`
- Final checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0199-val_loss=0.002.ckpt`
- Training status: completed at epoch 199 / global step 267999.
- Final validation loss: about `0.0015848265`.
- Objective: single-point Gaussian clean heatmap label encoded by frozen Cosmos tokenizer, trained
  with latent diffusion/noise MSE only.

### Temporal-Window Run

- Output:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_`
- Training tmux:
  `gaze_wam_open_cosmos_temporal_delta_noise_8gpu`
- Checkpoint watcher:
  `gaze_wam_temporal_delta_noise_ckpt_preview_watch`
- Episode watcher:
  `gaze_wam_temporal_delta_noise_episode_preview_watch`
- Temporal label settings:
  `mode=bidirectional`, `window_radius=30`, `beta=10.0`, `sigma_px=6.0`,
  `current_weight=2.0`.
- Epoch 0 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.018.ckpt`
- Epoch 0 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_142032_epoch=0000-val_loss=0.018`
- Epoch 0 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_142153_epoch=0000-val_loss=0.018_episode86`
- Epoch 0 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_episode86\comparison.mp4`
- Epoch 1 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.011.ckpt`
- Epoch 1 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_152126_epoch=0001-val_loss=0.011`
- Epoch 1 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_152120_epoch=0001-val_loss=0.011_episode86`
- Epoch 1 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_episode86\comparison.mp4`
- Epoch 2 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.008.ckpt`
- Epoch 2 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_162219_epoch=0002-val_loss=0.008`
- Epoch 2 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_162158_epoch=0002-val_loss=0.008_episode86`
- Epoch 2 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_episode86\comparison.mp4`
- Epoch 3 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0003-val_loss=0.007.ckpt`
- Epoch 3 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_172313_epoch=0003-val_loss=0.007`
- Epoch 3 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_172533_epoch=0003-val_loss=0.007_episode86`
- Epoch 3 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_episode86\comparison.mp4`
- Epoch 4 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0004-val_loss=0.004.ckpt`
- Epoch 4 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_182410_epoch=0004-val_loss=0.004`
- Epoch 4 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_182344_epoch=0004-val_loss=0.004_episode86`
- Epoch 4 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_episode86\comparison.mp4`
- Epoch 5 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0005-val_loss=0.003.ckpt`
- Epoch 5 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_192505_epoch=0005-val_loss=0.003`
- Epoch 5 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_192449_epoch=0005-val_loss=0.003_episode86`
- Epoch 5 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_episode86\comparison.mp4`
- Epoch 6 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0006-val_loss=0.003.ckpt`
- Epoch 6 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_202600_epoch=0006-val_loss=0.003`
- Epoch 6 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_202558_epoch=0006-val_loss=0.003_episode86`
- Epoch 6 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_episode86\comparison.mp4`
- Epoch 8 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0008-val_loss=0.003.ckpt`
- Epoch 8 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260613_222749_epoch=0008-val_loss=0.003`
- Epoch 8 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260613_222811_epoch=0008-val_loss=0.003_episode86`
- Epoch 8 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_episode86\comparison.mp4`
- Epoch 11 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0011-val_loss=0.004.ckpt`
- Epoch 11 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_013035_epoch=0011-val_loss=0.004`
- Epoch 11 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_013117_epoch=0011-val_loss=0.004_episode86`
- Epoch 11 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_episode86\comparison.mp4`
- Epoch 12 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0012-val_loss=0.003.ckpt`
- Epoch 12 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_023128_epoch=0012-val_loss=0.003`
- Epoch 12 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_023203_epoch=0012-val_loss=0.003_episode86`
- Epoch 12 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_episode86\comparison.mp4`
- Epoch 13 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0013-val_loss=0.004.ckpt`
- Epoch 13 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_033223_epoch=0013-val_loss=0.004`
- Epoch 13 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_033306_epoch=0013-val_loss=0.004_episode86`
- Epoch 13 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_episode86\comparison.mp4`
- Epoch 14 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0014-val_loss=0.003.ckpt`
- Epoch 14 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_043317_epoch=0014-val_loss=0.003`
- Epoch 14 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_043401_epoch=0014-val_loss=0.003_episode86`
- Epoch 14 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_episode86\comparison.mp4`
- Epoch 15 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0015-val_loss=0.004.ckpt`
- Epoch 15 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_053413_epoch=0015-val_loss=0.004`
- Epoch 15 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_053508_epoch=0015-val_loss=0.004_episode86`
- Epoch 15 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_episode86\comparison.mp4`
- Epoch 16 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0016-val_loss=0.003.ckpt`
- Epoch 16 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_063509_epoch=0016-val_loss=0.003`
- Epoch 16 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_063620_epoch=0016-val_loss=0.003_episode86`
- Epoch 16 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_episode86\comparison.mp4`

- Epoch 17 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0017-val_loss=0.003.ckpt`
- Epoch 17 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_073600_epoch=0017-val_loss=0.003`
- Epoch 17 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_073641_epoch=0017-val_loss=0.003_episode86`
- Epoch 17 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_episode86\comparison.mp4`
- Epoch 18 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0018-val_loss=0.004.ckpt`
- Epoch 18 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_083455_epoch=0018-val_loss=0.004`
- Epoch 18 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_083728_epoch=0018-val_loss=0.004_episode86`
- Epoch 18 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_episode86\comparison.mp4`

- Epoch 19 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/checkpoints/epoch=0019-val_loss=0.004.ckpt`
- Epoch 19 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/ckpt_heatmap/watched/20260614_093549_epoch=0019-val_loss=0.004`
- Epoch 19 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_delta_noise_mse_8gpu_amp_/media/episode_heatmap/watched/20260614_093835_epoch=0019-val_loss=0.004_episode86`
- Epoch 19 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_episode86\comparison.mp4`

### Temporal-Window Mixed-Loss Run

- Output:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_`
- Training tmux:
  `gaze_wam_open_cosmos_temporal_mixed_loss_8gpu`
- Checkpoint watcher:
  `gaze_wam_temporal_mixed_loss_ckpt_preview_watch`
- Episode watcher:
  `gaze_wam_temporal_mixed_loss_episode_preview_watch`
- Objective:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_xy_loss_weight=0.05`, `heatmap_js_loss_weight=0.10`.
- Status: stopped intentionally at 2026-06-15 15:33 +08 after preserving epoch 19 output as
  A-side reference for the `mix-LNLL` A/B test.
- Latest preserved checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0019-val_loss=0.016.ckpt`
- Latest preserved checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_152145_epoch=0019-val_loss=0.016`
- Latest preserved episode 86 video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_152255_epoch=0019-val_loss=0.016_episode86/comparison.mp4`
- Temporal label settings:
  `mode=bidirectional`, `window_radius=30`, `beta=10.0`, `sigma_px=6.0`,
  `current_weight=2.0`.

### Temporal-Window Mix-LNLL Run

- Output:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_`
- Training tmux:
  `gaze_wam_open_cosmos_temporal_mixed_nll_8gpu`
- Checkpoint watcher:
  `gaze_wam_temporal_mixed_nll_ckpt_preview_watch`
- Episode watcher:
  `gaze_wam_temporal_mixed_nll_episode_preview_watch`
- Objective:
  `heatmap_objective=diffusion`, `latent_mse_loss=true`,
  `heatmap_diffusion_final_loss_enabled=true`,
  `heatmap_final_loss_timestep_weighting=alpha_cumprod`,
  `heatmap_xy_loss_weight=0.0`, `heatmap_point_nll_loss_weight=0.001`,
  `heatmap_js_loss_weight=0.10`.
- Formula:
  `L = L_diffusion_noise_mse + alpha_bar_t * (0.001 * L_point_NLL + 0.10 * L_JS)`.
- Temporal label settings:
  `mode=bidirectional`, `window_radius=30`, `beta=10.0`, `sigma_px=6.0`,
  `current_weight=2.0`.
- Initial status at 2026-06-15 15:40 +08: healthy in epoch 0, `global_step=41`, all 8 GPUs busy.
  The first logs include `train_heatmap_point_nll_loss`, confirming the new term is active.
- Epoch 0 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.031.ckpt`
- Epoch 0 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_163925_epoch=0000-val_loss=0.031`
- Epoch 0 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0000`
- Epoch 0 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_163925_epoch=0000-val_loss=0.031_episode86`
- Epoch 0 local A/B composites against DSNT-mix epoch 0:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0000_vs_nll_epoch0000\comparisons`
- Epoch 0 same-epoch A/B read: both A and B have the same scalar validation loss naming
  (`0.031`) and are visually near-identical. `mix-LNLL` does not introduce obvious checkerboard
  collapse, but it also does not yet fix hard object/table locks. In checkpoint samples, B has a
  slightly stronger side hotspot in sample 1 and a slightly stronger upper side peak in sample 3.
  In validation samples, B slightly weakens one isolated left hotspot in sample 2, but the main
  predicted mass remains scene/object biased. Episode 86 stills at frames 0/60/120/180 are
  effectively tied and still locked to the tabletop/object structure rather than fully matching
  the temporal target.
- Epoch 1 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.025.ckpt`
- Epoch 1 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_174048_epoch=0001-val_loss=0.025`
- Epoch 1 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0001`
- Epoch 1 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_174106_epoch=0001-val_loss=0.025_episode86`
- Epoch 1 local A/B composites against DSNT-mix epoch 1:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0001_vs_nll_epoch0001\comparisons`
- Epoch 1 same-epoch A/B read: A has slightly lower scalar validation naming (`0.024`) than B
  (`0.025`). B is tighter and closer to gaze on checkpoint samples 0/1/3, but sample 2 regresses
  with a more dispersed multi-peak response around carton/scene structure. In validation, B is
  closer on samples 0/3 but adds weak lower/right blobs on samples 1/2. Episode 86 stills at
  frames 0/60/120/180 remain essentially tied, with no clear object-lock improvement.
- Epoch 2 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.020.ckpt`
- Epoch 2 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260615_184200_epoch=0002-val_loss=0.020`
- Epoch 2 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0002`
- Epoch 2 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260615_184226_epoch=0002-val_loss=0.020_episode86`
- Epoch 2 local A/B composites against DSNT-mix epoch 2:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0002_vs_nll_epoch0002\comparisons`
- Epoch 2 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0002_vs_nll_epoch0002/episode86_A_dsnt_vs_B_nll_epoch0002.mp4`
- Epoch 2 same-epoch A/B read: A keeps the better scalar validation naming (`0.018`) versus B
  (`0.020`). B is close to A and remains stable, but it does not clearly improve the hard
  multi-modal cases: checkpoint sample 0 gains weak upper/side scatter, sample 1 is only slightly
  tighter, sample 2 still locks to carton/scene structure, and sample 3 is near-identical.
  Validation samples are slightly more fragmented under B in several views. Episode 86 stills at
  frames 0/60/120/180 remain effectively tied, with no visible checkerboard regression but no clear
  reduction in tabletop/object-lock bias.
- Epoch 9 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0009-val_loss=0.015.ckpt`
- Epoch 9 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_015018_epoch=0009-val_loss=0.015`
- Epoch 9 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0009`
- Epoch 9 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_014955_epoch=0009-val_loss=0.015_episode86`
- Epoch 9 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_051552_epoch=0009-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0009`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_051501_epoch=0009-val_loss=0.014_episode86`
- Epoch 9 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0009_vs_nll_epoch0009\comparisons`
- Epoch 9 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0009_vs_nll_epoch0009/episode86_A_dsnt_vs_B_nll_epoch0009.mp4`
- Epoch 9 same-epoch A/B read: A keeps the better scalar validation naming (`0.014`) versus B
  (`0.015`). B remains stable and does not show checkerboard collapse, but it is not visually
  cleaner. In checkpoint samples, B adds extra side/double modes on samples 0 and 2 and remains
  comparable rather than better on samples 1 and 3. In validation samples, B is more fragmented and
  split across the row, while A stays more compact around its dominant prediction. Episode 86
  frames 0/60/120/180 are essentially tied and clean, but do not show a clear object-lock/tabletop
  bias reduction. Net: replacing DSNT `XY` with point NLL at weight `0.001` is stable, but current
  same-budget evidence still favors DSNT-mix.
- Epoch 10 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0010-val_loss=0.015.ckpt`
- Epoch 10 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_025129_epoch=0010-val_loss=0.015`
- Epoch 10 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0010`
- Epoch 10 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_025118_epoch=0010-val_loss=0.015_episode86`
- Epoch 10 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_061703_epoch=0010-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0010`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_061617_epoch=0010-val_loss=0.013_episode86`
- Epoch 10 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0010_vs_nll_epoch0010\comparisons`
- Epoch 10 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0010_vs_nll_epoch0010/episode86_A_dsnt_vs_B_nll_epoch0010.mp4`
- Epoch 10 same-epoch A/B read: A has the better scalar validation naming (`0.013`) versus B
  (`0.015`), and the visual gap is clearer than epoch 9. B remains stable and does not show
  checkerboard collapse, but it is not cleaner: checkpoint sample 0 adds extra side/upper modes,
  sample 2 keeps a separated vertical multi-lobe structure, and sample 3 adds a side blob.
  Validation samples 0/1/2/3 are all more fragmented under B, with multiple object/table-biased
  lobes, while A stays more compact around its dominant mode. Episode 86 frames 0/60/120/180 are
  essentially tied and clean, but do not show a clear object-lock/tabletop-bias reduction. Net:
  point-NLL at `0.001` is stable but currently inferior to the DSNT `XY` anchor for this
  temporal-window mixed objective.
- Epoch 11 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0011-val_loss=0.015.ckpt`
- Epoch 11 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_035238_epoch=0011-val_loss=0.015`
- Epoch 11 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0011`
- Epoch 11 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_035244_epoch=0011-val_loss=0.015_episode86`
- Epoch 11 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_071515_epoch=0011-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0011`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_071720_epoch=0011-val_loss=0.014_episode86`
- Epoch 11 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0011_vs_nll_epoch0011\comparisons`
- Epoch 11 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0011_vs_nll_epoch0011/episode86_A_dsnt_vs_B_nll_epoch0011.mp4`
- Epoch 11 same-epoch A/B read: A keeps the better scalar validation naming (`0.014`) versus B
  (`0.015`). B is stable and does not show checkerboard collapse. Checkpoint samples are mixed:
  B is close on samples 0/1/3, but sample 2 remains the hard case with a separated vertical
  multi-lobe response around the carton/scene edge. Validation remains the deciding negative
  signal for B because all four B samples are more split/fragmented with extra table/object-biased
  lobes, while A stays more compact around its dominant mode. Episode 86 frames 0/60/120/180 are
  essentially tied and clean, with no clear object-lock/tabletop-bias reduction. Net: point-NLL at
  `0.001` remains stable but still does not outperform the DSNT `XY` anchor.
- Epoch 12 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0012-val_loss=0.015.ckpt`
- Epoch 12 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_045349_epoch=0012-val_loss=0.015`
- Epoch 12 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0012`
- Epoch 12 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_045403_epoch=0012-val_loss=0.015_episode86`
- Epoch 12 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_081625_epoch=0012-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0012`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_081732_epoch=0012-val_loss=0.013_episode86`
- Epoch 12 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0012_vs_nll_epoch0012\comparisons`
- Epoch 12 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0012_vs_nll_epoch0012/episode86_A_dsnt_vs_B_nll_epoch0012.mp4`
- Epoch 12 same-epoch A/B read: A has the better scalar validation naming (`0.013`) versus B
  (`0.015`). B is occasionally compact and close in checkpoint samples, but it is not a decisive
  improvement. Validation is worse for the target failure mode: B shows off-target table/object
  blobs and split modes across all four samples, especially the hard sample 2. Episode 86 frames
  0/60/120/180 remain essentially tied and clean, with no checkerboard regression but also no clear
  object-lock/tabletop-bias reduction. Net: point-NLL at `0.001` is stable, but same-budget epoch
  12 evidence still favors DSNT-mix.
- Epoch 13 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0013-val_loss=0.016.ckpt`
- Epoch 13 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_055200_epoch=0013-val_loss=0.016`
- Epoch 13 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0013`
- Epoch 13 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_055224_epoch=0013-val_loss=0.016_episode86`
- Epoch 13 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_091736_epoch=0013-val_loss=0.014`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0013`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_091853_epoch=0013-val_loss=0.014_episode86`
- Epoch 13 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0013_vs_nll_epoch0013\comparisons`
- Epoch 13 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0013_vs_nll_epoch0013/episode86_A_dsnt_vs_B_nll_epoch0013.mp4`
- Epoch 13 same-epoch A/B read: A has the better scalar validation naming (`0.014`) versus B
  (`0.016`). B is sometimes compact in checkpoint samples and partially suppresses the upper false
  modes on sample 2, but validation is again the deciding negative signal: all four B samples are
  more split or shifted toward table/object structure, with sample 2 turning into a lower
  table/cup/object blob. Episode 86 frames 0/60/120/180 remain essentially tied and clean, with no
  checkerboard regression but also no clear object-lock/tabletop-bias reduction. Net:
  point-NLL at `0.001` remains stable, but same-budget epoch 13 evidence still favors DSNT-mix.
- Epoch 14 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0014-val_loss=0.015.ckpt`
- Epoch 14 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_065309_epoch=0014-val_loss=0.015`
- Epoch 14 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0014`
- Epoch 14 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_065339_epoch=0014-val_loss=0.015_episode86`
- Epoch 14 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_101847_epoch=0014-val_loss=0.013`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0014`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_101931_epoch=0014-val_loss=0.013_episode86`
- Epoch 14 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0014_vs_nll_epoch0014\comparisons`
- Epoch 14 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0014_vs_nll_epoch0014/episode86_A_dsnt_vs_B_nll_epoch0014.mp4`
- Epoch 14 same-epoch A/B read: A has the better scalar validation naming (`0.013`) versus B
  (`0.015`). B remains close in checkpoint samples but does not remove the hard sample 2
  vertical/object-biased multi-lobe structure. Validation is again the deciding negative signal:
  all four B samples are more fragmented or shifted toward table/object structure, especially the
  hard sample 2. Episode 86 frames 0/60/120/180 remain essentially tied and clean, with no
  checkerboard regression but also no clear object-lock/tabletop-bias reduction. Net:
  point-NLL at `0.001` remains stable, but same-budget epoch 14 evidence still favors DSNT-mix.
- Epoch 15 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0015-val_loss=0.017.ckpt`
- Epoch 15 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_075409_epoch=0015-val_loss=0.017`
- Epoch 15 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0015`
- Epoch 15 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_075457_epoch=0015-val_loss=0.017_episode86`
- Epoch 15 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_111959_epoch=0015-val_loss=0.015`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0015`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_111751_epoch=0015-val_loss=0.015_episode86`
- Epoch 15 local A/B composites:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0015_vs_nll_epoch0015\comparisons`
- Epoch 15 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0015_vs_nll_epoch0015/episode86_A_dsnt_vs_B_nll_epoch0015.mp4`
- Epoch 15 same-epoch A/B read: A has the better scalar validation naming (`0.015`) versus B
  (`0.017`). Checkpoint samples 0/1/3 are close, but B is worse on hard sample 2, where the carton
  case becomes a vertical/object-biased multi-lobe structure. Validation is again the deciding
  negative signal: all four B samples are more fragmented or shifted toward table/cup/object
  structure. Episode 86 frames 0/60/120/180 remain essentially tied and clean, with no checkerboard
  regression but also no clear object-lock/tabletop-bias reduction. Net: point-NLL at `0.001`
  remains stable, but same-budget epoch 15 evidence still favors DSNT-mix.
- Epoch 18 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0018-val_loss=0.017.ckpt`
- Epoch 18 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_105441_epoch=0018-val_loss=0.017`
- Epoch 18 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0018`
- Epoch 18 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_105610_epoch=0018-val_loss=0.017_episode86`
- Epoch 18 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_142035_epoch=0018-val_loss=0.015`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0018`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_142137_epoch=0018-val_loss=0.015_episode86`
- Epoch 18 server A/B bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0018_vs_nll_epoch0018`
- Epoch 18 local A/B bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0018_vs_nll_epoch0018`
- Epoch 18 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0018_vs_nll_epoch0018/episode86_A_dsnt_vs_B_nll_epoch0018.mp4`
- Epoch 18 same-epoch A/B read: A has the better scalar validation naming (`0.015`) versus B
  (`0.017`). Checkpoint samples 0/1 are close, but B adds more side/upper mass on sample 1,
  remains worse on hard sample 2 with vertical carton/object-biased lobes, and shifts sample 3
  left/off-target relative to A. Validation is worse for B across all four samples, with extra
  table/cup/object blobs and split modes. Episode 86 frames 60/120/180 remain near-tied and clean,
  while frame 0 still shows the same tabletop/object lock. Net: through epoch 18, point NLL at
  weight `0.001` remains stable but does not beat the DSNT `XY` anchor.
- Epoch 19 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0019-val_loss=0.018.ckpt`
- Epoch 19 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ckpt_heatmap/watched/20260616_115551_epoch=0019-val_loss=0.018`
- Epoch 19 validation preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/val_heatmap/epoch_0019`
- Epoch 19 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/episode_heatmap/watched/20260616_115734_epoch=0019-val_loss=0.018_episode86`
- Epoch 19 same-epoch A-side references:
  - checkpoint preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_152145_epoch=0019-val_loss=0.016`
  - validation preview:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0019`
  - episode 86:
    `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_152255_epoch=0019-val_loss=0.016_episode86`
- Epoch 19 server A/B bundle:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0019`
- Epoch 19 local A/B bundle:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_dsnt_epoch0019_vs_nll_epoch0019`
- Epoch 19 server A/B episode video:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0019/episode86_A_dsnt_vs_B_nll_epoch0019.mp4`
- Epoch 19 same-epoch A/B read: A has the better scalar validation naming (`0.016`) versus B
  (`0.018`). Checkpoint samples 0/1/3 remain broadly close, but B is not cleaner; hard sample 2
  loses target-centered mass and keeps a vertical object/carton-biased structure. Validation is
  worse under B across all four samples, with more fragmented table/cup/object side modes.
  Episode 86 frames are clean and checkerboard-safe, but frame 0 still shows the same
  tabletop/object lock and later frames are near-tied rather than improved. Net: through the
  preserved A-side same-budget limit of epoch 19, point NLL at weight `0.001` does not beat the
  DSNT `XY` anchor.
- Epoch 20 fallback comparison: B epoch 20 exists at
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0020-val_loss=0.018.ckpt`,
  but A-side DSNT-mix has no epoch 20 output. The generated comparison therefore uses A epoch
  19/latest and is explicitly marked mismatched-budget:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0020_mismatched_budget`.
  It does not reverse the conclusion: B remains checkerboard-safe but validation is still more
  fragmented, and episode 86 remains near-tied without clear object-lock improvement.
- Epoch 21 fallback comparison: B epoch 21 exists at
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0021-val_loss=0.018.ckpt`,
  again compared against A epoch 19/latest as mismatched-budget:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0021_mismatched_budget`.
  It continues the same trend: stable and checkerboard-safe, but validation remains fragmented and
  the hard object/carton case is not recovered.
- Epoch 23 fallback comparison: B epoch 23 exists at
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/checkpoints/epoch=0023-val_loss=0.018.ckpt`,
  again compared against A epoch 19/latest as mismatched-budget:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_epoch0023_mismatched_budget`.
  It still does not reverse the strict epoch 19 conclusion: stable and checkerboard-safe, but
  validation is still more fragmented and the hard carton/object case remains object-biased.
- Latest/epoch25 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260616_180251_latest`, validation preview `epoch_0025`, and episode 86 preview
  `20260616_180250_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0025_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint samples are broadly clean but the
  hard carton/object case still keeps side mass; validation is more fragmented and multi-peaked
  than A epoch 19 across all four samples; episode 86 remains near-tied without clear
  object/table-lock improvement.
- Latest/epoch26 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260616_190415_latest`, validation preview `epoch_0026`, and episode 86 preview
  `20260616_190429_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0026_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint samples are broadly clean but the
  hard carton/object case still keeps side mass; validation is more fragmented and multi-peaked
  than A epoch 19 across all four samples; episode 86 remains near-tied without clear
  object/table-lock improvement.
- Latest/epoch29 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260616_220746_latest`, validation preview `epoch_0029`, and episode 86 preview
  `20260616_220536_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0029_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint samples are mostly compact and
  checkerboard-safe, though B sample 0 keeps taller side mass around the target; validation remains
  more fragmented and multi-peaked than A epoch 19 on the table/cup/object cases; episode 86
  remains near-tied without clear object/table-lock improvement.
- Latest/epoch30 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260616_230858_latest`, validation preview `epoch_0030`, and episode 86 preview
  `20260616_230657_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0030_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint samples are compact enough and
  checkerboard-safe, but B keeps side/vertical mass in the hard carton/object sample; validation
  remains more fragmented and object/table-biased than A epoch 19; episode 86 remains near-tied
  without clear tabletop-lock improvement.
- Latest/epoch31 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_000710_latest`, validation preview `epoch_0031`, and episode 86 preview
  `20260617_000819_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0031_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint samples stay usable and
  checkerboard-safe, but the hard carton/object case keeps secondary vertical mass; validation
  shows even clearer extra side modes in spots and remains more fragmented than A epoch 19; episode
  86 remains near-tied without clear object/table-lock improvement.
- Latest/epoch32 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_010819_latest`, validation preview `epoch_0032`, and episode 86 preview
  `20260617_010940_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0032_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint sample 0 has a small side speckle,
  and the hard carton/object case still keeps side/vertical mass; validation remains fragmented
  with side modes around table/cup/object regions; episode 86 remains near-tied without clear
  object/table-lock improvement.
- Latest/epoch33 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_020930_latest`, validation preview `epoch_0033`, and episode 86 preview
  `20260617_021102_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0033_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: B remains checkerboard-safe, but checkpoint
  sample 0 now has a stronger off-target upper mode, the hard carton/object case still keeps
  side/vertical mass, validation remains fragmented with table/cup/object side modes, and episode
  86 remains near-tied without clear object/table-lock improvement.
- Latest/epoch34 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_031041_latest`, validation preview `epoch_0034`, and episode 86 preview
  `20260617_030925_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0034_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: B remains compact and checkerboard-safe, but
  checkpoint sample 0 keeps a vertical/off-object component, the hard carton/object case remains
  ambiguous, validation is more fragmented with table/cup/object side modes, and episode 86 remains
  near-tied without clear object/table-lock improvement.
- Latest/epoch35 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_041153_latest`, validation preview `epoch_0035`, and episode 86 preview
  `20260617_041037_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0035_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: B remains compact and checkerboard-safe, but
  checkpoint sample 0 keeps a vertical/off-object component, the hard carton/object case remains
  ambiguous, validation is more fragmented with table/cup/object side modes, and episode 86 remains
  near-tied without clear object/table-lock improvement.
- Latest/epoch36 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_051305_latest`, validation preview `epoch_0036`, and episode 86 preview
  `20260617_051204_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0036_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: B remains compact and checkerboard-safe, but
  checkpoint sample 0 keeps an off-target vertical/object-side component, the hard carton/object
  case remains ambiguous, validation is more fragmented with table/cup/object side modes, and
  episode 86 remains near-tied without clear object/table-lock improvement.
- Latest/epoch37 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_061117_latest`, validation preview `epoch_0037`, and episode 86 preview
  `20260617_061321_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0037_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: B remains compact and checkerboard-safe, with
  slightly tighter validation in a couple of cells, but checkpoint sample 0 keeps a second
  off-object lobe, the hard carton/object case remains ambiguous, and episode 86 remains near-tied
  without clear object/table-lock improvement.
- Latest/epoch38 fallback comparison: B latest evidence now has checkpoint watcher preview
  `20260617_071226_latest`, validation preview `epoch_0038`, and episode 86 preview
  `20260617_071142_latest_episode86`. The generated bundle compares it against A epoch 19/latest
  and is explicitly marked mismatched-budget-latest:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_nll_8gpu_amp_/media/ab_compare/dsnt_epoch0019_vs_nll_latest_epoch0038_mismatched_budget`.
  It remains a trend check rather than strict A/B. The A-vs-B episode video is valid at
  `2592x256`, 15 FPS, `3599` frames. Visual read: checkpoint sample 0 is a little cleaner than
  epoch37, but the hard carton/object sample remains ambiguous; validation is worse evidence, with
  separated off-target lobes and table/cup/object side modes; episode 86 remains near-tied without
  clear object/table-lock improvement.
- Epoch 0 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0000-val_loss=0.031.ckpt`
- Epoch 0 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_201120_epoch=0000-val_loss=0.031`
- Epoch 0 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_201028_epoch=0000-val_loss=0.031_episode86`
- Epoch 0 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_ckpt_preview\20260614_201120_epoch=0000-val_loss=0.031`
- Epoch 0 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_episode86\20260614_201028_epoch=0000-val_loss=0.031_episode86\comparison.mp4`
- Epoch 1 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0001-val_loss=0.024.ckpt`
- Epoch 1 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_211230_epoch=0001-val_loss=0.024`
- Epoch 1 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_211141_epoch=0001-val_loss=0.024_episode86`
- Epoch 1 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_ckpt_preview\20260614_211230_epoch=0001-val_loss=0.024`
- Epoch 1 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_episode86\20260614_211141_epoch=0001-val_loss=0.024_episode86\comparison.mp4`
- Epoch 2 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0002-val_loss=0.018.ckpt`
- Epoch 2 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_221041_epoch=0002-val_loss=0.018`
- Epoch 2 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_221300_epoch=0002-val_loss=0.018_episode86`
- Epoch 2 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0002_ckpt_preview\20260614_221041_epoch=0002-val_loss=0.018`
- Epoch 2 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0002_episode86\20260614_221300_epoch=0002-val_loss=0.018_episode86\comparison.mp4`
- Epoch 3 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0003-val_loss=0.016.ckpt`
- Epoch 3 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260614_231150_epoch=0003-val_loss=0.016`
- Epoch 3 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260614_231119_epoch=0003-val_loss=0.016_episode86`
- Epoch 3 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0003_ckpt_preview\20260614_231150_epoch=0003-val_loss=0.016`
- Epoch 3 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0003_episode86\20260614_231119_epoch=0003-val_loss=0.016_episode86\comparison.mp4`
- Epoch 4 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0004-val_loss=0.013.ckpt`
- Epoch 4 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_001259_epoch=0004-val_loss=0.013`
- Epoch 4 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_001220_epoch=0004-val_loss=0.013_episode86`
- Epoch 4 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_ckpt_preview\20260615_001259_epoch=0004-val_loss=0.013`
- Epoch 4 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_episode86\20260615_001220_epoch=0004-val_loss=0.013_episode86\comparison.mp4`
- Epoch 5 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0005-val_loss=0.012.ckpt`
- Epoch 5 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_011409_epoch=0005-val_loss=0.012`
- Epoch 5 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_011306_epoch=0005-val_loss=0.012_episode86`
- Epoch 5 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_ckpt_preview\20260615_011409_epoch=0005-val_loss=0.012`
- Epoch 5 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_episode86\20260615_011306_epoch=0005-val_loss=0.012_episode86\comparison.mp4`
- Epoch 6 checkpoint:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/checkpoints/epoch=0006-val_loss=0.012.ckpt`
- Epoch 6 validation heatmap preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/val_heatmap/epoch_0006`
- Epoch 6 checkpoint preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/ckpt_heatmap/watched/20260615_021521_epoch=0006-val_loss=0.012`
- Epoch 6 full-episode preview:
  `/mnt/workspace/shenyibo/gaze-wam/data/outputs/hot3d_open_cosmos_temporal_window_mixed_loss_8gpu_amp_/media/episode_heatmap/watched/20260615_021423_epoch=0006-val_loss=0.012_episode86`
- Epoch 6 local checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_ckpt_preview\20260615_021521_epoch=0006-val_loss=0.012`
- Epoch 6 local validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_val_preview\epoch_0006`
- Epoch 6 local episode video:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_episode86\20260615_021423_epoch=0006-val_loss=0.012_episode86\comparison.mp4`

## Visual Findings

### Baseline

The final single-point Gaussian baseline is stable but visually plateaued. Across late checkpoints,
sample 0 and sample 3 are near the gaze point but retain small local offsets. Sample 1 repeatedly
locks onto an off-target object/hand/cup region left of the gaze marker. Sample 2 remains displaced
below/right of the target. A fixed lower-left speckle cluster, faint right-edge texture, and blue
background wash persist.

This suggests the frozen RGB Cosmos latent denoising path can learn a stable single-mode target,
but the single-point Gaussian label and codec path do not fully remove texture-like artifacts or
fixed semantic offsets.

### Temporal-Window Label

The temporal-window target panels match the intended weak-density label. The current gaze point is
still the dominant mode, while neighboring frames produce a short smooth tail or nearby secondary
support. The full episode 86 target video is temporally continuous and does not show obvious
checkerboard artifacts in the target itself.

Epoch 0 predictions are only a sanity check. They still show clear offset hotspots, object/texture
locks, occasional border activations, and lower-left speckles. Some frames already activate near the
gaze point, but convergence needs later checkpoint evidence.

Epoch 1 is numerically better (`val_loss=0.011` versus epoch 0 `0.018`) but not yet visually
decisive. The target panels remain clean and current-frame weighted. Predictions begin to put some
mass near the gaze point, especially in sample 3 and around episode frame 180, but sample 1/2 still
show off-target object/texture locks, lower-left speckles, and occasional vertical or border
residuals. This is an encouraging early trend rather than a solved result.

Epoch 2 continues the numeric improvement (`val_loss=0.008`) and strengthens the partial visual
trend. Samples 0 and 3 now put their main mass close to the gaze/target, although both retain
right/up tails or secondary peaks plus the persistent lower-left speckle cluster. Sample 1 still
locks to a right-side object/edge region instead of the hand/gaze target, and sample 2 remains a
clear off-target miss below the target on object texture. In the full episode 86 preview, frames
120 and 180 show better near-target mass than epoch 1, while frames 0/60 still show spurious peaks
and semantic/texture locking. This is promising, but not yet enough to call the temporal run
converged or clean.

Epoch 3 improves numerically again (`val_loss=0.007`) and keeps the same trend. Sample 0 is close
to the gaze point but still has the fixed lower-left speckle cluster. Sample 3 is also close, with a
rightward tail. The harder cases remain hard: sample 1 still places a strong mode on the right-side
bottle/hand/object area, and sample 2 still activates on the box/object texture below and right of
the target. In the full episode 86 preview, frames 120 and 180 are near-target and look usable as
early evidence, while frames 0 and 60 still show scene-object attraction and spurious peaks.
Therefore the temporal-window label remains promising, but pure latent MSE has not yet fully
removed texture/semantic locking.

Epoch 4 is the best checkpoint so far (`val_loss=0.004`). Sample 3 is much cleaner, with its main
mode close to the gaze point and only a weaker right tail. Sample 0 is still close but remains
slightly right/up of the target and still carries the fixed lower-left speckle cluster. Sample 2 is
more compact than epoch 3 but remains biased onto box/object texture. Sample 1 remains the clearest
sampled failure: a strong right-side bottle/hand/object false mode persists. In the full episode 86
preview, frames 60, 120, and 180 are near-target and visually usable, while frame 0 still fails by
placing the main mass on the tabletop/object area instead of the left-up gaze target. This supports
using the temporal-window target as the DiT clean label, but it also shows that pure latent MSE can
still learn a scene prior when the visual context is ambiguous.

Epoch 5 is numerically better (`val_loss=0.003`) and improves the previous hardest sampled case:
sample 1 moves its strongest mass back near the gaze/hand target, although the right-side object
false mode remains as a secondary peak. Other samples are not strictly monotonic: sample 0 remains
near-target but slightly right/up with lower-left speckles, sample 2 remains vertically elongated on
the box/object texture, and sample 3 is worse than epoch 4 because its main mode shifts above the
gaze point. In episode 86, frames 120 and 180 remain usable and near-target, frame 60 remains
near-target with a small left bias, and frame 0 still fails on the tabletop/object area. The
important update is that temporal-window training is still improving numerically and fixes some
object-lock cases, but pure latent MSE still has local visual oscillations.

Epoch 6 holds the low validation loss (`val_loss=0.003`) and keeps the sample 1 improvement from
epoch 5: the strongest mass stays near the gaze/hand target and the old right-side object false
mode is much weaker than in epoch 4. It is not visually better on every case. Sample 2 regresses
back to a lower box/object-texture lock, and sample 3 is horizontally stretched and shifted above
the gaze point compared with the cleaner epoch 4. Episode frames 60, 120, and 180 remain near-target
and usable; frame 0 still fails on the tabletop/object area. This makes the current conclusion more
specific: the temporal-window clean label helps, but pure latent MSE alone can still encode a scene
prior and should not be judged by validation loss alone.

Epoch 8 still has low validation loss
(`val_loss=0.003`) and keeps the important sample 1 recovery: the main mode is near the target, with
only a secondary right-side object mode. Sample 3 also returns near target, though with a rightward
tail. Sample 0 keeps a small right/up offset and the fixed lower-left speckles. Sample 2 remains the
clearest sampled failure, with the main mass locked to the lower box/object texture instead of the
gaze point. In the full episode 86 preview, frames 60, 120, and 180 are near-target and visually
usable, while frame 0 remains a stable failure on the tabletop/object area. This is the strongest
evidence so far that temporal-window labels are useful, but also that pure latent MSE alone does
not fully remove scene-prior attraction.

Epoch 11 kept the useful
trend: sample 0, sample 1, and sample 3 place their main decoded mass near the temporal target,
though sample 0 still has lower-left speckles and sample 1 is slightly left-biased. The persistent
failure remains sample 2, where the strongest mass locks onto the lower box/object texture and a
secondary false peak appears above the target. In the full episode 86 preview, frames 60, 120, and
180 remain near-target and usable; frame 0 still locks to the tabletop/object area instead of the
left-up gaze target. This reinforces the current interpretation: temporal-window labels are useful
as clean labels, but pure latent MSE still leaves scene-prior attraction.

Epoch 12 lowers the scalar validation loss back to `0.003`, but it is not a clean visual
improvement over epoch 11. Sample 0 remains near-target with a right/up bias, lower-left speckles,
and a faint upper residual. Sample 1 is still much better than the old single-point baseline
failure, but the predicted mass is left-biased and wider than the target. Sample 2 remains the
main hard failure: instead of a single object-texture lock, it becomes a multi-peak prediction with
only weak mass near the gaze target. Sample 3 stays near-target but adds a clear right-side false
peak. Episode 86 frames 60, 120, and 180 remain near-target and usable, while frame 0 still locks
to the tabletop/object area instead of the left-up gaze target. This makes the current result more
specific: temporal-window labels improve the label semantics, but pure latent MSE still needs help
if the goal is to suppress scene-prior and multi-peak artifacts.

Epoch 13 returns to `val_loss=0.004` and is visually close to epoch 12. Sample 0 remains near the
target but right/up biased with lower-left speckles and a faint upper residual. Sample 1 remains
usable but left-biased and wider than the target. Sample 2 remains the main hard failure: the
prediction is multi-peak around the box/highlight region and does not make the gaze-centered target
the main mode. Sample 3 improves slightly because the right-side false peak is weaker and the main
peak is more compact, though it is still offset above the target. Episode 86 frames 60, 120, and
180 remain near-target and usable; frame 0 remains a stable tabletop/object lock. This supports a
plateau interpretation: temporal-window labels are better clean labels, but pure latent MSE alone
still does not suppress scene-prior and multi-peak artifacts.

Epoch 14 returns to `val_loss=0.003` and is a small useful stabilization rather than a complete
visual break from the plateau. Sample 0, sample 1, and sample 3 place their main decoded mass close
to the target and are usable, while keeping mild local bias and the persistent lower-left speckles.
Sample 2 remains the clearest hard failure: the strongest modes attach to the bright box/light and
nearby scene texture above the gaze marker, so the temporal target is not the main mode. Episode 86
frames 60, 120, and 180 remain near-target and visually usable; frame 0 still locks to the
tabletop/object area instead of the left-up gaze target. This strengthens the current read: the
temporal-window clean label is a good DiT final-label candidate, but pure latent MSE alone still
allows scene-prior attraction in ambiguous frames.

Epoch 15 returns to `val_loss=0.004` and is visually close to epoch 14. Sample 0, sample 1, and
sample 3 remain usable and near the target, with small local offsets and the persistent lower-left
speckle cluster. Sample 2 is still the main hard failure and is not improved: the strongest decoded
mass attaches to the bright box/light and nearby scene texture rather than making the temporal
target the dominant mode. Episode 86 frames 60, 120, and 180 remain near-target and usable, while
frame 0 remains a stable tabletop/object lock. This supports a plateau interpretation rather than a
new improvement: temporal-window labels are useful, but the pure latent objective still needs help
if the goal is to suppress scene-prior attraction in hard frames.

Epoch 16 returns to `val_loss=0.003` but is not a visual improvement over epoch 15. Sample 0 and
sample 1 remain near-target with mild offsets and lower-left speckles, and sample 3 remains usable
but horizontally stretched/right-biased. Sample 2 regresses into a clearer multi-peak and vertical
scene-texture lock around the box/light region; the temporal target is not the dominant mode.
Episode 86 frames 60, 120, and 180 remain near-target and usable, while frame 0 remains the stable
tabletop/object failure. This further decouples scalar validation loss from visual quality and
supports treating the next step as a true mixed-loss ablation if artifact suppression is required.

Epoch 17 again has `val_loss=0.003` and is visually redundant with epoch 16 rather than better.
Sample 0 is near target but splits vertically and keeps the lower-left speckles. Sample 1 shifts
left toward the cup/hand region and is less centered on the target marker. Sample 2 remains the
hard failure: its strongest modes attach to the box/light/scene texture above/right of the gaze
marker, so target-centered mass is weak. Sample 3 remains usable but above/right biased. Full
episode 86 frames 60, 120, and 180 remain near-target and usable; frame 0 still locks to the
tabletop/object area instead of the left-up target. This strengthens the plateau interpretation:
the temporal-window clean label helps the target semantics, but pure latent diffusion/noise MSE
does not remove scene-prior attraction, multi-peak hard cases, or fixed speckles.

Epoch 18 keeps the same picture. Sample 0 remains near the target with a slight right/up bias and
the persistent lower-left speckle cluster. Sample 1 is usable but still drifts left of the target
toward the cup/object region. Sample 2 stays the hardest failure and locks to box/highlight scene
texture instead of the temporal target. Sample 3 remains usable but above/right biased with a
horizontal trail. In the full episode, later frames are generally usable, but frame 0 remains a
tabletop/object lock. The temporal-window label is therefore still the preferred DiT final clean
label candidate, while the pure latent objective remains visually plateaued.

Epoch 19 is visually redundant with epochs 17-18 rather than better. Sample 0 is near the target
but keeps a small right/up bias and the lower-left speckles. Sample 1 remains usable but biased
left toward the cup/object region. Sample 2 is still the clearest failure: the decoded map is
multi-peak and attaches to box/highlight scene texture instead of making the temporal target the
dominant mode. Sample 3 is usable but above/right biased with a horizontal trail. In the full
episode, frames 60, 120, and 180 remain near-target and useful, while frame 0 remains a clear
tabletop/object lock. This strengthens the current conclusion that the temporal-window target is a
good clean-label shape, but pure latent diffusion/noise MSE alone is not enough to suppress the
scene-prior artifacts.

Epoch 20 keeps the same plateau. Sample 0 remains close to the target but slightly right/up with
the fixed lower-left speckle cluster. Sample 1 remains usable but left-biased toward the cup/object
region. Sample 2 is still the clearest failure: the decoded mass remains multi-peak and attaches to
bright box/light/scene texture rather than the temporal target. Sample 3 remains usable but above/
right biased with a weak trailing mode. In the full episode, frames 60, 120, and 180 remain
near-target and useful, while frame 0 remains a tabletop/object lock. The new checkpoint does not
change the conclusion: the temporal-window label is a good clean-label candidate, but pure frozen-
Cosmos latent diffusion/noise MSE still leaves scene-prior attraction and speckles.

Epoch 21 is not an improvement and slightly sharpens the hard-case failure. Sample 0 remains close
to the target with a small right/up offset and the same lower-left speckles. Sample 1 is still
usable but left-biased toward the cup/object region. Sample 2 regresses visually: decoded mass
forms multiple strong upper modes around the bright light/box scene texture plus a downward
trailing mode, while the target-centered mass is weak. Sample 3 remains usable but above/right
biased. In the full episode, frames 60, 120, and 180 remain near-target and useful, while frame 0
still locks to the tabletop/object area. This reinforces the plateau/regression interpretation and
points toward a true mixed-loss ablation rather than more pure latent MSE alone.

### Temporal-Window Mixed Loss

Epoch 0 verifies that the true mixed objective runs end to end: latent diffusion/noise MSE is active
and the decoded final heatmap receives weighted `XY` and `JS` losses. The checkpoint and full
episode 86 previews were generated successfully with no EMA and no gaze condition.

Visually, this first checkpoint is not yet an improvement over pure temporal epoch 28. The four
sample comparisons are broad and multi-peak. Sample 0 still spreads around the table/object area,
sample 1 has several object-region peaks around the cup/carton scene, sample 2 remains a hard
failure locked to carton/edge/scene geometry, and sample 3 is closer but still broad with an extra
object-biased blob. Episode 86 frames 0, 60, 120, and 180 show the same behavior: the temporal
targets are compact and stable, while predictions are still larger, multi-lobed, and more
object-biased than the pure temporal epoch 28 stills.

This does not disprove the mixed objective. It says epoch 0 is too early to use as positive
evidence. The next useful question is whether epoch 1+ shrinks these broad object-biased modes
faster than the pure temporal run did.

Epoch 1 improves the scalar validation loss from `0.031` to `0.024` and visually tightens the
predictions compared with epoch 0. The change is real but incomplete: sample 0 becomes a smaller
two-lobe prediction, sample 1 remains pulled toward the cup/carton scene, sample 2 is still the
hard carton/edge/scene-geometry failure, and sample 3 is closer but still wider than the target.
Episode 86 frames 0, 60, 120, and 180 show partial tightening but remain broader and more
object-biased than the pure temporal epoch 28 stills. Net: mixed loss is moving in the right
direction after one epoch, but it has not yet surpassed the pure temporal evidence.

Epoch 2 continues the scalar improvement from `0.024` to `0.018` and is visibly tighter than epoch
1. Sample 0 becomes compact and near the target with only mild table/object bias. Sample 1 is more
compact and closer, though a small side speck/edge bias remains. Sample 3 is tighter and closer but
still slightly broad or offset. Sample 2 remains the important hard case: the prediction is more
compact than epoch 1, but it still leans below/right toward scene geometry/carton texture instead
of centering cleanly on the target. In episode 86, frames 60, 120, and 180 are close to target and
look usable, while frame 0 still locks to the tabletop/object area rather than the left-up gaze
target. Net: epoch 2 mixed loss is now approaching pure temporal epoch 28 on easier frames, but it
has not clearly beaten it because the hardest object-lock evidence remains.

Epoch 3 improves validation loss again from `0.018` to `0.016` and gives another small visual
tightening step. Sample 0 is compact and near target with reduced stray texture. Sample 1 remains
usable and tighter than epoch 2, with only a small side bias. Sample 2 is still the hardest sampled
case: it is more compact and organized than epoch 2, but it still follows carton/scene geometry and
does not fully center on the target. Sample 3 is the cleanest sampled case, with a tight near-target
mode. In episode 86, frames 60, 120, and 180 are close to the target and stable; frame 0 remains the
main failure, still locking to the tabletop/object region instead of the left-up gaze target. Net:
mixed loss continues in the right direction, but the hard-case object prior has not disappeared.

Epoch 4 improves validation loss again from `0.016` to `0.013`, but it is not a clean visual
breakthrough. Sample 0 is compact and near target with slight right/up object/table bias and faint
lower-left speckles. Sample 1 is compact but shifted left/below target toward the cup/carton/object
area. Sample 2 regresses relative to epoch 3: it is compact, but its main mass is below/right on
carton/scene texture rather than target-centered. Sample 3 remains usable and near target, with the
main mode above or slightly right of the marker. In episode 86, frames 60, 120, and 180 remain close
and stable, though frame 180 is still a little right-biased; frame 0 remains the strongest failure
and still locks to the tabletop/object region. Net: mixed loss keeps easy cases usable and lowers
the scalar metric, but epoch 4 does not clearly beat pure temporal epoch 28 on the hard object-lock
evidence.

Epoch 5 lowers validation loss again from `0.013` to `0.012` and keeps the easy frames stable, but
it still does not resolve the hard-case scene prior. Sample 0 is compact and close to the target
with faint lower-left speckles. Sample 1 improves versus epoch 4 because the main mass moves nearer
the target, though it remains slightly left/low. Sample 2 is the strongest failure: a tall
multi-lobe prediction follows carton/scene texture instead of collapsing onto the target. Sample 3
is compact and usable with a mild above/right bias. Episode 86 frames 60, 120, and 180 remain close
and stable; frame 0 still locks to the tabletop/object area. Net: mixed loss is numerically
improving, but the decoded final distribution term has not yet beaten pure temporal epoch 28 on the
hard object-lock evidence.

Epoch 6 stays at `val_loss=0.012` and looks like a visual plateau rather than a breakthrough.
Sample 0 remains compact but slightly vertically stretched/right-biased near the table target, with
faint lower-left speckles. Sample 1 is compact and near the target but still left/low. Sample 2 is
still the strongest failure: its prediction is multi-lobed and follows carton/scene texture instead
of centering on the gaze marker, though it is less vertically stretched than epoch 5. Sample 3 stays
usable and compact with a small above/right bias. Episode 86 frames 60, 120, and 180 remain close
and stable; frame 0 still locks to the tabletop/object area. Net: no new checkerboard collapse, but
the mixed decoded final loss still has not removed the hard object-lock/scene-prior artifact.

Epoch 7 remains on the `val_loss=0.012` plateau but is slightly cleaner in the easy cases. Sample
0 is mostly target-centered yet still keeps a small upper double mode and faint lower-left speckles.
Sample 1 is compact and usable, though still a little left/low of the gaze point. Sample 2 is still
the clearest hard failure: the prediction forms a vertical multi-lobe structure along the carton/
scene edge instead of collapsing onto the gaze marker. Sample 3 is compact and near target with only
a mild above/right bias. Episode 86 frames 60, 120, and 180 remain close and stable, but frame 0
still locks to the tabletop/object area. Net: there is still no checkerboard collapse, and the
decoded final loss has not yet solved the object-lock/scene-prior artifact better than the pure
temporal epoch 28 evidence.

Epoch 8 regresses slightly in scalar validation loss from `0.012` to `0.013`, and visually it is
another plateau checkpoint rather than a hard-case recovery. Sample 0 is cleaner than epoch 7: the
upper double mode is mostly gone and the main mass is target-centered, though faint lower-left
speckles persist. Sample 1 stays compact and usable with a small left/low bias. Sample 2 remains
the strongest failure, with a vertical multi-lobe strip following the carton/scene edge instead of
collapsing onto the gaze marker. Sample 3 is compact and near target with a mild above/right bias.
Episode 86 frames 60, 120, and 180 remain close and stable, while frame 0 still locks to the
tabletop/object area. Net: no checkerboard collapse, but the decoded final distribution term still
does not beat the pure temporal epoch 28 hard-case evidence.

Epoch 9 worsens the scalar trend again from `0.013` to `0.014` and remains a plateau/regression
checkpoint. Sample 0 is compact but slightly right/down biased with faint lower-left speckles.
Sample 1 is compact but left/low of the marker and still pulled toward the object cluster. Sample 2
remains the key hard failure: the prediction is narrower than epoch 8 but still locks to the
carton/scene edge instead of the gaze marker. Sample 3 remains compact and near target with mild
above/right bias. Episode 86 frames 60, 120, and 180 remain close and stable; frame 0 still locks
to the tabletop/object area. Net: no checkerboard collapse, but the decoded final distribution
term has not removed the object-lock/scene-prior failure.

Epoch 10 recovers scalar validation loss from `0.014` to `0.013`, but the visual conclusion remains
mostly unchanged. Sample 0 is compact near the table target with faint lower-left speckles. Sample 1
is compact and usable with a small object-cluster pull. Sample 2 is still the key hard failure: the
decoded mass follows the carton/scene edge instead of collapsing onto the gaze marker. Sample 3 is
compact and near target with mild above/right bias. In the full episode 86 render, frames 60, 120,
and 180 remain close and stable, while frame 0 still locks to the tabletop/object area. Net:
epoch 10 is not a checkerboard failure, but it still does not beat the pure temporal epoch 28
hard-case evidence.

Epoch 11 regresses scalar validation loss back from `0.013` to `0.014`, and visually it reinforces
the same plateau. Sample 0 and sample 3 are compact and usable near the target. Sample 1 remains
compact with minor object-cluster pull. Sample 2 still fails by following the carton/scene edge
rather than the gaze marker, and episode 86 frame 0 remains locked to the tabletop/object area while
frames 60/120/180 stay close. Net: the mixed decoded final distribution term has not yet removed
the hard scene-prior/object-lock behavior.

Epoch 13 remains on the same validation plateau at `val_loss=0.014`. Sample 0 and sample 3 are
compact and usable, but sample 2 regresses to a stronger vertical multi-peak strip along the
carton/scene edge instead of the gaze marker. Episode 86 frame 0 still locks to the tabletop/object
area rather than the left-up gaze target; frames 60/120/180 remain close. Net: no checkerboard
collapse, but the decoded final distribution term still has not beaten the pure temporal epoch 28
hard-case evidence.

Epoch 14 recovers scalar validation loss to `val_loss=0.013`, but visually it remains on the same
plateau. Checkpoint samples 0 and 3 are compact and usable, sample 1 keeps a left/lower off-target
drag, and sample 2 still forms a vertical multi-peak strip along the carton/scene edge rather than
collapsing onto the gaze marker. The validation preview is even more object-prior biased: all four
selected samples shift toward left-side tabletop objects while the target stays on the gaze point.
Episode 86 frame 0 still locks to the tabletop/object area, while frames 60/120/180 remain close.
Net: epoch 14 is a numeric stabilization, not a hard-case breakthrough.

Epoch 15 regresses scalar validation loss to `val_loss=0.015`. Checkpoint samples 0 and 3 remain
compact, but sample 0 is lower than target; sample 1 is close with object-cluster pull; sample 2 is
still the key hard failure, with decoded mass on the carton/scene edge rather than the gaze marker.
The validation preview remains strongly object-prior biased across all four selected samples, with
mass shifted left toward tabletop objects. Episode 86 frame 0 still locks to the tabletop/object
area, while frames 60/120/180 remain close. Net: epoch 15 adds plateau/regression evidence and does
not improve on mixed epoch 14 or pure temporal epoch 28.

Epoch 16 recovers scalar validation loss to `val_loss=0.013`. Following the user's A/B-test
correction, the pure temporal-window epoch 28/latest checkpoint preview, validation preview, and
episode 86 render were pulled and compared directly against mixed epoch 16. The checkpoint A/B is
mixed: samples 0/1 are comparable, sample 3 is slightly cleaner for mixed, but sample 2 still has
the same vertical carton/scene-edge multi-peak failure. The validation A/B is worse for mixed:
samples 2 and 3 show stronger object-related side peaks and tabletop bias than pure temporal
epoch 28. Episode 86 frame 0 remains a tabletop/object lock in both A and B, while frames
60/120/180 are broadly comparable. Net: the true pulled-output A/B confirms that mixed epoch 16 is
not a hard-case improvement over pure temporal epoch 28.

## Interpretation

The temporal-window label addresses a supervision problem, not a loss-mixing problem. It gives the
DiT a more realistic clean `H0` distribution by using neighboring gaze samples from the episode.
This should be less contradictory than forcing every frame into an isolated single Gaussian.

The pure temporal-window run established that the clean temporal label is a better target than the
single-point Gaussian baseline, but it still leaves some hard scene-prior failures. The active
mixed-loss run now tests the next question directly: latent diffusion/noise MSE plus a small decoded
final heatmap `XY`/`JS` loss. This is now a direct same-sample pulled-output A/B against the
completed pure temporal-window run, although still not a strict multi-seed statistical A/B test.
Epoch 0 was only a sanity preview; epochs 1-16 show real tightening versus startup, especially a
reduction in broad multi-peak blobs. However, the direct mixed epoch 16 vs pure temporal epoch 28
A/B still does not beat pure temporal on the hardest sampled and episode frames, so the decoded
final distribution term should be treated as not yet proven and currently plateaued.

## Current Recommendation

Use the temporal-window heatmap as the preferred DiT final clean label candidate. Continue the
mixed-loss 8-GPU run and judge convergence from later checkpoint previews and full-episode videos.
The main criterion is whether the decoded final distribution loss tightens the broad predictions and
suppresses scene-texture/object-lock artifacts relative to the pure temporal epoch 28 evidence.

## Local Artifacts

- Mixed-loss epoch 0 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_episode86\20260614_201028_epoch=0000-val_loss=0.031_episode86\comparison.mp4`
- Mixed-loss epoch 0 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_ckpt_preview\20260614_201120_epoch=0000-val_loss=0.031\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_ckpt_preview\20260614_201120_epoch=0000-val_loss=0.031\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_ckpt_preview\20260614_201120_epoch=0000-val_loss=0.031\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0000_ckpt_preview\20260614_201120_epoch=0000-val_loss=0.031\sample_003\comparison.png`
- Mixed-loss epoch 1 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_episode86\20260614_211141_epoch=0001-val_loss=0.024_episode86\comparison.mp4`
- Mixed-loss epoch 1 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_ckpt_preview\20260614_211230_epoch=0001-val_loss=0.024\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_ckpt_preview\20260614_211230_epoch=0001-val_loss=0.024\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_ckpt_preview\20260614_211230_epoch=0001-val_loss=0.024\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0001_ckpt_preview\20260614_211230_epoch=0001-val_loss=0.024\sample_003\comparison.png`
- Mixed-loss epoch 4 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_episode86\20260615_001220_epoch=0004-val_loss=0.013_episode86\comparison.mp4`
- Mixed-loss epoch 4 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_ckpt_preview\20260615_001259_epoch=0004-val_loss=0.013\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_ckpt_preview\20260615_001259_epoch=0004-val_loss=0.013\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_ckpt_preview\20260615_001259_epoch=0004-val_loss=0.013\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0004_ckpt_preview\20260615_001259_epoch=0004-val_loss=0.013\sample_003\comparison.png`
- Mixed-loss epoch 5 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_episode86\20260615_011306_epoch=0005-val_loss=0.012_episode86\comparison.mp4`
- Mixed-loss epoch 5 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_ckpt_preview\20260615_011409_epoch=0005-val_loss=0.012\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_ckpt_preview\20260615_011409_epoch=0005-val_loss=0.012\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_ckpt_preview\20260615_011409_epoch=0005-val_loss=0.012\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0005_ckpt_preview\20260615_011409_epoch=0005-val_loss=0.012\sample_003\comparison.png`
- Mixed-loss epoch 6 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_episode86\20260615_021423_epoch=0006-val_loss=0.012_episode86\comparison.mp4`
- Mixed-loss epoch 6 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_ckpt_preview\20260615_021521_epoch=0006-val_loss=0.012\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_ckpt_preview\20260615_021521_epoch=0006-val_loss=0.012\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_ckpt_preview\20260615_021521_epoch=0006-val_loss=0.012\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0006_ckpt_preview\20260615_021521_epoch=0006-val_loss=0.012\sample_003\comparison.png`
- Mixed-loss epoch 7 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_episode86\20260615_031523_epoch=0007-val_loss=0.012_episode86\comparison.mp4`
- Mixed-loss epoch 7 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_ckpt_preview\20260615_031631_epoch=0007-val_loss=0.012\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_ckpt_preview\20260615_031631_epoch=0007-val_loss=0.012\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_ckpt_preview\20260615_031631_epoch=0007-val_loss=0.012\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0007_ckpt_preview\20260615_031631_epoch=0007-val_loss=0.012\sample_003\comparison.png`
- Mixed-loss epoch 8 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_episode86\20260615_041640_epoch=0008-val_loss=0.013_episode86\comparison.mp4`
- Mixed-loss epoch 8 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_ckpt_preview\20260615_041443_epoch=0008-val_loss=0.013\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_ckpt_preview\20260615_041443_epoch=0008-val_loss=0.013\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_ckpt_preview\20260615_041443_epoch=0008-val_loss=0.013\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0008_ckpt_preview\20260615_041443_epoch=0008-val_loss=0.013\sample_003\comparison.png`
- Mixed-loss epoch 9 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_episode86\20260615_051501_epoch=0009-val_loss=0.014_episode86\comparison.mp4`
- Mixed-loss epoch 9 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_ckpt_preview\20260615_051552_epoch=0009-val_loss=0.014\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_ckpt_preview\20260615_051552_epoch=0009-val_loss=0.014\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_ckpt_preview\20260615_051552_epoch=0009-val_loss=0.014\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0009_ckpt_preview\20260615_051552_epoch=0009-val_loss=0.014\sample_003\comparison.png`
- Mixed-loss epoch 10 checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_ckpt_preview\20260615_061703_epoch=0010-val_loss=0.013`
- Mixed-loss epoch 10 validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_val_preview\epoch_0010`
- Mixed-loss epoch 10 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_episode86\20260615_061617_epoch=0010-val_loss=0.013_episode86\comparison.mp4`
- Mixed-loss epoch 10 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_ckpt_preview\20260615_061703_epoch=0010-val_loss=0.013\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_ckpt_preview\20260615_061703_epoch=0010-val_loss=0.013\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_ckpt_preview\20260615_061703_epoch=0010-val_loss=0.013\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0010_ckpt_preview\20260615_061703_epoch=0010-val_loss=0.013\sample_003\comparison.png`
- Mixed-loss epoch 11 checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_ckpt_preview\20260615_071515_epoch=0011-val_loss=0.014`
- Mixed-loss epoch 11 validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_val_preview\epoch_0011`
- Mixed-loss epoch 11 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_episode86\20260615_071720_epoch=0011-val_loss=0.014_episode86\comparison.mp4`
- Mixed-loss epoch 11 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_ckpt_preview\20260615_071515_epoch=0011-val_loss=0.014\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_ckpt_preview\20260615_071515_epoch=0011-val_loss=0.014\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_ckpt_preview\20260615_071515_epoch=0011-val_loss=0.014\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0011_ckpt_preview\20260615_071515_epoch=0011-val_loss=0.014\sample_003\comparison.png`
- Mixed-loss epoch 13 checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_ckpt_preview\20260615_091736_epoch=0013-val_loss=0.014`
- Mixed-loss epoch 13 validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_val_preview\epoch_0013`
- Mixed-loss epoch 13 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_episode86\20260615_091853_epoch=0013-val_loss=0.014_episode86\comparison.mp4`
- Mixed-loss epoch 13 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_ckpt_preview\20260615_091736_epoch=0013-val_loss=0.014\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_ckpt_preview\20260615_091736_epoch=0013-val_loss=0.014\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_ckpt_preview\20260615_091736_epoch=0013-val_loss=0.014\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0013_ckpt_preview\20260615_091736_epoch=0013-val_loss=0.014\sample_003\comparison.png`
- Mixed-loss epoch 14 checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_ckpt_preview\20260615_101847_epoch=0014-val_loss=0.013`
- Mixed-loss epoch 14 validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_val_preview\epoch_0014`
- Mixed-loss epoch 14 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_episode86\20260615_101931_epoch=0014-val_loss=0.013_episode86\comparison.mp4`
- Mixed-loss epoch 14 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_ckpt_preview\20260615_101847_epoch=0014-val_loss=0.013\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_ckpt_preview\20260615_101847_epoch=0014-val_loss=0.013\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_ckpt_preview\20260615_101847_epoch=0014-val_loss=0.013\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0014_ckpt_preview\20260615_101847_epoch=0014-val_loss=0.013\sample_003\comparison.png`
- Mixed-loss epoch 15 checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_ckpt_preview\20260615_111959_epoch=0015-val_loss=0.015`
- Mixed-loss epoch 15 validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_val_preview\epoch_0015`
- Mixed-loss epoch 15 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_episode86\20260615_111751_epoch=0015-val_loss=0.015_episode86\comparison.mp4`
- Mixed-loss epoch 15 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_ckpt_preview\20260615_111959_epoch=0015-val_loss=0.015\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_ckpt_preview\20260615_111959_epoch=0015-val_loss=0.015\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_ckpt_preview\20260615_111959_epoch=0015-val_loss=0.015\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0015_ckpt_preview\20260615_111959_epoch=0015-val_loss=0.015\sample_003\comparison.png`
- Mixed-loss epoch 16 checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0016_ckpt_preview\20260615_122112_epoch=0016-val_loss=0.013`
- Mixed-loss epoch 16 validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0016_val_preview\epoch_0016`
- Mixed-loss epoch 16 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_mixed_epoch0016_episode86\20260615_121906_epoch=0016-val_loss=0.013_episode86\comparison.mp4`
- Pure temporal epoch 28/latest A-side checkpoint preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_pure_temporal_epoch0028_latest_ckpt_preview\20260614_184212_latest`
- Pure temporal epoch 28 A-side validation preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_pure_temporal_epoch0028_val_preview\epoch_0028`
- Pure temporal epoch 28/latest A-side full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_pure_temporal_epoch0028_episode86\20260614_184351_latest_episode86\comparison.mp4`
- Direct A/B composites, pure temporal epoch 28/latest vs mixed epoch 16:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\ab_temporal_pure_epoch0028_vs_mixed_epoch0016`

- Temporal epoch 0 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_episode86\comparison.mp4`
- Temporal epoch 1 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_episode86\comparison.mp4`
- Temporal epoch 2 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_episode86\comparison.mp4`
- Temporal epoch 3 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_episode86\comparison.mp4`
- Temporal epoch 4 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_episode86\comparison.mp4`
- Temporal epoch 5 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_episode86\comparison.mp4`
- Temporal epoch 6 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_episode86\comparison.mp4`
- Temporal epoch 8 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_episode86\comparison.mp4`
- Temporal epoch 11 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_episode86\comparison.mp4`
- Temporal epoch 12 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_episode86\comparison.mp4`
- Temporal epoch 13 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_episode86\comparison.mp4`
- Temporal epoch 14 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_episode86\comparison.mp4`
- Temporal epoch 15 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_episode86\comparison.mp4`
- Temporal epoch 16 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_episode86\comparison.mp4`
- Temporal epoch 0 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0000_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 1 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0001_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 2 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0002_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 3 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0003_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 4 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0004_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 5 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0005_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 6 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0006_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 8 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0008_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 11 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0011_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 12 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0012_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 13 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0013_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 14 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0014_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 15 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0015_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 16 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0016_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 17 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_episode86\comparison.mp4`
- Temporal epoch 17 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0017_ckpt_preview\sample_003\comparison.png`
- Baseline final full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\episode_inference_epoch0199_val_ep0086_full\comparison.mp4`
- Temporal target-only full episode preview:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_heatmap_epoch0199_val_ep0086_w30_b10_s6_full\comparison.mp4`
- Temporal epoch 18 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_episode86\comparison.mp4`
- Temporal epoch 18 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0018_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 19 full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_episode86\comparison.mp4`
- Temporal epoch 19 sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_ckpt_preview\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_ckpt_preview\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_ckpt_preview\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_epoch0019_ckpt_preview\sample_003\comparison.png`
- Temporal epoch 24 latest full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144022_episode86\20260614_144022_latest_episode86\comparison.mp4`
- Temporal epoch 24 latest sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144021_ckpt_preview\20260614_144021_latest\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144021_ckpt_preview\20260614_144021_latest\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144021_ckpt_preview\20260614_144021_latest\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_144021_ckpt_preview\20260614_144021_latest\sample_003\comparison.png`
- Temporal epoch 25 latest full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154057_episode86\20260614_154057_latest_episode86\comparison.mp4`
- Temporal epoch 25 latest sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154128_ckpt_preview\20260614_154128_latest\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154128_ckpt_preview\20260614_154128_latest\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154128_ckpt_preview\20260614_154128_latest\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_154128_ckpt_preview\20260614_154128_latest\sample_003\comparison.png`

- Temporal epoch 26 latest full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164137_episode86\20260614_164137_latest_episode86\comparison.mp4`
- Temporal epoch 26 latest sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164022_ckpt_preview\20260614_164022_latest\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164022_ckpt_preview\20260614_164022_latest\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164022_ckpt_preview\20260614_164022_latest\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_164022_ckpt_preview\20260614_164022_latest\sample_003\comparison.png`
- Temporal epoch 27 latest full episode:
  `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174245_episode86\20260614_174245_latest_episode86\comparison.mp4`
- Temporal epoch 27 latest sample previews:
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174118_ckpt_preview\20260614_174118_latest\sample_000\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174118_ckpt_preview\20260614_174118_latest\sample_001\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174118_ckpt_preview\20260614_174118_latest\sample_002\comparison.png`
  - `W:\瀹為獙瀹ら」鐩甛gaze-wam\.codex_tmp\temporal_latest_20260614_174118_ckpt_preview\20260614_174118_latest\sample_003\comparison.png`

## Pending Evidence

- Strict same-epoch `mix-LNLL` vs DSNT-mix A/B is complete through epoch 19, which is the latest
  preserved A-side checkpoint. Later B checkpoints can still be inspected, but should be marked as
  mismatched-budget comparisons unless a new A-side reference is restored or trained.
- A follow-up ablation could change point-NLL weighting only after accepting that `point_nll=0.001`
  remained on the validation-fragmentation plateau through the same-budget epoch 19 comparison.





