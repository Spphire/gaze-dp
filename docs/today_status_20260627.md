# Today's Status — 2026-06-27

End-of-day handoff. Covers everything done today, what is still running on the
server unattended, and what to do tomorrow.

## TL;DR

- Repo is cleaned and pushed: only `gaze-wam-cleanup` on `gaze-dp` (default
  branch) and `master` on `origin`. No rollback tag or sync branch left.
- Server env on `H200-4042` is fully set up: venv, all dependencies, Cosmos
  tokenizer, DINOv3 backbone, gaze-wam editable import working.
- HOT3D val/train zarr **were converted at 224×224, but the mixed_nll/Cosmos
  pipeline needs 256×256.** Both zarrs were deleted and are being
  reconverted right now in the background.
- One major empirical finding: in the `mixed_nll` config the `point_nll` term
  at weight 0.001 is effectively dead — the config is closer to
  `diffusion + 0.1·JS` than its name suggests. See
  `docs/experiments/smoke_mixed_nll_diagnostic.md`.

## What is running unattended on the server right now

**Update at 19:20:** val (24 G) and Cosmos latent stats are both already
done. Only `train` conversion is still running.

| step | status | path / pid |
|---|---|---|
| val zarr 256² | ✅ done (24 G) | `data/hot3d_open_val.zarr` |
| Cosmos latent stats | ✅ done (4.3 K JSON, scale=0.25 confirmed) | `data/outputs/cosmos_heatmap_latent_stats/hot3d_open_ci16x16_random4096_seed42.json` |
| train zarr 256² | ⏳ running | server pid 2624618, log `/tmp/convert_train_256.log` |

The cascade orchestration script attempted earlier was discovered to have
been spawned locally rather than on the server, so it never ran end-to-end.
The three jobs above were instead launched directly via server-side `nohup`
and confirmed with `ps`. Train conversion alone is now the only remaining
unattended job; expected duration ≈ 3 h CPU-bound video decode.

## Tomorrow's first commands

```bash
ssh H200-4042 bash -c '
  ps -p 2624618 -o stat,etime --no-headers 2>/dev/null || echo "train finished"
  tail -3 /tmp/convert_train_256.log
  du -sh /mnt/workspace/shenyibo/gaze-wam/data/hot3d_open_*.zarr 2>/dev/null
  ls -lh /mnt/workspace/shenyibo/gaze-wam/data/outputs/cosmos_heatmap_latent_stats/
'
```

Expected steady state:
- `hot3d_open_val.zarr` ≈ 24 G (was 19 G at 224²; **already finished**)
- `hot3d_open_train.zarr` ≈ 90–100 G (still growing)
- `hot3d_open_ci16x16_random4096_seed42.json` 4.3 K, scale=0.25
  (**already finished**)

Then run the full preflight to make sure all three issues from today are now
resolved:

```bash
cd /mnt/workspace/shenyibo/gaze-wam
export PYTHONPATH=$(pwd):$PYTHONPATH
.venv/bin/python scripts/preflight_gaze_wam.py \
  --config-name train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace \
  --override task.open_dataset_path=data/hot3d_open_train.zarr \
  --override task.robot_dataset_path=null \
  --device cuda:0
```

Want to see `errors: []` and `warnings: []` in the JSON. If it still
complains about image_size, the cascade started train conversion before the
sed edit landed — check `head -3 /tmp/convert_train_256.log` for the actual
flags used.

## Today's findings (one-liners)

- **HOT3D val gaze baselines** — center=0.184, random=0.42, prior-mean=0.093,
  prev-frame=0.012 normalised-L2. Any trained model must clear 0.13 to be
  using image content, 0.08 to be doing more than learning a marginal.
  See `docs/experiments/gaze_baseline_hot3d_val.md`.
- **mixed_nll smoke** (130 steps, val-as-train, mixed_nll config):
  total loss 1.30 → 0.33 (−74 %), entirely driven by the latent diffusion MSE
  (heatmap_loss 1.27 → 0.31, weight 1.0). `point_nll` oscillated 4.5 ± 0.3,
  contributing nothing at weight 0.001. `xy` dropped 8 % despite weight 0,
  i.e. centroid quality is a free side-effect of latent denoising.
  See `docs/experiments/smoke_mixed_nll_diagnostic.md`.
- **Cleanup verified safe**: AST transitive import walk from 6 gaze-wam
  entrypoints touches 32 modules, zero broken imports after deleting
  378 datascalinglaw files. `umi/common/pose_util.py` was the one piece that
  had to be relocated into `diffusion_policy/common/pose_util.py`.

## Suggested next experiments (in order)

Based on the smoke decomposition. None of these were run today.

1. **Bump NLL weight 0.001 → 0.05** with same config, same 130 steps. If
   point_nll trace finally trends down without diffusion exploding, the term
   was just under-weighted; if it diverges, the single-bin gradient is the
   real issue.
2. **Add `xy_loss_weight: 0.05`** on top of mixed_nll. xy already improves
   8 % unsupervised; direct supervision should converge faster.
3. **Change `distribution_mode`** — config currently uses `intensity_softplus`
   (correct for Cosmos-decoded heatmaps). Worth a one-step ablation against
   `intensity_clamp` to check that softplus's bias really helps.
4. Only after the smoke ablation lands: full 8-GPU run via
   `train_scripts/train_gaze_wam_open_only_8gpu_amp.sh`.

## Repo state

- Branch: `gaze-wam-cleanup` at commit `1449162`. Pushed to
  `gaze-dp/gaze-wam-cleanup` (default branch on the remote).
- Removed branches: `gaze-wam-nll-abtest-sync`. Removed tag:
  `rollback/pre-cleanup-20260627`. Their commit `39e1d9c` is still in
  history as an ancestor — accessible by SHA or via `git log gaze-wam-cleanup`.
- `master` (`bd6941e`) still tracks `origin/master` (datascalinglaw upstream,
  read-only).

## Security TODO

Two GitHub PATs were pasted into chat today. Both should already be revoked.
If not yet: https://github.com/settings/tokens.
