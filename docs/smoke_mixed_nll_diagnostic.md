# Smoke Training Diagnostic — mixed_nll (val-as-train, 130 steps)

**Date:** 2026-06-27
**Config:** `train_gaze_wam_open_only_cosmos_temporal_mixed_nll_workspace`
**Setup:** 1× GPU AMP bf16, batch=16, val zarr as train data (sanity only), 130 logging steps × 4 grad accum
**Loss config (effective weights):**
- heatmap_loss_weight: 1.0 (latent diffusion MSE)
- heatmap_xy_loss_weight: **0.0**
- heatmap_point_nll_loss_weight: **0.001**
- heatmap_js_loss_weight: **0.1**
- final_loss_enabled: True (alpha_cumprod weighting)

## Loss component trajectory

| Step range | total | heatmap (diffusion) | xy | point_nll | js |
|---|---|---|---|---|---|
| 0-5 | 1.295 | 1.266 | 0.083 | **4.528** | 0.244 |
| 10-15 | 1.086 | 1.060 | 0.076 | **4.189** | 0.221 |
| 30-35 | 0.998 | 0.971 | 0.080 | **4.561** | 0.233 |
| 60-65 | 0.820 | 0.790 | 0.086 | **4.726** | 0.252 |
| 100-105 | 0.402 | 0.375 | 0.080 | **4.383** | 0.229 |
| 120-125 | 0.333 | 0.306 | 0.076 | **4.420** | 0.225 |

**Δ:**  total: −74%  |  heatmap MSE: −76%  |  point_nll: ~0  |  js: −8%  |  xy: −8%

## Findings

### ① Latent diffusion drives ~all of the learning
`heatmap_loss` (weight=1.0) accounts for ~99% of the total drop. The model is doing pure cosmos-latent denoising, exactly as the diffusion branch in compute_loss_components expects.

### ② point_nll is effectively dead
`point_nll` started at 4.53 and is at 4.42 after 130 logging steps (4.53→4.19→4.56→4.73→4.38→4.42, oscillating). The 0.001 weight × single-bin gradient is too weak to register. This is direct empirical evidence for what was already suspected from code review.

### ③ js makes a tiny contribution
js: 0.244 → 0.225 (−8%). With weight 0.1 it shapes the spatial distribution mildly, but does not lead the optimization.

### ④ xy improves despite zero weight
xy: 0.083 → 0.076 (−8%). This is unsupervised — xy_loss_weight=0.0 means xy gets no gradient. Yet DSNT-derived xy got slightly better because **good latent denoising → decoded heatmap centroid lands closer to GT**. Indirect signal that the diffusion path *can* localize, just not as fast as direct xy supervision would.

## Implications for the mixed_nll design

1. The configuration **mixed_nll = diffusion ⊕ (0·xy + 0.001·NLL + 0.1·JS)** essentially reduces to **diffusion ⊕ 0.1·JS**. The `mixed_nll` name is misleading: NLL is not contributing.

2. To get NLL to actually do something, options (in order of risk/reward):
   - **Easy fix:** raise weight to 0.01 or 0.05 and re-measure. If it explodes, the implementation is unstable; if it joins the optimization, it's just been under-weighted.
   - **Structural fix:** replace single-bin floor() target with bilinear-interpolated probability at the continuous (x,y). Var of gradient drops, weight can go up.
   - **Replace fix:** drop NLL entirely; rely on JS. Less expressive in theory, but JS is what's actually doing the shape-matching work here.

3. Adding a small `xy_loss_weight` (e.g. 0.05) almost certainly helps. The xy signal is currently *free* — it improves naturally — but a small direct gradient on the centroid would converge faster than indirect-via-latent.

## Reproducibility

- Run dir: `data/outputs/smoke_mixed_nll_v3/`
- Loss CSV: `data/outputs/smoke_mixed_nll_v3/loss_trace.csv` (130 rows)
- wandb: https://wandb.ai/cwen/gaze_wam/runs/yk29ufhf
- Log: `data/outputs/smoke_mixed_nll_v3/logs.json.txt`
