# HOT3D Val Gaze Baselines

Computed on `data/hot3d_open_val.zarr` (40 sequences, 136,951 frames).
All L2 errors are in normalized coordinates [0,1]² unless noted.

## Numbers

| Baseline | L2 (norm) | px on 224² | Cheat level |
|---|---|---|---|
| ① center (0.5, 0.5) | 0.1841 | ~58 | none |
| ⑤ uniform random in [0,1]² | 0.4156 | ~131 | none (worst) |
| ⑥ random sample from gaze dist | 0.1330 | ~42 | knows distribution |
| ② global mean = (0.580, 0.640) | 0.0929 | ~29 | cheats: val stats |
| ③ per-episode mean | 0.0824 | ~26 | cheats: per-segment mean |
| ④ previous-frame copy | 0.0116 | ~3.7 | cheats: prev-frame GT |

## Reference

- gaze x: mean=0.580, std=0.077
- gaze y: mean=0.640, std=0.077
- range: x∈[0.21, 0.95], y∈[0.20, 0.85]
- 1 std ≈ 17 px on 224²
- L2 0.1 ≈ 31.7 px

## Interpretation

1. **True floor (no leakage): center prediction = 0.184**.
   Any model below this is learning the conditional p(gaze | image), not just the marginal.

2. **Distribution-aware floor ≈ 0.13**. A model that learned only the gaze prior ("gaze is around (0.58, 0.64)") would land here. **Below 0.13 means the model uses image content.**

3. **Strong reference ≈ 0.08**. Beating the per-episode mean means the model tracks gaze movement within a segment, not just the segment average.

4. **Previous-frame copy = 0.012**. Sets an aspirational upper bound for what tightly-coupled temporal models could approach. The mixed_nll setup uses n_obs_steps=2 (visual prev frame in input but no prev gaze label), so 0.02-0.05 is a reasonable realistic target.

## Important caveat for robot dataset

HOT3D is head-mounted Aria, third-person ego-view, hand-task focused → gaze biased to (0.58, 0.64).
A robot dataset shot from a different viewpoint (e.g. external camera) will have a DIFFERENT gaze prior.
Do NOT compare numbers across datasets without re-running baselines on each.
