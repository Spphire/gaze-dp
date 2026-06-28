"""Evaluate a Gaze-WAM checkpoint's predicted heatmaps with multimodal-aware metrics.

For each of N sampled val frames, run the policy's iterative heatmap denoising,
decode to a full-res heatmap image, and compute:

  argmax_l2  : L2 between the predicted heatmap's argmax peak and the true gaze
               point (normalized coords). Multimodal-friendly — only the
               dominant peak is scored, so multi-peak predictions are not
               penalized as long as one peak is on the gaze.
  point_nll  : -log p(true gaze pixel) under the normalized predicted heatmap.
               Coverage of the true point. Lower is better.
  peak_count : number of local maxima above 0.5*max (min-separated). >1 means
               the prediction is multimodal. We WANT this to stay > 1, not
               collapse to 1 as the NLL weight grows.
  entropy    : spatial entropy of the normalized heatmap / log(H*W) in [0,1].
               Higher = more spread. A collapse to a single sharp peak drives
               this toward 0.

Usage:
  .venv/bin/python scripts/ops/eval_multimodal_heatmap.py \
    --checkpoint <run>/checkpoints/latest.ckpt \
    --val-zarr data/hot3d_open_val.zarr \
    --n-samples 128 --device cuda:0 \
    --output-json <run>/multimodal_eval.json
"""
import argparse
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import maximum_filter

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval


def to_hw(img):
    t = img.detach().float().cpu()
    if t.ndim == 4:
        t = t[:, 0]
    elif t.ndim == 3 and t.shape[0] == 1:
        t = t
    return t  # [B,H,W]


def count_peaks(p, rel=0.6, min_dist=16):
    """Count well-separated dominant modes (local maxima >= rel*max).

    Tightened defaults (rel 0.6, min_dist 16) so a diffuse/grainy heatmap does
    not register dozens of noise peaks; entropy is the smoother multimodality
    signal, peak_count is the discrete companion.
    """
    mx = float(p.max())
    if mx <= 0:
        return 0
    fp = maximum_filter(p, size=min_dist)
    peaks = (p == fp) & (p >= rel * mx)
    return int(peaks.sum())


def spatial_entropy(p):
    flat = p.flatten().astype(np.float64)
    s = flat.sum()
    if s <= 0:
        return 0.0
    q = flat / s
    nz = q[q > 0]
    ent = -np.sum(nz * np.log(nz))
    return float(ent / np.log(len(flat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-zarr", default="data/hot3d_open_val.zarr")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    policy, cfg = load_policy_for_eval(checkpoint=args.checkpoint, device=args.device, use_ema=True)
    policy.eval()

    # Build val dataset from the checkpoint cfg's open_dataset spec, pointed at val zarr.
    import hydra
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.open_dataset, resolve=True))
    ds_cfg.dataset_path = args.val_zarr
    dataset = hydra.utils.instantiate(ds_cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    H = W = int(cfg.task.open_dataset.image_size[0])
    yy, xx = np.mgrid[0:H, 0:W]

    argmax_l2, point_nll, peak_counts, entropies = [], [], [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
            gaze = batch["gaze_xy"].to(args.device)
            has = batch.get("has_gaze_label")
            obs_dict = dict(obs)
            obs_dict["gaze_xy"] = gaze
            if has is not None:
                obs_dict["has_gaze_label"] = has.to(args.device)
            pred = policy.predict_heatmap(obs_dict, decode=True)
            himg = to_hw(pred["heatmap_image"]).numpy()  # [B,H,W]
            g = gaze.detach().float().cpu().numpy()       # [B,2] normalized xy
            B = himg.shape[0]
            for i in range(B):
                p = himg[i].astype(np.float64)
                p = np.clip(p, 0, None)
                # argmax peak
                ay, ax = np.unravel_index(int(np.argmax(p)), p.shape)
                px, py = (ax + 0.5) / W, (ay + 0.5) / H
                argmax_l2.append(float(np.hypot(px - g[i, 0], py - g[i, 1])))
                # point nll
                s = p.sum()
                pn = p / s if s > 0 else p
                tx = min(W - 1, max(0, int(g[i, 0] * W)))
                ty = min(H - 1, max(0, int(g[i, 1] * H)))
                point_nll.append(float(-np.log(max(pn[ty, tx], 1e-8))))
                # peaks + entropy
                peak_counts.append(count_peaks(p))
                entropies.append(spatial_entropy(p))
            n += B
            if n >= args.n_samples:
                break

    res = {
        "checkpoint": args.checkpoint,
        "val_zarr": args.val_zarr,
        "n_samples": int(n),
        "argmax_l2_mean": float(np.mean(argmax_l2)),
        "argmax_l2_median": float(np.median(argmax_l2)),
        "point_nll_mean": float(np.mean(point_nll)),
        "peak_count_mean": float(np.mean(peak_counts)),
        "peak_count_frac_multi": float(np.mean([c > 1 for c in peak_counts])),
        "entropy_mean": float(np.mean(entropies)),
    }
    print(json.dumps(res, indent=2))
    if args.output_json:
        import os
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        json.dump(res, open(args.output_json, "w"), indent=2)


if __name__ == "__main__":
    main()
