"""Render predicted heatmaps from multiple checkpoints side by side for inspection.

For a fixed set of val samples, produces one PNG per sample with columns:
  RGB(+true gaze) | GT temporal-fused heatmap | pred@ckpt1 | pred@ckpt2 | ...

Usage:
  .venv/bin/python scripts/ops/render_pred_heatmap_compare.py \
    --val-zarr data/hot3d_open_val.zarr \
    --ckpt nll0:<dir>/nll0.0_js0.1/checkpoints/latest.ckpt \
    --ckpt nll0.05:<dir>/nll0.05_js0.1/checkpoints/latest.ckpt \
    --ckpt nll1.0:<dir>/nll1.0_js0.1/checkpoints/latest.ckpt \
    --n 5 --out-dir data/outputs/pred_heatmap_compare --device cuda:0
"""
import argparse, os
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval


def heat_to_color(h):
    h = np.clip(h, 0, None)
    m = h.max()
    if m > 0:
        h = h / m
    return cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)


def overlay(rgb_bgr, heat):
    return cv2.addWeighted(rgb_bgr, 0.55, heat_to_color(heat), 0.45, 0)


def to_hw(img):
    t = img.detach().float().cpu()
    if t.ndim == 4:
        t = t[:, 0]
    return t.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-zarr", default="data/hot3d_open_val.zarr")
    ap.add_argument("--ckpt", action="append", required=True, help="label:path")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out-dir", default="data/outputs/pred_heatmap_compare")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labels = [c.split(":", 1)[0] for c in args.ckpt]
    paths = [c.split(":", 1)[1] for c in args.ckpt]

    # load first policy + cfg to build dataset
    import hydra
    pol0, cfg = load_policy_for_eval(checkpoint=paths[0], device=args.device, use_ema=True)
    pol0.eval()
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.open_dataset, resolve=True))
    ds_cfg.dataset_path = args.val_zarr
    dataset = hydra.utils.instantiate(ds_cfg)
    torch.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.n, shuffle=True, num_workers=2)
    batch = next(iter(loader))

    obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
    gaze = batch["gaze_xy"].to(args.device)
    has = batch.get("has_gaze_label")
    obs_dict = dict(obs); obs_dict["gaze_xy"] = gaze
    if has is not None:
        obs_dict["has_gaze_label"] = has.to(args.device)
    gt_heat = None
    if "heatmap_image" in batch:
        gt_heat = to_hw(batch["heatmap_image"])  # [B,H,W]

    # camera key for RGB display: take the last obs frame
    cam_key = cfg.task.open_dataset.camera_key
    cam = batch["obs"][cam_key]  # [B,T,C,H,W] or [B,T,H,W,C]
    cam = cam.detach().float().cpu().numpy()

    def rgb_of(i):
        x = cam[i]
        x = x[-1] if x.ndim == 4 else x  # last time step -> [C,H,W] or [H,W,C]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, -1)
        if x.max() <= 1.5:
            x = x * 255
        return x[:, :, ::-1].astype(np.uint8).copy()  # to BGR

    # predict per checkpoint
    preds = {}
    for lab, pth in zip(labels, paths):
        pol, _ = load_policy_for_eval(checkpoint=pth, device=args.device, use_ema=True)
        pol.eval()
        with torch.no_grad():
            out = pol.predict_heatmap(obs_dict, decode=True)
        preds[lab] = to_hw(out["heatmap_image"])
        del pol
        torch.cuda.empty_cache()

    H = W = int(cfg.task.open_dataset.image_size[0])
    g = gaze.detach().float().cpu().numpy()
    for i in range(args.n):
        rgb = rgb_of(i)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        gx, gy = int(g[i, 0] * (W - 1)), int(g[i, 1] * (H - 1))
        cols = []
        base = rgb.copy(); cv2.circle(base, (gx, gy), 5, (255, 255, 255), 2)
        cv2.putText(base, "RGB+gaze", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cols.append(base)
        if gt_heat is not None:
            gt = overlay(rgb, gt_heat[i]); cv2.putText(gt, "GT-fused", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cols.append(gt)
        for lab in labels:
            ov = overlay(rgb, preds[lab][i])
            cv2.circle(ov, (gx, gy), 5, (255, 255, 255), 1)
            cv2.putText(ov, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cols.append(ov)
        combo = np.concatenate(cols, axis=1)
        out = f"{args.out_dir}/sample_{i:02d}.png"
        cv2.imwrite(out, combo)
        print("wrote", out)


if __name__ == "__main__":
    main()
