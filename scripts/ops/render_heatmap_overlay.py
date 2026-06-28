"""Render bidirectional temporal-fusion gaze heatmap overlays for inspection.

Reproduces diffusion_policy/dataset/gaze_wam_dataset.py::_sample_temporal_heatmap_image
exactly (exp(-dt/beta) decay, current_weight boost, gaussian splat), overlaid on
the RGB frame. Saves side-by-side PNGs.
"""
import argparse
import numpy as np
import zarr
import cv2


def normalize_gaze(g):
    return np.asarray(g, dtype=np.float32).reshape(2)


def splat(heatmap, gaze_xy, weight, sigma_px, radius_sigma=3.0):
    h, w = heatmap.shape
    x = float(gaze_xy[0]) * (w - 1)
    y = float(gaze_xy[1]) * (h - 1)
    r = max(1, int(round(radius_sigma * sigma_px)))
    x0, x1 = max(0, int(np.floor(x)) - r), min(w, int(np.floor(x)) + r + 1)
    y0, y1 = max(0, int(np.floor(y)) - r), min(h, int(np.floor(y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma_px ** 2))
    heatmap[y0:y1, x0:x1] += float(weight) * patch.astype(np.float32)


def fuse(gaze, ep_start, ep_end, cur, mode, radius, beta, sigma_px, current_weight, hw):
    h, w = hw
    heat = np.zeros((h, w), dtype=np.float32)
    if mode == "causal":
        lo, hi = max(ep_start, cur - radius), cur + 1
    else:  # bidirectional
        lo, hi = max(ep_start, cur - radius), min(ep_end, cur + radius + 1)
    for s in range(lo, hi):
        dt = abs(s - cur)
        wgt = float(np.exp(-dt / beta))
        if dt == 0:
            wgt *= current_weight
        splat(heat, normalize_gaze(gaze[s]), wgt, sigma_px)
    m = heat.max()
    if m > 1e-12:
        heat /= m
    return heat


def overlay(rgb, heat):
    cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(rgb, 0.55, cmap, 0.45, 0)
    return blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/hot3d_open_train.zarr")
    ap.add_argument("--out-dir", default="data/outputs/heatmap_overlay_preview")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--radius", type=int, default=30)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--sigma-px", type=float, default=6.0)
    ap.add_argument("--current-weight", type=float, default=2.0)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    z = zarr.open(args.zarr, mode="r")
    rgb_arr = z["data/camera0_rgb"]
    gaze = z["data/gaze_xy"][:]
    ep_ends = z["meta/episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    H, W = rgb_arr.shape[1], rgb_arr.shape[2]

    # default: pick frames in the first episode, spaced out, where gaze moves a lot
    if not args.frames:
        s, e = int(ep_starts[0]), int(ep_ends[0])
        # find frames with large local gaze displacement (saccades)
        win = gaze[s:e]
        disp = np.linalg.norm(np.diff(win, axis=0), axis=1)
        # cumulative motion over +-15 frames
        motion = np.convolve(disp, np.ones(30), mode="same")
        cand = np.argsort(motion)[-8:] + s
        args.frames = sorted(int(c) for c in cand[:6])

    print(f"zarr={args.zarr} HxW={H}x{W} frames={args.frames}")
    print(f"params: radius={args.radius} beta={args.beta} sigma_px={args.sigma_px} cur_w={args.current_weight}")

    for cur in args.frames:
        # find episode
        ei = int(np.searchsorted(ep_ends, cur, side="right"))
        ep_s, ep_e = int(ep_starts[ei]), int(ep_ends[ei])
        rgb = np.asarray(rgb_arr[cur])[:, :, ::-1].copy()  # to BGR for cv2

        heat_bi = fuse(gaze, ep_s, ep_e, cur, "bidirectional", args.radius, args.beta, args.sigma_px, args.current_weight, (H, W))
        heat_ca = fuse(gaze, ep_s, ep_e, cur, "causal", args.radius, args.beta, args.sigma_px, args.current_weight, (H, W))

        ov_bi = overlay(rgb, heat_bi)
        ov_ca = overlay(rgb, heat_ca)

        # mark current single gaze point
        gx, gy = int(gaze[cur][0] * (W - 1)), int(gaze[cur][1] * (H - 1))
        for img in (rgb, ov_bi, ov_ca):
            cv2.circle(img, (gx, gy), 4, (255, 255, 255), 1)

        # count distinct gaze clusters in the bidir window for "blob" sanity
        lo, hi = max(ep_s, cur - args.radius), min(ep_e, cur + args.radius + 1)
        win = gaze[lo:hi]
        spread = float(np.linalg.norm(win.std(axis=0)))

        labels = []
        for name, img in [("rgb", rgb), ("bidir", ov_bi), ("causal", ov_ca)]:
            tag = img.copy()
            cv2.putText(tag, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            labels.append(tag)
        combo = np.concatenate(labels, axis=1)
        cv2.putText(combo, f"frame {cur} ep{ei} win[{lo}:{hi}] spread={spread:.3f}", (5, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        out = f"{args.out_dir}/frame_{cur:06d}.png"
        cv2.imwrite(out, combo)
        print(f"  wrote {out}  (bidir window spread={spread:.3f})")


if __name__ == "__main__":
    main()
