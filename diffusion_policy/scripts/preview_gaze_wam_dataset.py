from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Sequence


def _ensure_preview_runtime():
    global GazeWamOpenDataset
    global GazeWamRobotDataset
    global HeatmapTokenCodec
    global as_optional_gaze_wam_key
    global cv2
    global np
    global torch
    try:
        return cv2
    except NameError:
        import cv2 as _cv2
        import numpy as _np
        import torch as _torch
        from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key as _as_optional_gaze_wam_key
        from diffusion_policy.dataset.gaze_wam_dataset import (
            GazeWamOpenDataset as _GazeWamOpenDataset,
            GazeWamRobotDataset as _GazeWamRobotDataset,
        )
        from diffusion_policy.model.gaze_wam.heatmap_codec import HeatmapTokenCodec as _HeatmapTokenCodec

        GazeWamOpenDataset = _GazeWamOpenDataset
        GazeWamRobotDataset = _GazeWamRobotDataset
        HeatmapTokenCodec = _HeatmapTokenCodec
        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        cv2 = _cv2
        np = _np
        torch = _torch
        return cv2


def _imwrite_unicode(path: pathlib.Path, image: np.ndarray) -> None:
    _ensure_preview_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for '{path}'.")
    encoded.tofile(str(path))


def _to_rgb_uint8(obs_image: torch.Tensor, frame_index: int = -1) -> np.ndarray:
    _ensure_preview_runtime()
    image = obs_image.detach().cpu().numpy()
    if image.ndim != 4:
        raise ValueError(f"Expected obs image [T,C,H,W], got {image.shape}.")
    frame = image[frame_index]
    if frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame[:3], 0, -1)
    elif frame.shape[-1] in (1, 3, 4):
        frame = frame[..., :3]
    else:
        raise ValueError(f"Cannot infer channel dimension for frame shape {frame.shape}.")
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).round().astype(np.uint8)


def _heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    _ensure_preview_runtime()
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - heatmap.min(initial=0.0)
    denom = heatmap.max(initial=0.0)
    if denom > 1e-12:
        heatmap = heatmap / denom
    return (heatmap * 255.0).round().astype(np.uint8)


def _draw_gaze(image_rgb: np.ndarray, gaze_xy: np.ndarray, valid: bool) -> np.ndarray:
    _ensure_preview_runtime()
    out = image_rgb.copy()
    if not valid:
        return out
    h, w = out.shape[:2]
    x = int(round(float(gaze_xy[0]) * (w - 1)))
    y = int(round(float(gaze_xy[1]) * (h - 1)))
    cv2.drawMarker(
        out,
        (x, y),
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(8, min(h, w) // 10),
        thickness=2,
    )
    cv2.circle(out, (x, y), radius=max(2, min(h, w) // 40), color=(255, 0, 0), thickness=-1)
    return out


def _make_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, gaze_xy: np.ndarray, valid: bool) -> np.ndarray:
    _ensure_preview_runtime()
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, heat_color_rgb, 0.45, 0.0)
    return _draw_gaze(overlay, gaze_xy=gaze_xy, valid=valid)


def _sample_heatmap_image(sample, heatmap_tokens: torch.Tensor, codec: HeatmapTokenCodec, image_size, decode_method):
    _ensure_preview_runtime()
    if "heatmap_image" in sample:
        heatmap = sample["heatmap_image"].detach().cpu().numpy()
        heatmap = np.asarray(heatmap).squeeze()
        if heatmap.ndim != 2:
            raise ValueError(f"Sample heatmap_image must squeeze to [H,W], got {heatmap.shape}.")
        return heatmap, "sample_heatmap_image"
    decoded = codec.decode_tokens(
        heatmap_tokens.to(dtype=torch.float32),
        image_size=image_size,
        method=decode_method,
    )
    heatmap = np.asarray(decoded.detach().cpu().numpy()).squeeze()
    if heatmap.ndim != 2:
        raise ValueError(f"Decoded heatmap must be [H,W] after squeeze, got {heatmap.shape}.")
    return heatmap, "decoded_tokens"


def _build_dataset(
    dataset_path: str,
    dataset_type: str,
    camera_key: str,
    gaze_key: Optional[str],
    heatmap_key: Optional[str],
    action_abs_key: str,
    tcp_pose_key: str,
    gripper_key: str,
    n_obs_steps: int,
    action_horizon: int,
    n_latency_steps: int,
    image_size: Sequence[int],
    image_resize_mode: str,
    heatmap_token_grid: Sequence[int],
    gaze_is_normalized: bool,
):
    common_kwargs = dict(
        dataset_path=dataset_path,
        camera_key=camera_key,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
        n_latency_steps=n_latency_steps,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        heatmap_token_grid=heatmap_token_grid,
        gaze_is_normalized=gaze_is_normalized,
        action_padding=True,
    )
    if dataset_type == "robot":
        return GazeWamRobotDataset(
            action_abs_key=action_abs_key,
            tcp_pose_key=tcp_pose_key,
            gripper_key=gripper_key,
            **common_kwargs,
        )
    if dataset_type == "open":
        return GazeWamOpenDataset(**common_kwargs)
    raise ValueError("dataset_type must be robot or open.")


def preview_gaze_wam_dataset(
    dataset_path: str,
    dataset_type: str,
    output_dir: str,
    sample_index: int = 0,
    camera_key: str = "camera0_rgb",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = "gaze_heatmap",
    action_abs_key: str = "action_abs_tcp",
    tcp_pose_key: str = "tcp_pose_abs",
    gripper_key: str = "gripper_width",
    n_obs_steps: int = 2,
    action_horizon: int = 16,
    n_latency_steps: int = 0,
    image_size: Sequence[int] = (256, 256),
    image_resize_mode: str = "stretch",
    heatmap_token_grid: Sequence[int] = (16, 16),
    heatmap_sigma_tokens: float = 1.25,
    gaze_is_normalized: bool = True,
    decode_method: str = "gaussian_splat",
) -> Dict[str, object]:
    """Write RGB / heatmap / overlay previews for one Gaze-WAM dataset sample."""
    _ensure_preview_runtime()
    gaze_key = as_optional_gaze_wam_key(gaze_key)
    heatmap_key = as_optional_gaze_wam_key(heatmap_key)
    dataset = _build_dataset(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        camera_key=camera_key,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
        action_abs_key=action_abs_key,
        tcp_pose_key=tcp_pose_key,
        gripper_key=gripper_key,
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
        n_latency_steps=n_latency_steps,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        heatmap_token_grid=heatmap_token_grid,
        gaze_is_normalized=gaze_is_normalized,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset adapter produced zero previewable samples.")
    sample_index = int(min(max(sample_index, 0), len(dataset) - 1))
    sample = dataset[sample_index]
    image_rgb = _to_rgb_uint8(sample["obs"][camera_key], frame_index=-1)
    heatmap_tokens = sample["heatmap"]
    if heatmap_tokens.ndim == 4:
        heatmap_tokens = heatmap_tokens[0]
    if heatmap_tokens.ndim == 3 and heatmap_tokens.shape[0] == 1:
        heatmap_tokens = heatmap_tokens[0]
    if heatmap_tokens.ndim != 2:
        raise ValueError(f"Expected heatmap tokens [N,1] after squeeze, got {heatmap_tokens.shape}.")
    codec = HeatmapTokenCodec(
        token_grid=heatmap_token_grid,
        image_size=image_size,
        sigma_tokens=heatmap_sigma_tokens,
    )
    heatmap, heatmap_source = _sample_heatmap_image(
        sample,
        heatmap_tokens=heatmap_tokens,
        codec=codec,
        image_size=image_size,
        decode_method=decode_method,
    )
    gaze_xy = sample["gaze_xy"].detach().cpu().numpy().astype(np.float32)
    has_gaze_label = bool(sample["has_gaze_label"].item())
    overlay_rgb = _make_overlay(image_rgb, heatmap, gaze_xy=gaze_xy, valid=has_gaze_label)
    heatmap_u8 = _heatmap_to_uint8(heatmap)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / "rgb.png"
    gaze_path = out_dir / "rgb_gaze.png"
    heatmap_path = out_dir / "heatmap.png"
    overlay_path = out_dir / "overlay.png"
    summary_path = out_dir / "summary.json"

    _imwrite_unicode(rgb_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    _imwrite_unicode(gaze_path, cv2.cvtColor(_draw_gaze(image_rgb, gaze_xy, has_gaze_label), cv2.COLOR_RGB2BGR))
    _imwrite_unicode(heatmap_path, heatmap_color_bgr)
    _imwrite_unicode(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    token_argmax = int(heatmap_tokens.reshape(-1).argmax().item())
    grid_h, grid_w = int(heatmap_token_grid[0]), int(heatmap_token_grid[1])
    token_y = token_argmax // grid_w
    token_x = token_argmax % grid_w
    summary = {
        "dataset_path": str(dataset_path),
        "dataset_type": dataset_type,
        "dataset_len": int(len(dataset)),
        "sample_index": int(sample_index),
        "camera_key": camera_key,
        "image_resize_mode": image_resize_mode,
        "image_shape": [int(v) for v in image_rgb.shape],
        "heatmap_token_shape": [int(v) for v in heatmap_tokens.shape],
        "heatmap_image_shape": [int(v) for v in heatmap.shape],
        "heatmap_image_source": heatmap_source,
        "heatmap_token_argmax": [int(token_y), int(token_x)],
        "gaze_xy": [float(v) for v in gaze_xy.tolist()],
        "has_gaze_label": has_gaze_label,
        "has_action": bool(sample["has_action"].item()),
        "has_heatmap": bool(sample["has_heatmap"].item()),
        "use_gaze_condition": bool(sample["use_gaze_condition"].item()),
        "is_gaze_condition_dropped": bool(sample["is_gaze_condition_dropped"].item()),
        "paths": {
            "rgb": str(rgb_path),
            "rgb_gaze": str(gaze_path),
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
            "summary": str(summary_path),
        },
    }
    if "action" in sample:
        summary["action_shape"] = [int(v) for v in sample["action"].shape]
    if "action_abs" in sample:
        summary["action_abs_shape"] = [int(v) for v in sample["action_abs"].shape]
    if "action_base_abs" in sample:
        summary["action_base_abs"] = [float(v) for v in sample["action_base_abs"].tolist()]

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Preview one Gaze-WAM robot/open zarr sample with gaze and heatmap overlays."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-type", choices=("robot", "open"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--heatmap-key", default="gaze_heatmap")
    parser.add_argument("--action-abs-key", default="action_abs_tcp")
    parser.add_argument("--tcp-pose-key", default="tcp_pose_abs")
    parser.add_argument("--gripper-key", default="gripper_width")
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--n-latency-steps", type=int, default=0)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument(
        "--image-resize-mode",
        choices=("stretch",),
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--heatmap-token-grid", type=int, nargs=2, default=(16, 16), metavar=("H", "W"))
    parser.add_argument("--heatmap-sigma-tokens", type=float, default=1.25)
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-method", choices=("gaussian_splat", "bilinear"), default="gaussian_splat")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = preview_gaze_wam_dataset(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
        camera_key=args.camera_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        action_abs_key=args.action_abs_key,
        tcp_pose_key=args.tcp_pose_key,
        gripper_key=args.gripper_key,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        n_latency_steps=args.n_latency_steps,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        heatmap_token_grid=args.heatmap_token_grid,
        heatmap_sigma_tokens=args.heatmap_sigma_tokens,
        gaze_is_normalized=args.gaze_is_normalized,
        decode_method=args.decode_method,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
