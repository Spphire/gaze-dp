from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import zarr


def _imwrite_unicode(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image for '{path}'.")
    encoded.tofile(str(path))


def _parse_int_list(value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    return [int(token) for token in tokens] if tokens else None


def _heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(np.min(heatmap, initial=0.0))
    denom = float(np.max(heatmap, initial=0.0))
    if denom > 1e-12:
        heatmap = heatmap / denom
    return (heatmap * 255.0).round().astype(np.uint8)


def _draw_gaze(image_rgb: np.ndarray, gaze_xy, valid: bool, color=(255, 0, 0)) -> np.ndarray:
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
    cv2.circle(out, (x, y), radius=max(2, min(h, w) // 40), color=color, thickness=-1)
    return out


def _overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, heat_color_rgb, 0.45, 0.0)
    return _draw_gaze(overlay, gaze_xy=gaze_xy, valid=valid)


def _heatmap_panel(heatmap: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    return _draw_gaze(heat_color_rgb, gaze_xy=gaze_xy, valid=valid)


def _title_panel(image_rgb: np.ndarray, title: str) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]
    bar_h = max(22, min(32, h // 9))
    cv2.rectangle(out, (0, 0), (w, bar_h), color=(0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        str(title),
        (8, max(16, bar_h - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, min(0.65, h / 420.0)),
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


def _comparison_strip(
    image_rgb: np.ndarray,
    current_heatmap: np.ndarray,
    temporal_heatmap: np.ndarray,
    gaze_xy,
    valid: bool,
) -> np.ndarray:
    panels = [
        _title_panel(_draw_gaze(image_rgb, gaze_xy, valid), "RGB + current gaze"),
        _title_panel(_heatmap_panel(current_heatmap, gaze_xy, valid), "Current Gaussian"),
        _title_panel(_overlay_heatmap(image_rgb, current_heatmap, gaze_xy, valid), "Current overlay"),
        _title_panel(_heatmap_panel(temporal_heatmap, gaze_xy, valid), "Temporal window"),
        _title_panel(_overlay_heatmap(image_rgb, temporal_heatmap, gaze_xy, valid), "Temporal overlay"),
    ]
    separator = np.full((panels[0].shape[0], max(4, panels[0].shape[1] // 64), 3), 255, dtype=np.uint8)
    parts = []
    for idx, panel in enumerate(panels):
        if idx > 0:
            parts.append(separator)
        parts.append(panel)
    return np.concatenate(parts, axis=1)


def _make_writer(path: pathlib.Path, frame_shape, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    for fourcc_name in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open MP4 writer for '{path}'.")


def _episode_bounds(episode_ends: np.ndarray, episode: int) -> Tuple[int, int]:
    if episode < 0 or episode >= len(episode_ends):
        raise ValueError(f"episode {episode} is out of range for {len(episode_ends)} episodes.")
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return start, end


def _normalized_gaze_to_pixel(gaze_xy: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    return float(gaze_xy[0]) * float(width - 1), float(gaze_xy[1]) * float(height - 1)


def _splat_gaussian(
    heatmap: np.ndarray,
    x: float,
    y: float,
    weight: float,
    sigma_px: float,
    radius_sigma: float = 3.0,
) -> None:
    height, width = heatmap.shape
    radius = max(1, int(round(float(radius_sigma) * float(sigma_px))))
    x0 = max(0, int(np.floor(x)) - radius)
    x1 = min(width, int(np.floor(x)) + radius + 1)
    y0 = max(0, int(np.floor(y)) - radius)
    y1 = min(height, int(np.floor(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * float(sigma_px) ** 2))
    heatmap[y0:y1, x0:x1] += float(weight) * patch.astype(np.float32)


def _window_heatmap(
    gaze_xy: np.ndarray,
    valid_mask: np.ndarray,
    center_local_idx: int,
    image_size: Tuple[int, int],
    window_radius: int,
    beta: float,
    sigma_px: float,
) -> np.ndarray:
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    lo = max(0, int(center_local_idx) - int(window_radius))
    hi = min(len(gaze_xy), int(center_local_idx) + int(window_radius) + 1)
    for local_idx in range(lo, hi):
        if not bool(valid_mask[local_idx]):
            continue
        dt = abs(int(local_idx) - int(center_local_idx))
        weight = np.exp(-float(dt) / max(float(beta), 1e-6))
        x, y = _normalized_gaze_to_pixel(gaze_xy[local_idx], width=width, height=height)
        _splat_gaussian(heatmap, x=x, y=y, weight=weight, sigma_px=sigma_px)
    denom = float(heatmap.sum())
    if denom > 1e-12:
        heatmap /= denom
    return heatmap


def _current_heatmap(
    gaze_xy,
    valid: bool,
    image_size: Tuple[int, int],
    sigma_px: float,
) -> np.ndarray:
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    if valid:
        x, y = _normalized_gaze_to_pixel(np.asarray(gaze_xy), width=width, height=height)
        _splat_gaussian(heatmap, x=x, y=y, weight=1.0, sigma_px=sigma_px)
        denom = float(heatmap.sum())
        if denom > 1e-12:
            heatmap /= denom
    return heatmap


def preview_temporal_heatmap(
    dataset_path: str,
    output_dir: str,
    episode: int,
    camera_key: str = "camera0_rgb",
    gaze_key: str = "gaze_xy",
    valid_key: str = "has_gaze_label",
    fps: float = 15.0,
    window_radius: int = 30,
    beta: float = 10.0,
    sigma_px: float = 6.0,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    still_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    root = zarr.open(dataset_path, mode="r")
    data = root["data"] if "data" in root else root
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    start, end = _episode_bounds(episode_ends, int(episode))
    frame_indices = list(range(start, end, int(frame_stride)))
    if max_frames is not None:
        frame_indices = frame_indices[: int(max_frames)]
    if not frame_indices:
        raise ValueError(f"episode {episode} produced no frames.")

    rgb_array = data[camera_key]
    gaze_episode = np.asarray(data[gaze_key][start:end], dtype=np.float32)
    if valid_key in data:
        valid_episode = np.asarray(data[valid_key][start:end], dtype=np.bool_)
    else:
        valid_episode = np.ones((end - start,), dtype=np.bool_)

    first_rgb = np.asarray(rgb_array[frame_indices[0]])
    height, width = int(first_rgb.shape[0]), int(first_rgb.shape[1])
    image_size = (height, width)

    out_dir = pathlib.Path(output_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "comparison.mp4"
    still_indices = list(still_indices or [])
    if not still_indices:
        count = len(frame_indices)
        still_indices = sorted(set([0, count // 4, count // 2, (count * 3) // 4, count - 1]))
    still_set = set(int(idx) for idx in still_indices if 0 <= int(idx) < len(frame_indices))

    writer = None
    first_frame_shape = None
    frame_summaries = []
    start_time = time.time()

    for rendered_idx, zarr_idx in enumerate(frame_indices):
        local_idx = int(zarr_idx) - start
        image_rgb = np.asarray(rgb_array[zarr_idx])
        if image_rgb.shape[-1] == 4:
            image_rgb = image_rgb[..., :3]
        image_rgb = image_rgb.astype(np.uint8)
        gaze_xy = gaze_episode[local_idx]
        valid = bool(valid_episode[local_idx])
        current = _current_heatmap(gaze_xy, valid=valid, image_size=image_size, sigma_px=float(sigma_px))
        temporal = _window_heatmap(
            gaze_episode,
            valid_episode,
            center_local_idx=local_idx,
            image_size=image_size,
            window_radius=int(window_radius),
            beta=float(beta),
            sigma_px=float(sigma_px),
        )
        frame_rgb = _comparison_strip(
            image_rgb=image_rgb,
            current_heatmap=current,
            temporal_heatmap=temporal,
            gaze_xy=gaze_xy,
            valid=valid,
        )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if writer is None:
            first_frame_shape = [int(v) for v in frame_bgr.shape]
            writer = _make_writer(video_path, frame_bgr.shape, fps=float(fps))
        writer.write(frame_bgr)
        if rendered_idx in still_set:
            still_path = frames_dir / f"frame_{rendered_idx:06d}.png"
            _imwrite_unicode(still_path, frame_bgr)
            frame_summaries.append(
                {
                    "frame_index": int(rendered_idx),
                    "zarr_index": int(zarr_idx),
                    "path": str(still_path),
                    "gaze_xy": [float(v) for v in gaze_xy.tolist()],
                    "has_gaze_label": valid,
                    "temporal_nonzero_pixels": int(np.count_nonzero(temporal > 0.0)),
                    "temporal_peak": float(np.max(temporal, initial=0.0)),
                }
            )

    if writer is None:
        raise RuntimeError("No frames were rendered.")
    writer.release()
    summary = {
        "dataset_path": str(dataset_path),
        "episode": int(episode),
        "episode_start_zarr_index": int(start),
        "episode_end_zarr_index": int(end),
        "episode_length_frames": int(end - start),
        "rendered_frames": int(len(frame_indices)),
        "frame_stride": int(frame_stride),
        "max_frames": None if max_frames is None else int(max_frames),
        "zarr_index_start": int(frame_indices[0]),
        "zarr_index_end": int(frame_indices[-1]),
        "camera_key": str(camera_key),
        "gaze_key": str(gaze_key),
        "valid_key": str(valid_key),
        "image_size": [int(height), int(width)],
        "window_radius": int(window_radius),
        "beta": float(beta),
        "sigma_px": float(sigma_px),
        "temporal_weight": "exp(-abs(delta_frame)/beta)",
        "video_fps": float(fps),
        "video_path": str(video_path),
        "first_frame_shape": first_frame_shape,
        "render_seconds": float(time.time() - start_time),
        "still_frames": frame_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Render current-point Gaussian vs temporal-window pseudo heatmap for one episode."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--valid-key", default="has_gaze_label")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--window-radius", type=int, default=30)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--sigma-px", type=float, default=6.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--still-indices", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    summary = preview_temporal_heatmap(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        episode=args.episode,
        camera_key=args.camera_key,
        gaze_key=args.gaze_key,
        valid_key=args.valid_key,
        fps=args.fps,
        window_radius=args.window_radius,
        beta=args.beta,
        sigma_px=args.sigma_px,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        still_indices=_parse_int_list(args.still_indices),
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
