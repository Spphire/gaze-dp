from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


HEATMAP_METHOD_ALIASES = {
    "gaussian": "gaussian_point",
    "point_gaussian": "gaussian_point",
    "gaussian_point": "gaussian_point",
}
HEATMAP_METHODS = tuple(sorted(set(HEATMAP_METHOD_ALIASES.values())))
HEATMAP_STORAGE_MODES = ("none", "token", "dense")
DEFAULT_HEATMAP_TOKEN_GRID = (16, 16)


def _ensure_hot3d_runtime():
    global HeatmapTokenCodec
    global check_gaze_bounds
    global cv2
    global gaussian_heatmaps_from_points
    global np
    global torch
    global zarr
    try:
        HeatmapTokenCodec
        torch
        return zarr
    except NameError:
        import cv2 as _cv2
        import numpy as _np
        import torch as _torch
        import zarr as _zarr
        from diffusion_policy.common.gaze_utils import (
            check_gaze_bounds as _check_gaze_bounds,
            gaussian_heatmaps_from_points as _gaussian_heatmaps_from_points,
        )
        from diffusion_policy.model.gaze_wam.heatmap_codec import (
            HeatmapTokenCodec as _HeatmapTokenCodec,
        )

        HeatmapTokenCodec = _HeatmapTokenCodec
        check_gaze_bounds = _check_gaze_bounds
        gaussian_heatmaps_from_points = _gaussian_heatmaps_from_points
        cv2 = _cv2
        np = _np
        torch = _torch
        zarr = _zarr
        return zarr


@dataclass(frozen=True)
class Hot3dSequence:
    sequence_id: str
    directory: pathlib.Path
    video_path: pathlib.Path
    gaze_csv_path: pathlib.Path
    summary_path: pathlib.Path


@dataclass(frozen=True)
class Hot3dFrame:
    sequence_id: str
    frame_index: int
    timestamp_ns: int
    gaze_xy: Tuple[float, float]


def _read_sequence_file(path: pathlib.Path) -> List[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.append(line)
    return values


def _selected_sequence_ids(
    processed_root: pathlib.Path,
    sequence: Optional[Sequence[str]] = None,
    sequence_file: Optional[str] = None,
    limit_sequences: Optional[int] = None,
) -> Optional[set[str]]:
    selected: List[str] = []
    if sequence:
        selected.extend(str(item) for item in sequence)
    if sequence_file:
        selected.extend(_read_sequence_file(pathlib.Path(sequence_file)))
    if not selected:
        return None
    if limit_sequences is not None:
        selected = selected[: int(limit_sequences)]
    available = {path.name for path in processed_root.iterdir() if path.is_dir()}
    missing = sorted(set(selected) - available)
    if missing:
        raise FileNotFoundError(f"Selected HOT3D processed sequences are missing: {missing[:10]}.")
    return set(selected)


def _discover_sequences(
    processed_root: pathlib.Path,
    sequence: Optional[Sequence[str]] = None,
    sequence_file: Optional[str] = None,
    limit_sequences: Optional[int] = None,
) -> List[Hot3dSequence]:
    selected = _selected_sequence_ids(
        processed_root=processed_root,
        sequence=sequence,
        sequence_file=sequence_file,
        limit_sequences=limit_sequences,
    )
    sequences = []
    for directory in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        if selected is not None and directory.name not in selected:
            continue
        video_path = directory / "raw_rgb.mp4"
        gaze_csv_path = directory / "gaze_projected_raw_rgb_normalized.csv"
        summary_path = directory / "processing_summary.json"
        if not video_path.exists() or not gaze_csv_path.exists() or not summary_path.exists():
            continue
        sequences.append(
            Hot3dSequence(
                sequence_id=directory.name,
                directory=directory,
                video_path=video_path,
                gaze_csv_path=gaze_csv_path,
                summary_path=summary_path,
            )
        )
    if limit_sequences is not None and selected is None:
        sequences = sequences[: int(limit_sequences)]
    if not sequences:
        raise FileNotFoundError(f"No processed HOT3D sequences found under '{processed_root}'.")
    return sequences


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _as_int(value: object, name: str) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {name}: {value!r}.") from exc


def _as_float(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for {name}: {value!r}.") from exc


def _read_hot3d_frames(
    sequence: Hot3dSequence,
    stride: int,
    max_frames: Optional[int],
    gaze_bounds_policy: str,
    require_visible_gaze: bool,
) -> Tuple[List[Hot3dFrame], Dict[str, int]]:
    _ensure_hot3d_runtime()
    frames: List[Hot3dFrame] = []
    counts = {
        "csv_rows": 0,
        "kept_rows": 0,
        "missing_gaze_rows": 0,
        "out_of_bounds_rows": 0,
        "not_visible_rows": 0,
        "stride_dropped_rows": 0,
    }
    with sequence.gaze_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["csv_rows"] += 1
            if stride > 1 and ((counts["csv_rows"] - 1) % stride) != 0:
                counts["stride_dropped_rows"] += 1
                continue
            if max_frames is not None and len(frames) >= int(max_frames):
                break
            if not _as_bool(row.get("gaze_available", False)):
                counts["missing_gaze_rows"] += 1
                continue
            if require_visible_gaze and not _as_bool(row.get("in_raw_bounds", False)):
                counts["not_visible_rows"] += 1
                continue
            x_value = row.get("upright_x_norm")
            y_value = row.get("upright_y_norm")
            if x_value in (None, "") or y_value in (None, ""):
                counts["missing_gaze_rows"] += 1
                continue
            gaze = np.asarray(
                [
                    _as_float(x_value, "upright_x_norm"),
                    _as_float(y_value, "upright_y_norm"),
                ],
                dtype=np.float32,
            )
            checked = check_gaze_bounds(
                gaze,
                policy=gaze_bounds_policy,
                row_idx=counts["csv_rows"] - 1,
            )
            if checked is None:
                counts["out_of_bounds_rows"] += 1
                continue
            frames.append(
                Hot3dFrame(
                    sequence_id=sequence.sequence_id,
                    frame_index=_as_int(row.get("frame_index"), "frame_index"),
                    timestamp_ns=_as_int(row.get("timecode_timestamp_ns"), "timecode_timestamp_ns"),
                    gaze_xy=(float(checked[0]), float(checked[1])),
                )
            )
    counts["kept_rows"] = len(frames)
    return frames, counts


def _scan_sequences(
    sequences: Sequence[Hot3dSequence],
    stride: int,
    max_frames_per_sequence: Optional[int],
    gaze_bounds_policy: str,
    require_visible_gaze: bool,
) -> Tuple[Dict[str, List[Hot3dFrame]], Dict[str, Dict[str, int]], List[int]]:
    frame_map: Dict[str, List[Hot3dFrame]] = {}
    count_map: Dict[str, Dict[str, int]] = {}
    episode_lengths = []
    for sequence in sequences:
        frames, counts = _read_hot3d_frames(
            sequence=sequence,
            stride=stride,
            max_frames=max_frames_per_sequence,
            gaze_bounds_policy=gaze_bounds_policy,
            require_visible_gaze=require_visible_gaze,
        )
        if not frames:
            continue
        frame_map[sequence.sequence_id] = frames
        count_map[sequence.sequence_id] = counts
        episode_lengths.append(len(frames))
    if not episode_lengths:
        raise ValueError("No HOT3D frames with usable gaze labels were found.")
    return frame_map, count_map, episode_lengths


def _resize_rgb(frame_bgr, image_size: Tuple[int, int]):
    _ensure_hot3d_runtime()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[:2] != image_size:
        interp = cv2.INTER_AREA if frame_rgb.shape[0] >= image_size[0] else cv2.INTER_LINEAR
        frame_rgb = cv2.resize(
            frame_rgb,
            (image_size[1], image_size[0]),
            interpolation=interp,
        )
    return np.ascontiguousarray(frame_rgb[:, :, :3], dtype=np.uint8)


def _read_video_frame(cap, video_path: pathlib.Path, frame_index: int, current_pos: int):
    _ensure_hot3d_runtime()
    if frame_index != current_pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index} from '{video_path}'.")
    return frame_bgr


def _make_output(output_path: pathlib.Path, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output '{output_path}' already exists. Pass --overwrite.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_heatmap_method(method: str) -> str:
    key = str(method).strip().lower()
    normalized = HEATMAP_METHOD_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported heatmap_method={method!r}. Supported methods: "
            f"{', '.join(HEATMAP_METHODS)}."
        )
    return normalized


def _validate_heatmap_token_grid(value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"heatmap_token_grid must be [height, width], got {value}.")
    grid_h = int(value[0])
    grid_w = int(value[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"heatmap_token_grid dimensions must be positive, got {value}.")
    return grid_h, grid_w


def _validate_positive_float(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float, got {value}.")
    return value


def _resolve_point_heatmap_sigma(
    image_size: Tuple[int, int],
    heatmap_token_grid: Sequence[int],
    point_heatmap_sigma_tokens: Optional[float],
    point_heatmap_sigma_px: Optional[float],
) -> Dict[str, object]:
    """Resolve the canonical token-space sigma into renderer pixel units."""
    _ensure_hot3d_runtime()
    token_grid = _validate_heatmap_token_grid(heatmap_token_grid)
    image_h, image_w = int(image_size[0]), int(image_size[1])
    patch_h = image_h / float(token_grid[0])
    patch_w = image_w / float(token_grid[1])
    if point_heatmap_sigma_px is not None:
        sigma_px = _validate_positive_float(
            "point_heatmap_sigma_px",
            point_heatmap_sigma_px,
        )
        sigma_tokens_y = sigma_px / patch_h
        sigma_tokens_x = sigma_px / patch_w
        sigma_tokens = float((sigma_tokens_y + sigma_tokens_x) / 2.0)
        source = "px_override"
    else:
        if point_heatmap_sigma_tokens is None:
            point_heatmap_sigma_tokens = 2.0
        sigma_tokens = _validate_positive_float(
            "point_heatmap_sigma_tokens",
            point_heatmap_sigma_tokens,
        )
        sigma_px_y = sigma_tokens * patch_h
        sigma_px_x = sigma_tokens * patch_w
        if not np.isclose(sigma_px_y, sigma_px_x, rtol=1e-6, atol=1e-6):
            raise ValueError(
                "point_heatmap_sigma_tokens currently requires square patch geometry "
                "because the Gaussian renderer uses one isotropic pixel sigma. Got "
                f"sigma_y={sigma_px_y}, sigma_x={sigma_px_x}."
            )
        sigma_px = float((sigma_px_y + sigma_px_x) / 2.0)
        source = "tokens"
    return {
        "sigma_px": float(sigma_px),
        "sigma_tokens": float(sigma_tokens),
        "sigma_norm_yx": [float(sigma_px / image_h), float(sigma_px / image_w)],
        "heatmap_token_grid": [int(token_grid[0]), int(token_grid[1])],
        "source": source,
    }


def _sigma_tokens_to_px_values(
    image_size: Tuple[int, int],
    heatmap_token_grid: Sequence[int],
    sigma_tokens_values: Sequence[float],
) -> Tuple[List[float], List[str]]:
    sigma_px_values = []
    labels = []
    for sigma_tokens in sigma_tokens_values:
        resolved = _resolve_point_heatmap_sigma(
            image_size=image_size,
            heatmap_token_grid=heatmap_token_grid,
            point_heatmap_sigma_tokens=sigma_tokens,
            point_heatmap_sigma_px=None,
        )
        sigma_px_values.append(float(resolved["sigma_px"]))
        labels.append(f"sigma={float(sigma_tokens):g}tok")
    return sigma_px_values, labels


def _generate_heatmaps_from_xy(
    gaze_xy,
    image_size: Tuple[int, int],
    method: str,
    point_heatmap_sigma_px: float,
    point_heatmap_window: int,
):
    """Generate dense heatmap labels from normalized HOT3D gaze points."""
    _ensure_hot3d_runtime()
    method = _normalize_heatmap_method(method)
    if method == "gaussian_point":
        return gaussian_heatmaps_from_points(
            gaze_xy,
            image_size=image_size,
            sigma_px=point_heatmap_sigma_px,
            window_size=point_heatmap_window,
            episode_ends=np.asarray([len(gaze_xy)], dtype=np.int64),
        ).astype(np.float32)
    raise AssertionError(f"Unhandled heatmap method {method!r}.")


def _encode_dense_heatmaps_to_tokens(dense_heatmaps, heatmap_token_grid: Sequence[int]):
    _ensure_hot3d_runtime()
    dense_heatmaps = np.asarray(dense_heatmaps, dtype=np.float32)
    if dense_heatmaps.ndim != 3:
        raise ValueError(f"dense_heatmaps must be [N,H,W], got {dense_heatmaps.shape}.")
    token_grid = _validate_heatmap_token_grid(heatmap_token_grid)
    n_frames, image_h, image_w = dense_heatmaps.shape
    if image_h % token_grid[0] != 0 or image_w % token_grid[1] != 0:
        raise ValueError(
            "Token heatmap storage requires image_size divisible by heatmap_token_grid, "
            f"got image {(image_h, image_w)} and grid {token_grid}."
        )
    patch_h = image_h // token_grid[0]
    patch_w = image_w // token_grid[1]
    tokens = dense_heatmaps.reshape(
        n_frames,
        token_grid[0],
        patch_h,
        token_grid[1],
        patch_w,
    ).mean(axis=(2, 4))
    tokens = tokens.reshape(n_frames, token_grid[0] * token_grid[1], 1)
    max_values = np.maximum(tokens.max(axis=1, keepdims=True), 1e-12)
    return (tokens / max_values).astype(np.float32)


def _decode_heatmap_tokens_to_image(tokens, image_size: Tuple[int, int], heatmap_token_grid):
    _ensure_hot3d_runtime()
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim == 1:
        tokens = tokens[:, None]
    if tokens.ndim != 2 or tokens.shape[-1] != 1:
        raise ValueError(f"Expected token heatmap shape [T,1] or [T], got {tokens.shape}.")
    codec = HeatmapTokenCodec(
        token_grid=heatmap_token_grid,
        image_size=image_size,
    )
    with torch.no_grad():
        decoded = codec.decode_tokens(
            torch.from_numpy(tokens).to(dtype=torch.float32),
            method="gaussian_splat",
        )
    return decoded.detach().cpu().numpy().astype(np.float32)


def _heatmap_value_to_preview_image(value, image_size: Tuple[int, int], heatmap_token_grid):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1 or (
        value.ndim == 2
        and value.shape == (int(heatmap_token_grid[0]) * int(heatmap_token_grid[1]), 1)
    ):
        return _decode_heatmap_tokens_to_image(
            value,
            image_size=image_size,
            heatmap_token_grid=heatmap_token_grid,
        )
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[-1] == 1:
        return value[..., 0]
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    raise ValueError(f"Unsupported heatmap preview value shape {value.shape}.")


def _blend_heatmap_red(rgb, heatmap, alpha: float):
    _ensure_hot3d_runtime()
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(heatmap.min(initial=0.0))
    denom = float(heatmap.max(initial=0.0))
    if denom > 0.0:
        heatmap = heatmap / denom
    heat = np.clip(heatmap, 0.0, 1.0)[..., None]
    rgb_float = np.asarray(rgb, dtype=np.float32)
    red = np.zeros_like(rgb_float)
    red[..., 0] = 255.0
    blended = rgb_float * (1.0 - alpha * heat) + red * (alpha * heat)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _draw_gaze_marker(rgb, gaze_xy):
    _ensure_hot3d_runtime()
    image = np.ascontiguousarray(rgb.copy())
    image_h, image_w = image.shape[:2]
    x = int(round(float(gaze_xy[0]) * (image_w - 1)))
    y = int(round(float(gaze_xy[1]) * (image_h - 1)))
    cv2.circle(image, (x, y), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), 6, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return image


def _heatmap_to_gray_rgb(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(heatmap.min(initial=0.0))
    denom = float(heatmap.max(initial=0.0))
    if denom > 0.0:
        heatmap = heatmap / denom
    gray = np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _add_top_label_bar(frame, labels: Sequence[str]):
    _ensure_hot3d_runtime()
    label_count = max(1, len(labels))
    frame_h, frame_w = frame.shape[:2]
    panel_w = frame_w // label_count
    bar = np.full((34, frame_w, 3), 18, dtype=np.uint8)
    for idx, label in enumerate(labels):
        x = idx * panel_w + 12
        cv2.putText(
            bar,
            str(label),
            (x, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return np.vstack([bar, frame])


def _make_overlay_heatmap_side_by_side(rgb, heatmap, gaze_xy, alpha: float, left_label: str):
    overlay = _blend_heatmap_red(rgb, heatmap, alpha=alpha)
    overlay = _draw_gaze_marker(overlay, gaze_xy)
    heatmap_rgb = _heatmap_to_gray_rgb(heatmap)
    frame = np.concatenate([overlay, heatmap_rgb], axis=1)
    return _add_top_label_bar(frame, [left_label, "bw heatmap"])


def _write_overlay_contact_sheet(frames: Sequence, output_path: pathlib.Path) -> None:
    _ensure_hot3d_runtime()
    if not frames:
        return
    frame_h, frame_w = frames[0].shape[:2]
    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    canvas = np.zeros((rows * frame_h, cols * frame_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        canvas[
            row * frame_h : (row + 1) * frame_h,
            col * frame_w : (col + 1) * frame_w,
        ] = frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _transcode_mp4_to_h264(src_path: pathlib.Path, output_path: pathlib.Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if output_path.exists():
        output_path.unlink()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-movflags",
        "+faststart",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError):
        if output_path.exists():
            output_path.unlink()
        return False
    return True


def _write_overlay_video(frames: Sequence, output_path: pathlib.Path, fps: float) -> str:
    _ensure_hot3d_runtime()
    if not frames:
        return "none"
    frame_h, frame_w = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.opencv_tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open preview video writer for '{tmp_path}'.")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if _transcode_mp4_to_h264(tmp_path, output_path):
        tmp_path.unlink(missing_ok=True)
        return "h264_yuv420p"
    if output_path.exists():
        output_path.unlink()
    shutil.move(str(tmp_path), str(output_path))
    return "opencv_mp4v"


def write_heatmap_overlay_preview(
    dataset_path: str,
    output_dir: str,
    camera_key: str = "camera0_rgb",
    gaze_key: str = "gaze_xy",
    heatmap_key: str = "gaze_heatmap",
    heatmap_token_grid: Sequence[int] = DEFAULT_HEATMAP_TOKEN_GRID,
    max_frames: int = 80,
    alpha: float = 0.45,
    fps: float = 8.0,
) -> Dict[str, object]:
    """Write red-overlay and black/white heatmap side-by-side video previews."""
    _ensure_hot3d_runtime()
    max_frames = int(max_frames)
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}.")

    root = zarr.open(str(dataset_path), mode="r")
    data = root["data"]
    attrs = root["meta"].attrs if "meta" in root else root.attrs
    total_frames = int(data[camera_key].shape[0])
    image_size = (int(data[camera_key].shape[1]), int(data[camera_key].shape[2]))
    num_frames = min(max_frames, total_frames)
    if gaze_key not in data:
        raise KeyError(f"Heatmap preview requires gaze key '{gaze_key}' in the zarr.")
    gaze_xy = np.asarray(data[gaze_key][:num_frames], dtype=np.float32)
    if heatmap_key in data:
        generated_heatmaps = None
        heatmap_source = "stored_zarr"
    else:
        sigma_px = float(attrs.get("point_heatmap_sigma_px", 32.0))
        window_size = int(attrs.get("point_heatmap_window", 1))
        generated_heatmaps = gaussian_heatmaps_from_points(
            gaze_xy,
            image_size=image_size,
            sigma_px=sigma_px,
            window_size=window_size,
            episode_ends=_preview_episode_ends(root, num_frames=num_frames),
        )
        heatmap_source = "generated_from_xy"
    frames = []
    for idx in range(num_frames):
        rgb = np.asarray(data[camera_key][idx], dtype=np.uint8)
        if generated_heatmaps is None:
            heatmap = _heatmap_value_to_preview_image(
                data[heatmap_key][idx],
                image_size=image_size,
                heatmap_token_grid=heatmap_token_grid,
            )
        else:
            heatmap = generated_heatmaps[idx]
        frames.append(
            _make_overlay_heatmap_side_by_side(
                rgb=rgb,
                heatmap=heatmap,
                gaze_xy=gaze_xy[idx],
                alpha=alpha,
                left_label="red overlay",
            )
        )

    output_dir_path = pathlib.Path(output_dir)
    contact_sheet_path = output_dir_path / "overlay_heatmap_side_by_side_contact_sheet.png"
    video_path = output_dir_path / "overlay_heatmap_side_by_side.mp4"
    _write_overlay_contact_sheet(frames, contact_sheet_path)
    video_codec = _write_overlay_video(frames, video_path, fps=fps)
    return {
        "output_dir": str(output_dir_path),
        "contact_sheet": str(contact_sheet_path),
        "video": str(video_path),
        "video_kind": "red_overlay_left_bw_heatmap_right",
        "video_codec": video_codec,
        "num_frames": int(num_frames),
        "heatmap_source": heatmap_source,
        "alpha": float(alpha),
        "fps": float(fps),
    }


def _preview_episode_ends(root, num_frames: int):
    if "meta" not in root or "episode_ends" not in root["meta"]:
        return np.asarray([num_frames], dtype=np.int64)
    source_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    preview_ends = [int(value) for value in source_ends.tolist() if 0 < int(value) < num_frames]
    if not preview_ends or preview_ends[-1] != num_frames:
        preview_ends.append(int(num_frames))
    return np.asarray(preview_ends, dtype=np.int64)


def write_sigma_comparison_preview(
    dataset_path: str,
    output_dir: str,
    sigma_px_values: Sequence[float],
    sigma_labels: Optional[Sequence[str]] = None,
    point_heatmap_window: int = 3,
    camera_key: str = "camera0_rgb",
    gaze_key: str = "gaze_xy",
    max_frames: int = 80,
    alpha: float = 0.45,
    fps: float = 8.0,
) -> Dict[str, object]:
    """Write a multi-sigma video comparison using RGB overlays and B/W heatmaps."""
    _ensure_hot3d_runtime()
    sigma_px_values = [float(value) for value in sigma_px_values]
    if not sigma_px_values:
        raise ValueError("sigma_px_values must contain at least one sigma.")
    for value in sigma_px_values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"sigma values must be positive finite floats, got {value}.")
    if sigma_labels is None:
        sigma_labels = [f"sigma={value:g}px" for value in sigma_px_values]
    else:
        sigma_labels = [str(value) for value in sigma_labels]
        if len(sigma_labels) != len(sigma_px_values):
            raise ValueError(
                "sigma_labels must have the same length as sigma_px_values, got "
                f"{len(sigma_labels)} vs {len(sigma_px_values)}."
            )
    max_frames = int(max_frames)
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}.")

    root = zarr.open(str(dataset_path), mode="r")
    data = root["data"]
    if gaze_key not in data:
        raise KeyError(f"Sigma comparison requires gaze key '{gaze_key}' in the zarr.")
    total_frames = int(data[camera_key].shape[0])
    num_frames = min(max_frames, total_frames)
    if num_frames <= 0:
        raise ValueError("Cannot write sigma comparison for an empty dataset.")
    image_h, image_w = data[camera_key].shape[1:3]
    gaze_xy = np.asarray(data[gaze_key][:num_frames], dtype=np.float32)
    episode_ends = _preview_episode_ends(root, num_frames=num_frames)
    heatmaps_by_sigma = [
        gaussian_heatmaps_from_points(
            gaze_xy,
            image_size=(int(image_h), int(image_w)),
            sigma_px=sigma,
            window_size=point_heatmap_window,
            episode_ends=episode_ends,
        )
        for sigma in sigma_px_values
    ]

    frames = []
    for frame_idx in range(num_frames):
        rgb = np.asarray(data[camera_key][frame_idx], dtype=np.uint8)
        rows = []
        for label, heatmaps in zip(sigma_labels, heatmaps_by_sigma):
            rows.append(
                _make_overlay_heatmap_side_by_side(
                    rgb=rgb,
                    heatmap=heatmaps[frame_idx],
                    gaze_xy=gaze_xy[frame_idx],
                    alpha=alpha,
                    left_label=f"{label} overlay",
                )
            )
        frames.append(np.vstack(rows))

    output_dir_path = pathlib.Path(output_dir)
    video_path = output_dir_path / "sigma_compare.mp4"
    contact_sheet_path = output_dir_path / "sigma_compare_contact_sheet.png"
    _write_overlay_contact_sheet(frames, contact_sheet_path)
    video_codec = _write_overlay_video(frames, video_path, fps=fps)
    return {
        "output_dir": str(output_dir_path),
        "contact_sheet": str(contact_sheet_path),
        "video": str(video_path),
        "video_kind": "sigma_comparison_red_overlay_left_bw_heatmap_right",
        "video_codec": video_codec,
        "sigma_px_values": [float(value) for value in sigma_px_values],
        "sigma_labels": [str(value) for value in sigma_labels],
        "point_heatmap_window": int(point_heatmap_window),
        "num_frames": int(num_frames),
        "alpha": float(alpha),
        "fps": float(fps),
    }


def write_token_decode_comparison_image(
    dataset_path: str,
    output_dir: str,
    frame_index: int = 40,
    camera_key: str = "camera0_rgb",
    gaze_key: str = "gaze_xy",
    heatmap_key: str = "gaze_heatmap",
    heatmap_token_grid: Sequence[int] = DEFAULT_HEATMAP_TOKEN_GRID,
) -> Dict[str, object]:
    """Compare native dense heatmap against the zarr token heatmap decoded to image space."""
    _ensure_hot3d_runtime()
    root = zarr.open(str(dataset_path), mode="r")
    data = root["data"]
    total_frames = int(data[camera_key].shape[0])
    if total_frames <= 0:
        raise ValueError("Cannot write token decode comparison for an empty dataset.")
    frame_index = max(0, min(int(frame_index), total_frames - 1))
    image_size = (int(data[camera_key].shape[1]), int(data[camera_key].shape[2]))
    attrs = root["meta"].attrs if "meta" in root else root.attrs
    sigma_px = float(attrs.get("point_heatmap_sigma_px", 32.0))
    window_size = int(attrs.get("point_heatmap_window", 1))
    episode_ends = (
        np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        if "meta" in root and "episode_ends" in root["meta"]
        else np.asarray([total_frames], dtype=np.int64)
    )
    episode_start = 0
    for episode_end in episode_ends.tolist():
        if frame_index < int(episode_end):
            break
        episode_start = int(episode_end)
    gaze_segment = np.asarray(data[gaze_key][episode_start : frame_index + 1], dtype=np.float32)
    native_sequence = gaussian_heatmaps_from_points(
        gaze_segment,
        image_size=image_size,
        sigma_px=sigma_px,
        window_size=window_size,
        episode_ends=np.asarray([len(gaze_segment)], dtype=np.int64),
    )
    native_heatmap = native_sequence[-1]

    stored_value = np.asarray(data[heatmap_key][frame_index], dtype=np.float32)
    if stored_value.ndim == 2 and stored_value.shape[-1] == 1:
        token_heatmap = stored_value
    else:
        token_heatmap = _encode_dense_heatmaps_to_tokens(
            stored_value.reshape((1,) + stored_value.shape[-2:]),
            heatmap_token_grid=heatmap_token_grid,
        )[0]
    decoded_heatmap = _decode_heatmap_tokens_to_image(
        token_heatmap,
        image_size=image_size,
        heatmap_token_grid=heatmap_token_grid,
    )
    diff_heatmap = np.abs(native_heatmap.astype(np.float32) - decoded_heatmap.astype(np.float32))

    panels = [
        _heatmap_to_gray_rgb(native_heatmap),
        _heatmap_to_gray_rgb(decoded_heatmap),
        _heatmap_to_gray_rgb(diff_heatmap),
    ]
    frame = np.concatenate(panels, axis=1)
    frame = _add_top_label_bar(frame, ["native dense", "token decoded", "abs diff"])
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"token_decode_vs_native_frame_{frame_index:06d}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    mse = float(np.mean((native_heatmap.astype(np.float32) - decoded_heatmap.astype(np.float32)) ** 2))
    return {
        "output": str(output_path),
        "frame_index": int(frame_index),
        "mse": mse,
        "native_shape": [int(v) for v in native_heatmap.shape],
        "decoded_shape": [int(v) for v in decoded_heatmap.shape],
        "token_shape": [int(v) for v in token_heatmap.shape],
    }


def convert_hot3d_processed_to_open_zarr(
    processed_root: str,
    output_zarr: str,
    image_size: Sequence[int] = (256, 256),
    stride: int = 1,
    max_frames_per_sequence: Optional[int] = None,
    sequence: Optional[Sequence[str]] = None,
    sequence_file: Optional[str] = None,
    limit_sequences: Optional[int] = None,
    overwrite: bool = False,
    camera_key: str = "camera0_rgb",
    gaze_key: str = "gaze_xy",
    heatmap_key: str = "gaze_heatmap",
    timestamp_key: str = "timestamp_ns",
    gaze_bounds_policy: str = "error",
    require_visible_gaze: bool = False,
    heatmap_method: str = "gaussian_point",
    heatmap_storage: str = "none",
    heatmap_token_grid: Sequence[int] = DEFAULT_HEATMAP_TOKEN_GRID,
    point_heatmap_sigma_tokens: Optional[float] = 2.0,
    point_heatmap_sigma_px: Optional[float] = None,
    point_heatmap_window: int = 1,
    image_chunk_frames: int = 16,
    image_write_batch_size: int = 512,
    preview_overlay_dir: Optional[str] = None,
    preview_overlay_max_frames: int = 80,
    preview_overlay_alpha: float = 0.45,
    preview_overlay_fps: float = 8.0,
    preview_sigma_compare_tokens: Optional[Sequence[float]] = None,
    preview_sigma_compare_px: Optional[Sequence[float]] = None,
    preview_token_compare_frame: int = 40,
    validate: bool = False,
) -> Dict[str, object]:
    """Convert compact HOT3D Aria preprocessing outputs into a Gaze-WAM open zarr."""
    _ensure_hot3d_runtime()
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if max_frames_per_sequence is not None and int(max_frames_per_sequence) <= 0:
        raise ValueError("max_frames_per_sequence must be positive when provided.")
    if int(image_chunk_frames) <= 0:
        raise ValueError("image_chunk_frames must be positive.")
    if int(image_write_batch_size) <= 0:
        raise ValueError("image_write_batch_size must be positive.")
    if gaze_bounds_policy not in ("error", "drop", "clip"):
        raise ValueError("gaze_bounds_policy must be one of: error, drop, clip.")
    image_size = (int(image_size[0]), int(image_size[1]))
    heatmap_method = _normalize_heatmap_method(heatmap_method)
    heatmap_storage = str(heatmap_storage).strip().lower()
    if heatmap_storage not in HEATMAP_STORAGE_MODES:
        raise ValueError(f"heatmap_storage must be one of {HEATMAP_STORAGE_MODES}, got {heatmap_storage!r}.")
    sigma_config = _resolve_point_heatmap_sigma(
        image_size=image_size,
        heatmap_token_grid=heatmap_token_grid,
        point_heatmap_sigma_tokens=point_heatmap_sigma_tokens,
        point_heatmap_sigma_px=point_heatmap_sigma_px,
    )
    point_heatmap_sigma_px = float(sigma_config["sigma_px"])
    heatmap_token_grid = tuple(int(value) for value in sigma_config["heatmap_token_grid"])
    if preview_sigma_compare_px and preview_overlay_dir is None:
        raise ValueError("--preview-sigma-compare-px requires --preview-overlay-dir.")
    if preview_sigma_compare_tokens and preview_overlay_dir is None:
        raise ValueError("--preview-sigma-compare-tokens requires --preview-overlay-dir.")
    if preview_sigma_compare_tokens and preview_sigma_compare_px:
        raise ValueError(
            "Use only one of --preview-sigma-compare-tokens or --preview-sigma-compare-px."
        )
    processed_root_path = pathlib.Path(processed_root)
    output_path = pathlib.Path(output_zarr)

    sequences = _discover_sequences(
        processed_root=processed_root_path,
        sequence=sequence,
        sequence_file=sequence_file,
        limit_sequences=limit_sequences,
    )
    frame_map, count_map, episode_lengths = _scan_sequences(
        sequences=sequences,
        stride=int(stride),
        max_frames_per_sequence=max_frames_per_sequence,
        gaze_bounds_policy=gaze_bounds_policy,
        require_visible_gaze=require_visible_gaze,
    )
    active_sequences = [sequence for sequence in sequences if sequence.sequence_id in frame_map]
    total_frames = int(sum(episode_lengths))
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))
    _make_output(output_path, overwrite=overwrite)

    store = zarr.DirectoryStore(str(output_path))
    root_group = zarr.group(store=store, overwrite=True)
    data = root_group.create_group("data")
    meta = root_group.create_group("meta")
    image_chunks = (min(int(image_chunk_frames), total_frames), image_size[0], image_size[1], 3)
    vector_chunks = (min(4096, total_frames),)
    num_heatmap_tokens = int(heatmap_token_grid[0]) * int(heatmap_token_grid[1])
    if heatmap_storage == "none":
        heatmap_shape = None
        heatmap_chunks = None
    elif heatmap_storage == "token":
        heatmap_shape = (total_frames, num_heatmap_tokens, 1)
        heatmap_chunks = (min(4096, total_frames), num_heatmap_tokens, 1)
    else:
        heatmap_shape = (total_frames, image_size[0], image_size[1])
        heatmap_chunks = (min(int(image_chunk_frames), total_frames), image_size[0], image_size[1])

    image_array = data.zeros(
        camera_key,
        shape=(total_frames, image_size[0], image_size[1], 3),
        chunks=image_chunks,
        dtype="uint8",
    )
    gaze_array = data.zeros(
        gaze_key,
        shape=(total_frames, 2),
        chunks=vector_chunks + (2,),
        dtype="float32",
    )
    heatmap_array = None
    if heatmap_storage != "none":
        heatmap_array = data.zeros(
            heatmap_key,
            shape=heatmap_shape,
            chunks=heatmap_chunks,
            dtype="float32",
        )
    timestamp_array = data.zeros(
        timestamp_key,
        shape=(total_frames,),
        chunks=vector_chunks,
        dtype="int64",
    )
    sequence_index_array = data.zeros(
        "source_sequence_index",
        shape=(total_frames,),
        chunks=vector_chunks,
        dtype="int32",
    )

    offset = 0
    written_sequences = []
    for sequence_idx, sequence in enumerate(active_sequences):
        frames = frame_map[sequence.sequence_id]
        cap = cv2.VideoCapture(str(sequence.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open HOT3D processed video '{sequence.video_path}'.")
        seq_start = offset
        seq_gaze = np.zeros((len(frames), 2), dtype=np.float32)
        seq_timestamps = np.zeros((len(frames),), dtype=np.int64)
        seq_indices = np.full((len(frames),), int(sequence_idx), dtype=np.int32)
        write_batch_size = min(int(image_write_batch_size), len(frames))
        image_buffer = np.zeros(
            (write_batch_size, image_size[0], image_size[1], 3),
            dtype=np.uint8,
        )
        image_buffer_count = 0

        def flush_image_buffer() -> None:
            nonlocal image_buffer_count, offset
            if image_buffer_count <= 0:
                return
            write_start = offset
            write_end = offset + image_buffer_count
            image_array[write_start:write_end] = image_buffer[:image_buffer_count]
            offset = write_end
            image_buffer_count = 0

        try:
            current_pos = -1
            for local_idx, frame in enumerate(frames):
                frame_bgr = _read_video_frame(
                    cap=cap,
                    video_path=sequence.video_path,
                    frame_index=frame.frame_index,
                    current_pos=current_pos,
                )
                current_pos = frame.frame_index + 1
                image_buffer[image_buffer_count] = _resize_rgb(frame_bgr, image_size=image_size)
                image_buffer_count += 1
                gaze = np.asarray(frame.gaze_xy, dtype=np.float32)
                seq_gaze[local_idx] = gaze
                seq_timestamps[local_idx] = int(frame.timestamp_ns)
                if image_buffer_count >= write_batch_size:
                    flush_image_buffer()
            flush_image_buffer()
        finally:
            cap.release()
        seq_end = offset
        gaze_array[seq_start:seq_end] = seq_gaze
        timestamp_array[seq_start:seq_end] = seq_timestamps
        sequence_index_array[seq_start:seq_end] = seq_indices
        if heatmap_storage != "none":
            dense_heatmaps = _generate_heatmaps_from_xy(
                seq_gaze,
                image_size=image_size,
                method=heatmap_method,
                point_heatmap_sigma_px=point_heatmap_sigma_px,
                point_heatmap_window=point_heatmap_window,
            )
            if heatmap_storage == "token":
                heatmap_array[seq_start:seq_end] = _encode_dense_heatmaps_to_tokens(
                    dense_heatmaps,
                    heatmap_token_grid=heatmap_token_grid,
                )
            else:
                heatmap_array[seq_start:seq_end] = dense_heatmaps
        written_sequences.append(sequence.sequence_id)

    has_gaze_label = np.ones((total_frames,), dtype=np.bool_)
    has_heatmap_image = np.full((total_frames,), heatmap_storage != "none", dtype=np.bool_)
    data.array("has_gaze_label", has_gaze_label, shape=has_gaze_label.shape, dtype=has_gaze_label.dtype)
    data.array(
        "has_heatmap_image",
        has_heatmap_image,
        shape=has_heatmap_image.shape,
        dtype=has_heatmap_image.dtype,
    )
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)
    meta.attrs["dataset_type"] = "open"
    meta.attrs["source_dataset"] = "HOT3D Aria"
    meta.attrs["source_processed_root"] = str(processed_root_path)
    meta.attrs["source_stage1_script"] = "scripts/preprocess_hot3d_aria.py"
    meta.attrs["image_size"] = list(image_size)
    meta.attrs["image_resize_mode"] = "stretch"
    meta.attrs["gaze_is_normalized"] = True
    meta.attrs["gaze_coordinate_source"] = "upright_x_norm/upright_y_norm"
    meta.attrs["label_mode"] = "point"
    meta.attrs["heatmap_storage"] = heatmap_storage
    meta.attrs["heatmap_source"] = (
        "absent:xy_only_dsnt_supervision"
        if heatmap_storage == "none"
        else f"generated_from_gaze_xy:{heatmap_method}"
    )
    meta.attrs["heatmap_method"] = heatmap_method
    meta.attrs["point_heatmap_sigma_px"] = float(point_heatmap_sigma_px)
    meta.attrs["point_heatmap_sigma_tokens"] = float(sigma_config["sigma_tokens"])
    meta.attrs["point_heatmap_sigma_norm_yx"] = list(sigma_config["sigma_norm_yx"])
    meta.attrs["point_heatmap_sigma_source"] = str(sigma_config["source"])
    meta.attrs["heatmap_token_grid_for_sigma"] = list(heatmap_token_grid)
    meta.attrs["point_heatmap_window"] = int(point_heatmap_window)
    meta.attrs["timestamp_key"] = timestamp_key
    meta.attrs["timestamp_unit"] = "ns"
    meta.attrs["presence_mask_keys"] = ["has_gaze_label", "has_heatmap_image"]
    meta.attrs["sequence_ids"] = written_sequences
    meta.attrs["stride"] = int(stride)
    meta.attrs["image_chunk_frames"] = int(image_chunk_frames)
    meta.attrs["image_write_batch_size"] = int(image_write_batch_size)
    meta.attrs["max_frames_per_sequence"] = (
        None if max_frames_per_sequence is None else int(max_frames_per_sequence)
    )

    summary: Dict[str, object] = {
        "output_zarr": str(output_path),
        "processed_root": str(processed_root_path),
        "dataset_type": "open",
        "num_sequences": int(len(written_sequences)),
        "num_frames": int(total_frames),
        "episode_ends": [int(value) for value in episode_ends.tolist()],
        "image_size": list(image_size),
        "image_resize_mode": "stretch",
        "gaze_key": gaze_key,
        "heatmap_key": heatmap_key,
        "camera_key": camera_key,
        "timestamp_key": timestamp_key,
        "heatmap_method": heatmap_method,
        "heatmap_storage": heatmap_storage,
        "heatmap_shape": None if heatmap_shape is None else [int(v) for v in heatmap_shape],
        "point_heatmap_sigma_px": float(point_heatmap_sigma_px),
        "point_heatmap_sigma_tokens": float(sigma_config["sigma_tokens"]),
        "point_heatmap_sigma_norm_yx": list(sigma_config["sigma_norm_yx"]),
        "point_heatmap_sigma_source": str(sigma_config["source"]),
        "heatmap_token_grid_for_sigma": list(heatmap_token_grid),
        "point_heatmap_window": int(point_heatmap_window),
        "image_chunk_frames": int(image_chunk_frames),
        "image_write_batch_size": int(image_write_batch_size),
        "stride": int(stride),
        "max_frames_per_sequence": None
        if max_frames_per_sequence is None
        else int(max_frames_per_sequence),
        "sequence_counts": count_map,
    }
    if validate:
        from diffusion_policy.scripts.validate_gaze_wam_zarr import validate_gaze_wam_zarr

        validation_heatmap_dim = (
            1
            if heatmap_storage == "token"
            else int(image_size[0] // heatmap_token_grid[0])
            * int(image_size[1] // heatmap_token_grid[1])
        )
        summary["validation"] = validate_gaze_wam_zarr(
            dataset_path=str(output_path),
            dataset_type="open",
            camera_key=camera_key,
            gaze_key=gaze_key,
            heatmap_key=None if heatmap_storage == "none" else heatmap_key,
            image_size=image_size,
            image_resize_mode="stretch",
            heatmap_token_grid=heatmap_token_grid,
            heatmap_dim=validation_heatmap_dim,
            check_dataset_sample=True,
        )
    if preview_overlay_dir is not None:
        summary["preview_overlay"] = write_heatmap_overlay_preview(
            dataset_path=str(output_path),
            output_dir=preview_overlay_dir,
            camera_key=camera_key,
            gaze_key=gaze_key,
            heatmap_key=heatmap_key,
            heatmap_token_grid=heatmap_token_grid,
            max_frames=preview_overlay_max_frames,
            alpha=preview_overlay_alpha,
            fps=preview_overlay_fps,
        )
        if heatmap_storage != "none":
            summary["preview_token_decode_comparison"] = write_token_decode_comparison_image(
                dataset_path=str(output_path),
                output_dir=preview_overlay_dir,
                frame_index=preview_token_compare_frame,
                camera_key=camera_key,
                gaze_key=gaze_key,
                heatmap_key=heatmap_key,
                heatmap_token_grid=heatmap_token_grid,
            )
        sigma_compare_px_values = None
        sigma_compare_labels = None
        if preview_sigma_compare_tokens:
            sigma_compare_px_values, sigma_compare_labels = _sigma_tokens_to_px_values(
                image_size=image_size,
                heatmap_token_grid=heatmap_token_grid,
                sigma_tokens_values=preview_sigma_compare_tokens,
            )
        elif preview_sigma_compare_px:
            sigma_compare_px_values = [float(value) for value in preview_sigma_compare_px]
            sigma_compare_labels = [f"sigma={float(value):g}px" for value in sigma_compare_px_values]
        if sigma_compare_px_values:
            summary["preview_sigma_compare"] = write_sigma_comparison_preview(
                dataset_path=str(output_path),
                output_dir=preview_overlay_dir,
                sigma_px_values=sigma_compare_px_values,
                sigma_labels=sigma_compare_labels,
                point_heatmap_window=point_heatmap_window,
                camera_key=camera_key,
                gaze_key=gaze_key,
                max_frames=preview_overlay_max_frames,
                alpha=preview_overlay_alpha,
                fps=preview_overlay_fps,
            )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert W:/HOT3D_processed-style HOT3D Aria packages into a Gaze-WAM open zarr."
    )
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--sequence", nargs="*", default=None)
    parser.add_argument("--sequence-file", default=None)
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--heatmap-key", default="gaze_heatmap")
    parser.add_argument("--timestamp-key", default="timestamp_ns")
    parser.add_argument(
        "--heatmap-token-grid",
        type=int,
        nargs=2,
        default=DEFAULT_HEATMAP_TOKEN_GRID,
        metavar=("GRID_H", "GRID_W"),
        help="Token grid used to derive pixel sigma from --point-heatmap-sigma-tokens.",
    )
    parser.add_argument(
        "--heatmap-method",
        choices=HEATMAP_METHODS,
        default="gaussian_point",
        help="Method used by stage-2 conversion to generate dense heatmaps from gaze_xy.",
    )
    parser.add_argument(
        "--heatmap-storage",
        choices=HEATMAP_STORAGE_MODES,
        default="none",
        help="Store heatmap labels as compact tokens, dense image heatmaps, or none.",
    )
    parser.add_argument(
        "--point-heatmap-sigma-tokens",
        type=float,
        default=2.0,
        help="Canonical Gaussian spread in heatmap-token/patch units.",
    )
    parser.add_argument(
        "--point-heatmap-sigma-px",
        type=float,
        default=None,
        help="Legacy exact pixel-sigma override. When set, it takes precedence over tokens.",
    )
    parser.add_argument("--point-heatmap-window", type=int, default=1)
    parser.add_argument("--image-chunk-frames", type=int, default=16)
    parser.add_argument("--image-write-batch-size", type=int, default=512)
    parser.add_argument("--preview-overlay-dir", default=None)
    parser.add_argument("--preview-overlay-max-frames", type=int, default=80)
    parser.add_argument("--preview-overlay-alpha", type=float, default=0.45)
    parser.add_argument("--preview-overlay-fps", type=float, default=8.0)
    parser.add_argument("--preview-token-compare-frame", type=int, default=40)
    parser.add_argument(
        "--preview-sigma-compare-px",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional sigma values for a dynamic comparison mp4. Requires "
            "--preview-overlay-dir."
        ),
    )
    parser.add_argument(
        "--preview-sigma-compare-tokens",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional token-space sigma values for a dynamic comparison mp4. Requires "
            "--preview-overlay-dir."
        ),
    )
    parser.add_argument(
        "--gaze-bounds-policy",
        choices=("error", "drop", "clip"),
        default="error",
    )
    parser.add_argument("--require-visible-gaze", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = convert_hot3d_processed_to_open_zarr(
        processed_root=args.processed_root,
        output_zarr=args.output_zarr,
        image_size=args.image_size,
        stride=args.stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
        sequence=args.sequence,
        sequence_file=args.sequence_file,
        limit_sequences=args.limit_sequences,
        overwrite=args.overwrite,
        camera_key=args.camera_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        timestamp_key=args.timestamp_key,
        gaze_bounds_policy=args.gaze_bounds_policy,
        require_visible_gaze=args.require_visible_gaze,
        heatmap_method=args.heatmap_method,
        heatmap_storage=args.heatmap_storage,
        heatmap_token_grid=args.heatmap_token_grid,
        point_heatmap_sigma_tokens=args.point_heatmap_sigma_tokens,
        point_heatmap_sigma_px=args.point_heatmap_sigma_px,
        point_heatmap_window=args.point_heatmap_window,
        image_chunk_frames=args.image_chunk_frames,
        image_write_batch_size=args.image_write_batch_size,
        preview_overlay_dir=args.preview_overlay_dir,
        preview_overlay_max_frames=args.preview_overlay_max_frames,
        preview_overlay_alpha=args.preview_overlay_alpha,
        preview_overlay_fps=args.preview_overlay_fps,
        preview_sigma_compare_tokens=args.preview_sigma_compare_tokens,
        preview_sigma_compare_px=args.preview_sigma_compare_px,
        preview_token_compare_frame=args.preview_token_compare_frame,
        validate=args.validate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
