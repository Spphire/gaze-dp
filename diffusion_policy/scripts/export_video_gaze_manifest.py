from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple


def _ensure_video_export_runtime(needs_zarr_converter: bool = False):
    global check_gaze_bounds
    global convert_open_gaze_manifest
    global cv2
    global np
    try:
        cv2
    except NameError:
        import cv2 as _cv2
        import numpy as _np
        from diffusion_policy.common.gaze_utils import check_gaze_bounds as _check_gaze_bounds

        check_gaze_bounds = _check_gaze_bounds
        cv2 = _cv2
        np = _np
    if needs_zarr_converter:
        try:
            convert_open_gaze_manifest
        except NameError:
            from diffusion_policy.scripts.convert_open_gaze_manifest import (
                convert_open_gaze_manifest as _convert_open_gaze_manifest,
            )

            convert_open_gaze_manifest = _convert_open_gaze_manifest


VIDEO_KEY_ALIASES = ("video_path", "video", "video_file", "clip_path", "mp4_path")
EPISODE_KEY_ALIASES = ("episode_id", "episode", "video_id", "sequence_id", "clip_id")
FRAME_KEY_ALIASES = ("frame_idx", "frame_index", "frame", "frame_number")
TIMESTAMP_KEY_ALIASES = ("timestamp", "time", "time_sec", "t")
GAZE_X_ALIASES = ("gaze_x", "x", "gx", "u")
GAZE_Y_ALIASES = ("gaze_y", "y", "gy", "v")
WIDTH_ALIASES = ("image_width", "width", "w")
HEIGHT_ALIASES = ("image_height", "height", "h")
SUPPORTED_IMAGE_RESIZE_MODES = ("stretch",)


def _first_present(row: Dict[str, object], aliases: Sequence[str]) -> Optional[object]:
    for key in aliases:
        value = row.get(key)
        if value is not None and str(value) != "":
            return value
    return None


def _read_rows(path: pathlib.Path) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("frames", data.get("items", data.get("rows")))
        if not isinstance(data, list):
            raise ValueError("JSON metadata must be a list or contain frames/items/rows list.")
        rows = list(data)
    else:
        raise ValueError(f"Unsupported metadata suffix '{path.suffix}'. Use .csv, .json, or .jsonl.")
    if not rows:
        raise ValueError(f"Metadata '{path}' has no rows.")
    return rows


def _resolve_path(value: object, root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path


def _as_float(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {name}: {value!r}") from exc


def _validate_image_resize_mode(image_resize_mode: str) -> str:
    if image_resize_mode not in SUPPORTED_IMAGE_RESIZE_MODES:
        raise ValueError(
            "Video gaze export currently supports only direct stretch resize. "
            f"Got image_resize_mode={image_resize_mode!r}; crop/letterbox modes must remap gaze "
            "coordinates before frame export."
        )
    return image_resize_mode


def _as_int(value: object, name: str) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {name}: {value!r}") from exc


def _video_id(row: Dict[str, object], video_path: pathlib.Path) -> str:
    episode = _first_present(row, EPISODE_KEY_ALIASES)
    if episode is not None:
        return str(episode)
    return video_path.stem


def _safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    stem = "".join(keep).strip("_")
    return stem or "episode"


def _imwrite_unicode(path: pathlib.Path, image: np.ndarray) -> None:
    _ensure_video_export_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for '{path}'.")
    encoded.tofile(str(path))


def _frame_index_from_row(row: Dict[str, object], fps: float) -> int:
    frame_value = _first_present(row, FRAME_KEY_ALIASES)
    if frame_value is not None:
        return _as_int(frame_value, "frame_idx")
    time_value = _first_present(row, TIMESTAMP_KEY_ALIASES)
    if time_value is None:
        raise KeyError(
            f"Row must contain one of frame keys {FRAME_KEY_ALIASES} or timestamp keys "
            f"{TIMESTAMP_KEY_ALIASES}."
        )
    return int(round(_as_float(time_value, "timestamp") * fps))


def _timestamp_from_row(row: Dict[str, object], frame_idx: int, fps: float) -> float:
    time_value = _first_present(row, TIMESTAMP_KEY_ALIASES)
    if time_value is not None:
        return _as_float(time_value, "timestamp")
    return float(frame_idx) / float(fps)


def _read_frame(video_path: pathlib.Path, frame_idx: int) -> Tuple[np.ndarray, float, int]:
    _ensure_video_export_runtime()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video '{video_path}'.")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        if frame_idx < 0:
            raise ValueError(f"frame_idx must be non-negative, got {frame_idx}.")
        if frame_count > 0 and frame_idx >= frame_count:
            raise IndexError(
                f"frame_idx={frame_idx} is outside video '{video_path}' with {frame_count} frames."
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from '{video_path}'.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame_rgb), fps, frame_count
    finally:
        cap.release()


def _normalize_gaze(
    row: Dict[str, object],
    source_hw: Tuple[int, int],
    gaze_is_normalized: bool,
    gaze_bounds_policy: str = "error",
    row_idx: Optional[int] = None,
) -> np.ndarray:
    _ensure_video_export_runtime()
    x_value = _first_present(row, GAZE_X_ALIASES)
    y_value = _first_present(row, GAZE_Y_ALIASES)
    if x_value is None or y_value is None:
        raise KeyError(f"Row has no gaze point. Expected one of {GAZE_X_ALIASES}/{GAZE_Y_ALIASES}.")
    gaze = np.asarray([_as_float(x_value, "gaze_x"), _as_float(y_value, "gaze_y")], dtype=np.float32)
    if not gaze_is_normalized:
        width_value = _first_present(row, WIDTH_ALIASES)
        height_value = _first_present(row, HEIGHT_ALIASES)
        if width_value is None or height_value is None:
            image_h, image_w = source_hw
        else:
            image_w = _as_float(width_value, "image_width")
            image_h = _as_float(height_value, "image_height")
        gaze = gaze / np.asarray([image_w, image_h], dtype=np.float32)
    checked = check_gaze_bounds(gaze, policy=gaze_bounds_policy, row_idx=row_idx)
    if checked is None:
        raise ValueError("Video gaze export does not support dropping rows before frame extraction.")
    return checked


def _write_manifest(rows: Sequence[Dict[str, object]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "image_path",
        "gaze_x",
        "gaze_y",
        "image_width",
        "image_height",
        "timestamp",
        "source_video",
        "source_frame_idx",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_video_gaze_manifest(
    metadata_path: str,
    output_manifest: str,
    frames_dir: str,
    root_dir: Optional[str] = None,
    image_size: Optional[Sequence[int]] = None,
    gaze_is_normalized: bool = True,
    overwrite: bool = False,
    output_zarr: Optional[str] = None,
    zarr_image_size: Sequence[int] = (256, 256),
    gaze_bounds_policy: str = "error",
    image_resize_mode: str = "stretch",
) -> Dict[str, object]:
    """Extract video frames and write a generic point-label open-gaze manifest."""
    _ensure_video_export_runtime(needs_zarr_converter=output_zarr is not None)
    if gaze_bounds_policy not in ("error", "clip"):
        raise ValueError("Video gaze export supports gaze_bounds_policy='error' or 'clip'.")
    image_resize_mode = _validate_image_resize_mode(str(image_resize_mode))
    metadata = pathlib.Path(metadata_path)
    root = pathlib.Path(root_dir) if root_dir is not None else metadata.parent
    output_manifest_path = pathlib.Path(output_manifest)
    frames_root = pathlib.Path(frames_dir)
    if output_manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output manifest '{output_manifest_path}' exists. Pass overwrite=True to replace."
        )
    frames_root.mkdir(parents=True, exist_ok=True)
    if image_size is not None:
        image_size = (int(image_size[0]), int(image_size[1]))
    zarr_image_size = (int(zarr_image_size[0]), int(zarr_image_size[1]))

    rows = _read_rows(metadata)
    manifest_rows: List[Dict[str, object]] = []
    video_cache: Dict[str, Tuple[float, int]] = {}

    for idx, row in enumerate(rows):
        video_value = _first_present(row, VIDEO_KEY_ALIASES)
        if video_value is None:
            raise KeyError(f"Row {idx} has no video path. Expected one of {VIDEO_KEY_ALIASES}.")
        video_path = _resolve_path(video_value, root)
        video_key = str(video_path)
        if video_key in video_cache:
            fps, _ = video_cache[video_key]
        else:
            cap = cv2.VideoCapture(video_key)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video '{video_path}'.")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps <= 0:
                fps = 30.0
            video_cache[video_key] = (fps, frame_count)

        frame_idx = _frame_index_from_row(row, fps=fps)
        timestamp = _timestamp_from_row(row, frame_idx=frame_idx, fps=fps)
        frame_rgb, fps, frame_count = _read_frame(video_path, frame_idx)
        source_h, source_w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
        gaze_xy = _normalize_gaze(
            row,
            source_hw=(source_h, source_w),
            gaze_is_normalized=gaze_is_normalized,
            gaze_bounds_policy=gaze_bounds_policy,
            row_idx=idx,
        )
        if image_size is not None and (source_h, source_w) != image_size:
            frame_rgb = cv2.resize(
                frame_rgb,
                (image_size[1], image_size[0]),
                interpolation=cv2.INTER_AREA if source_h >= image_size[0] else cv2.INTER_LINEAR,
            )
        episode_id = _video_id(row, video_path)
        episode_dir = frames_root / _safe_stem(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"frame_{frame_idx:08d}_{idx:08d}.png"
        image_path = episode_dir / image_name
        _imwrite_unicode(image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        try:
            image_path_for_manifest = str(image_path.relative_to(output_manifest_path.parent))
        except ValueError:
            image_path_for_manifest = str(image_path)
        manifest_rows.append(
            {
                "episode_id": episode_id,
                "image_path": image_path_for_manifest,
                "gaze_x": float(gaze_xy[0]),
                "gaze_y": float(gaze_xy[1]),
                "image_width": int(frame_rgb.shape[1]),
                "image_height": int(frame_rgb.shape[0]),
                "timestamp": float(timestamp),
                "source_video": str(video_path),
                "source_frame_idx": int(frame_idx),
            }
        )

    _write_manifest(manifest_rows, output_manifest_path)
    summary: Dict[str, object] = {
        "metadata_path": str(metadata),
        "output_manifest": str(output_manifest_path),
        "frames_dir": str(frames_root),
        "num_frames": int(len(manifest_rows)),
        "num_videos": int(len(video_cache)),
        "gaze_is_normalized": True,
        "gaze_bounds_policy": gaze_bounds_policy,
        "image_resize_mode": image_resize_mode,
    }
    if output_zarr is not None:
        zarr_summary = convert_open_gaze_manifest(
            manifest_path=str(output_manifest_path),
            output_path=output_zarr,
            image_size=zarr_image_size,
            gaze_is_normalized=True,
            label_mode="point",
            root_dir=str(output_manifest_path.parent),
            overwrite=overwrite,
            gaze_bounds_policy=gaze_bounds_policy,
            image_resize_mode=image_resize_mode,
        )
        summary["output_zarr"] = output_zarr
        summary["zarr"] = zarr_summary
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Extract video frames and export a generic Gaze-WAM open-gaze manifest."
    )
    parser.add_argument("--metadata", required=True, help="CSV, JSON, or JSONL video gaze metadata.")
    parser.add_argument("--output-manifest", required=True, help="Output generic CSV manifest.")
    parser.add_argument("--frames-dir", required=True, help="Directory for extracted frame PNGs.")
    parser.add_argument("--root-dir", default=None, help="Root for relative video paths.")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument(
        "--image-resize-mode",
        choices=SUPPORTED_IMAGE_RESIZE_MODES,
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gaze-bounds-policy",
        choices=("error", "clip"),
        default="error",
        help="How to handle point labels outside [0,1] after normalization.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-zarr", default=None, help="Optional direct open zarr output.")
    parser.add_argument("--zarr-image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = export_video_gaze_manifest(
        metadata_path=args.metadata,
        output_manifest=args.output_manifest,
        frames_dir=args.frames_dir,
        root_dir=args.root_dir,
        image_size=args.image_size,
        gaze_is_normalized=args.gaze_is_normalized,
        overwrite=args.overwrite,
        output_zarr=args.output_zarr,
        zarr_image_size=args.zarr_image_size,
        gaze_bounds_policy=args.gaze_bounds_policy,
        image_resize_mode=args.image_resize_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
