from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _ensure_convert_runtime():
    global as_optional_gaze_wam_key
    global check_gaze_bounds
    global cv2
    global gaussian_heatmaps_from_points
    global np
    global zarr
    try:
        return zarr
    except NameError:
        import cv2 as _cv2
        import numpy as _np
        import zarr as _zarr
        from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key as _as_optional_gaze_wam_key
        from diffusion_policy.common.gaze_utils import check_gaze_bounds as _check_gaze_bounds
        from diffusion_policy.common.gaze_utils import (
            gaussian_heatmaps_from_points as _gaussian_heatmaps_from_points,
        )

        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        check_gaze_bounds = _check_gaze_bounds
        gaussian_heatmaps_from_points = _gaussian_heatmaps_from_points
        cv2 = _cv2
        np = _np
        zarr = _zarr
        return zarr

IMAGE_KEY_ALIASES = ("image_path", "image", "rgb_path", "frame_path", "path")
HEATMAP_KEY_ALIASES = ("heatmap_path", "gaze_heatmap_path", "mask_path")
EPISODE_KEY_ALIASES = ("episode_id", "episode", "video_id", "sequence_id", "clip_id")
FRAME_KEY_ALIASES = ("frame_idx", "frame_index", "timestamp", "time")
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


def _read_manifest(path: pathlib.Path) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    rows: List[Dict[str, object]] = []
    if suffix == ".jsonl":
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
            raise ValueError("JSON manifest must be a list or contain frames/items/rows list.")
        rows = list(data)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported manifest suffix '{path.suffix}'. Use .csv, .json, or .jsonl.")
    if not rows:
        raise ValueError(f"Manifest '{path}' has no rows.")
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
            "Open gaze conversion currently supports only direct stretch resize. "
            f"Got image_resize_mode={image_resize_mode!r}; crop/letterbox modes must remap gaze "
            "coordinates and dense heatmaps before conversion."
        )
    return image_resize_mode


def _read_timestamp(
    row: Dict[str, object],
    row_idx: int,
    timestamp_key: Optional[str] = None,
) -> Optional[float]:
    if timestamp_key is not None:
        value = row.get(timestamp_key)
        if value is None or str(value) == "":
            raise KeyError(f"Manifest row {row_idx} has no timestamp key '{timestamp_key}'.")
    else:
        value = _first_present(row, TIMESTAMP_KEY_ALIASES)
    if value is None:
        return None
    return _as_float(value, "timestamp")


def _imread_unicode(path: pathlib.Path, flags=None):
    _ensure_convert_runtime()
    if flags is None:
        flags = cv2.IMREAD_UNCHANGED
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _read_rgb(path: pathlib.Path, image_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    _ensure_convert_runtime()
    image = _imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image '{path}'.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for '{path}'.")
    source_hw = (int(image.shape[0]), int(image.shape[1]))
    if source_hw != image_size:
        interp = cv2.INTER_AREA if image.shape[0] >= image_size[0] else cv2.INTER_LINEAR
        image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=interp)
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8), source_hw


def _read_heatmap(path: pathlib.Path, image_size: Tuple[int, int]) -> np.ndarray:
    _ensure_convert_runtime()
    heatmap = _imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if heatmap is None:
        raise FileNotFoundError(f"Could not read heatmap '{path}'.")
    if heatmap.ndim == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    original_dtype = heatmap.dtype
    heatmap = heatmap.astype(np.float32)
    if heatmap.max(initial=0.0) > 1.5:
        if np.issubdtype(original_dtype, np.integer):
            heatmap = heatmap / float(np.iinfo(original_dtype).max)
        else:
            heatmap = heatmap / 255.0
    if heatmap.shape[:2] != image_size:
        heatmap = cv2.resize(heatmap, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(heatmap, dtype=np.float32)


def _normalize_gaze(
    row: Dict[str, object],
    source_hw: Tuple[int, int],
    gaze_is_normalized: bool,
    gaze_bounds_policy: str = "error",
    row_idx: Optional[int] = None,
) -> Optional[np.ndarray]:
    _ensure_convert_runtime()
    x_value = _first_present(row, GAZE_X_ALIASES)
    y_value = _first_present(row, GAZE_Y_ALIASES)
    if x_value is None or y_value is None:
        return None
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
    return check_gaze_bounds(gaze, policy=gaze_bounds_policy, row_idx=row_idx)


def _episode_ends(rows: Sequence[Dict[str, object]]) -> np.ndarray:
    _ensure_convert_runtime()
    ends = []
    current_episode = None
    seen_closed = set()
    for idx, row in enumerate(rows):
        episode = _first_present(row, EPISODE_KEY_ALIASES)
        if episode is None:
            episode = "default"
        episode = str(episode)
        if current_episode is None:
            current_episode = episode
        elif episode != current_episode:
            seen_closed.add(current_episode)
            if episode in seen_closed:
                raise ValueError(
                    "Manifest rows must be grouped by episode. "
                    f"Episode '{episode}' reappeared at row {idx}."
                )
            ends.append(idx)
            current_episode = episode
    ends.append(len(rows))
    return np.asarray(ends, dtype=np.int64)


def convert_open_gaze_manifest(
    manifest_path: str,
    output_path: str,
    image_size: Sequence[int] = (256, 256),
    gaze_is_normalized: bool = True,
    label_mode: str = "auto",
    root_dir: Optional[str] = None,
    overwrite: bool = False,
    camera_key: str = "camera0_rgb",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = "gaze_heatmap",
    timestamp_key: Optional[str] = None,
    output_timestamp_key: str = "timestamp",
    gaze_bounds_policy: str = "error",
    image_resize_mode: str = "stretch",
    point_heatmap_sigma_px: float = 20.0,
    point_heatmap_window: int = 1,
) -> Dict[str, object]:
    """Convert CSV/JSON/JSONL open gaze manifests into the Gaze-WAM open zarr schema."""
    _ensure_convert_runtime()
    if label_mode not in ("auto", "point", "heatmap"):
        raise ValueError("label_mode must be one of: auto, point, heatmap.")
    if gaze_bounds_policy not in ("error", "drop", "clip"):
        raise ValueError("gaze_bounds_policy must be one of: error, drop, clip.")
    image_resize_mode = _validate_image_resize_mode(str(image_resize_mode))
    gaze_key = as_optional_gaze_wam_key(gaze_key)
    heatmap_key = as_optional_gaze_wam_key(heatmap_key)
    if gaze_key is None:
        raise ValueError("Canonical open zarr conversion requires a non-null gaze_key.")
    if heatmap_key is None:
        raise ValueError("Canonical open zarr conversion requires a non-null heatmap_key.")
    manifest = pathlib.Path(manifest_path)
    root = pathlib.Path(root_dir) if root_dir is not None else manifest.parent
    output = pathlib.Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output '{output}' already exists. Pass overwrite=True to replace.")
    image_size = (int(image_size[0]), int(image_size[1]))

    rows = _read_manifest(manifest)
    images = []
    gaze_points = []
    heatmaps = []
    has_point = []
    has_heatmap = []
    timestamps = []
    has_timestamp = []

    for idx, row in enumerate(rows):
        image_value = _first_present(row, IMAGE_KEY_ALIASES)
        if image_value is None:
            raise KeyError(f"Manifest row {idx} has no image path. Expected one of {IMAGE_KEY_ALIASES}.")
        image, source_hw = _read_rgb(_resolve_path(image_value, root), image_size=image_size)
        images.append(image)

        gaze = _normalize_gaze(
            row,
            source_hw=source_hw,
            gaze_is_normalized=gaze_is_normalized,
            gaze_bounds_policy=gaze_bounds_policy,
            row_idx=idx,
        )
        heatmap_value = _first_present(row, HEATMAP_KEY_ALIASES)
        has_point.append(gaze is not None)
        has_heatmap.append(heatmap_value is not None)
        gaze_points.append(gaze if gaze is not None else np.zeros(2, dtype=np.float32))
        if heatmap_value is not None:
            heatmaps.append(_read_heatmap(_resolve_path(heatmap_value, root), image_size=image_size))
        else:
            heatmaps.append(None)
        timestamp = _read_timestamp(row, row_idx=idx, timestamp_key=timestamp_key)
        has_timestamp.append(timestamp is not None)
        if timestamp is not None:
            timestamps.append(timestamp)

    all_point = all(has_point)
    all_heatmap = all(has_heatmap)
    episode_ends = _episode_ends(rows)
    if label_mode == "auto":
        label_mode = "heatmap" if all_heatmap else "point"
    if label_mode == "point" and not all_point:
        raise ValueError("label_mode='point' requires gaze_x/gaze_y labels on every manifest row.")
    if label_mode == "heatmap" and (not all_point or not all_heatmap):
        raise ValueError(
            "label_mode='heatmap' requires gaze_x/gaze_y and heatmap_path labels on every "
            "manifest row."
        )
    if not all_point:
        raise ValueError("Canonical open zarr conversion requires gaze_x/gaze_y on every row.")
    if any(has_timestamp) and not all(has_timestamp):
        raise ValueError(
            "Timestamp preservation requires either every manifest row or no manifest rows to "
            "provide a timestamp."
        )

    store = zarr.DirectoryStore(str(output))
    zarr.group(store=store, overwrite=True)
    root_group = zarr.open(store=store, mode="w")
    data = root_group.create_group("data")
    meta = root_group.create_group("meta")

    image_array = np.stack(images, axis=0).astype(np.uint8)
    chunks = (min(64, len(image_array)),) + image_array.shape[1:]
    data.array(camera_key, image_array, shape=image_array.shape, chunks=chunks, dtype=image_array.dtype)

    gaze_array = np.stack(gaze_points, axis=0).astype(np.float32)
    data.array(gaze_key, gaze_array, shape=gaze_array.shape, dtype=gaze_array.dtype)

    if label_mode == "heatmap":
        heatmap_array = np.stack(heatmaps, axis=0).astype(np.float32)
        heatmap_source = "manifest"
    else:
        heatmap_array = gaussian_heatmaps_from_points(
            gaze_array,
            image_size=image_size,
            sigma_px=point_heatmap_sigma_px,
            window_size=point_heatmap_window,
            episode_ends=episode_ends,
        ).astype(np.float32)
        heatmap_source = "generated_from_gaze_xy"
    data.array(
        heatmap_key,
        heatmap_array,
        shape=heatmap_array.shape,
        chunks=(min(64, len(heatmap_array)),) + heatmap_array.shape[1:],
        dtype=heatmap_array.dtype,
    )
    has_gaze_label = np.ones((len(gaze_array),), dtype=np.bool_)
    has_heatmap_image = np.ones((len(heatmap_array),), dtype=np.bool_)
    data.array(
        "has_gaze_label",
        has_gaze_label,
        shape=has_gaze_label.shape,
        dtype=has_gaze_label.dtype,
    )
    data.array(
        "has_heatmap_image",
        has_heatmap_image,
        shape=has_heatmap_image.shape,
        dtype=has_heatmap_image.dtype,
    )
    wrote_timestamp = bool(all(has_timestamp))
    if wrote_timestamp:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        data.array(
            output_timestamp_key,
            timestamp_array,
            shape=timestamp_array.shape,
            dtype=timestamp_array.dtype,
        )
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)
    meta.attrs["source_manifest"] = str(manifest)
    meta.attrs["dataset_type"] = "open"
    meta.attrs["label_mode"] = label_mode
    meta.attrs["gaze_is_normalized"] = True
    meta.attrs["gaze_bounds_policy"] = gaze_bounds_policy
    meta.attrs["image_size"] = list(image_size)
    meta.attrs["image_resize_mode"] = image_resize_mode
    meta.attrs["heatmap_source"] = heatmap_source
    meta.attrs["point_heatmap_sigma_px"] = float(point_heatmap_sigma_px)
    meta.attrs["point_heatmap_window"] = int(point_heatmap_window)
    meta.attrs["timestamp_key"] = output_timestamp_key if wrote_timestamp else None
    meta.attrs["presence_mask_keys"] = ["has_gaze_label", "has_heatmap_image"]

    return {
        "output_path": str(output),
        "num_frames": int(len(rows)),
        "num_episodes": int(len(episode_ends)),
        "dataset_type": "open",
        "label_mode": label_mode,
        "gaze_bounds_policy": gaze_bounds_policy,
        "image_size": image_size,
        "image_resize_mode": image_resize_mode,
        "heatmap_source": heatmap_source,
        "point_heatmap_sigma_px": float(point_heatmap_sigma_px),
        "point_heatmap_window": int(point_heatmap_window),
        "has_timestamp": wrote_timestamp,
        "timestamp_key": output_timestamp_key if wrote_timestamp else None,
        "presence_mask_keys": ["has_gaze_label", "has_heatmap_image"],
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Convert open-source gaze manifests to Gaze-WAM zarr.")
    parser.add_argument("--manifest", required=True, help="CSV, JSON, or JSONL manifest path.")
    parser.add_argument("--output", required=True, help="Output zarr directory.")
    parser.add_argument("--root-dir", default=None, help="Root for relative image/heatmap paths.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument(
        "--image-resize-mode",
        choices=SUPPORTED_IMAGE_RESIZE_MODES,
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-mode", choices=("auto", "point", "heatmap"), default="auto")
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--heatmap-key", default="gaze_heatmap")
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--output-timestamp-key", default="timestamp")
    parser.add_argument("--point-heatmap-sigma-px", type=float, default=20.0)
    parser.add_argument(
        "--point-heatmap-window",
        type=int,
        default=1,
        help="Trailing in-episode frame window used when generating dense heatmaps from gaze_xy.",
    )
    parser.add_argument(
        "--gaze-bounds-policy",
        choices=("error", "drop", "clip"),
        default="error",
        help="How to handle point labels outside [0,1] after normalization.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = convert_open_gaze_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        image_size=args.image_size,
        gaze_is_normalized=args.gaze_is_normalized,
        label_mode=args.label_mode,
        root_dir=args.root_dir,
        overwrite=args.overwrite,
        camera_key=args.camera_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        timestamp_key=args.timestamp_key,
        output_timestamp_key=args.output_timestamp_key,
        gaze_bounds_policy=args.gaze_bounds_policy,
        image_resize_mode=args.image_resize_mode,
        point_heatmap_sigma_px=args.point_heatmap_sigma_px,
        point_heatmap_window=args.point_heatmap_window,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
