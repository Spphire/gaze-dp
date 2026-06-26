from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence


def _ensure_metadata_export_runtime():
    global export_video_gaze_manifest
    try:
        return export_video_gaze_manifest
    except NameError:
        from diffusion_policy.scripts.export_video_gaze_manifest import (
            export_video_gaze_manifest as _export_video_gaze_manifest,
        )

        export_video_gaze_manifest = _export_video_gaze_manifest
        return export_video_gaze_manifest


def _read_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
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
            data = data.get("frames", data.get("items", data.get("rows", data.get("annotations"))))
        if not isinstance(data, list):
            raise ValueError("JSON metadata must be a list or contain frames/items/rows/annotations list.")
        rows = list(data)
    else:
        raise ValueError(f"Unsupported metadata suffix '{path.suffix}'. Use .csv, .json, or .jsonl.")
    if not rows:
        raise ValueError(f"Metadata '{path}' has no rows.")
    return rows


def _write_rows(rows: Sequence[Dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "video_path",
        "frame_idx",
        "timestamp",
        "gaze_x",
        "gaze_y",
        "image_width",
        "image_height",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _get_nested(row: Dict[str, Any], key: Optional[str]) -> Any:
    if key is None or key == "":
        return None
    current: Any = row
    for part in key.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _as_required(row: Dict[str, Any], key: Optional[str], output_name: str, row_idx: int) -> Any:
    value = _get_nested(row, key)
    if value is None or str(value) == "":
        raise KeyError(f"Row {row_idx} is missing required field '{output_name}' from key '{key}'.")
    return value


def _as_optional(row: Dict[str, Any], key: Optional[str]) -> Any:
    value = _get_nested(row, key)
    if value is None or str(value) == "":
        return ""
    return value


def _passes_filters(row: Dict[str, Any], filters: Sequence[str]) -> bool:
    for expr in filters:
        if "=" not in expr:
            raise ValueError(f"Filter must be KEY=VALUE, got {expr!r}.")
        key, expected = expr.split("=", 1)
        value = _get_nested(row, key)
        if str(value) != expected:
            return False
    return True


def _load_key_map(path_or_json: Optional[str]) -> Dict[str, str]:
    if path_or_json is None:
        return {}
    maybe_path = pathlib.Path(path_or_json)
    try:
        is_path = maybe_path.exists()
    except OSError:
        is_path = False
    if is_path:
        data = json.loads(maybe_path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path_or_json)
    if not isinstance(data, dict):
        raise ValueError("key_map must be a JSON object.")
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _resolve_key(explicit: Optional[str], key_map: Dict[str, str], canonical_name: str) -> Optional[str]:
    return explicit or key_map.get(canonical_name)


def _normalize_episode(value: Any, default: str) -> str:
    if value is None or str(value) == "":
        return default
    return str(value)


def adapt_open_video_gaze_metadata(
    metadata_path: str,
    output_metadata: str,
    video_key: Optional[str] = None,
    gaze_x_key: Optional[str] = None,
    gaze_y_key: Optional[str] = None,
    episode_key: Optional[str] = None,
    frame_key: Optional[str] = None,
    timestamp_key: Optional[str] = None,
    width_key: Optional[str] = None,
    height_key: Optional[str] = None,
    key_map: Optional[Dict[str, str]] = None,
    filters: Sequence[str] = (),
    limit: Optional[int] = None,
    drop_missing: bool = False,
    output_manifest: Optional[str] = None,
    frames_dir: Optional[str] = None,
    root_dir: Optional[str] = None,
    image_size: Optional[Sequence[int]] = None,
    gaze_is_normalized: bool = True,
    output_zarr: Optional[str] = None,
    zarr_image_size: Sequence[int] = (256, 256),
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Map dataset-specific video-gaze metadata into the generic Gaze-WAM video schema.

    Keys may be dotted paths such as ``clip.path`` or ``gaze.0`` for nested JSON metadata.
    The canonical output can be passed directly to ``export_video_gaze_manifest.py``.
    """

    if frame_key is None and timestamp_key is None:
        frame_key = (key_map or {}).get("frame_idx") if key_map is not None else None
        timestamp_key = (key_map or {}).get("timestamp") if key_map is not None else None
    if frame_key is None and timestamp_key is None:
        raise ValueError("Either frame_key or timestamp_key must be provided.")
    if output_manifest is not None and frames_dir is None:
        raise ValueError("frames_dir is required when output_manifest is provided.")

    key_map = key_map or {}
    video_key = _resolve_key(video_key, key_map, "video_path")
    gaze_x_key = _resolve_key(gaze_x_key, key_map, "gaze_x")
    gaze_y_key = _resolve_key(gaze_y_key, key_map, "gaze_y")
    if video_key is None or gaze_x_key is None or gaze_y_key is None:
        raise ValueError("video_path, gaze_x, and gaze_y key mappings are required.")
    episode_key = _resolve_key(episode_key, key_map, "episode_id")
    frame_key = _resolve_key(frame_key, key_map, "frame_idx")
    timestamp_key = _resolve_key(timestamp_key, key_map, "timestamp")
    width_key = _resolve_key(width_key, key_map, "image_width")
    height_key = _resolve_key(height_key, key_map, "image_height")

    source = pathlib.Path(metadata_path)
    output = pathlib.Path(output_metadata)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output metadata '{output}' exists. Pass overwrite=True to replace.")

    rows = _read_rows(source)
    canonical_rows: List[Dict[str, Any]] = []
    skipped = 0
    for row_idx, row in enumerate(rows):
        if not _passes_filters(row, filters):
            skipped += 1
            continue
        try:
            video_path = _as_required(row, video_key, "video_path", row_idx)
            gaze_x = _as_required(row, gaze_x_key, "gaze_x", row_idx)
            gaze_y = _as_required(row, gaze_y_key, "gaze_y", row_idx)
            frame_idx = _as_optional(row, frame_key)
            timestamp = _as_optional(row, timestamp_key)
            if frame_idx == "" and timestamp == "":
                raise KeyError(
                    f"Row {row_idx} must provide frame_idx via '{frame_key}' or timestamp via "
                    f"'{timestamp_key}'."
                )
            canonical_rows.append(
                {
                    "episode_id": _normalize_episode(
                        _as_optional(row, episode_key),
                        default=pathlib.Path(str(video_path)).stem,
                    ),
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "image_width": _as_optional(row, width_key),
                    "image_height": _as_optional(row, height_key),
                }
            )
        except (KeyError, ValueError):
            if not drop_missing:
                raise
            skipped += 1
            continue
        if limit is not None and len(canonical_rows) >= int(limit):
            break

    if not canonical_rows:
        raise ValueError("No metadata rows remained after filtering/mapping.")
    _write_rows(canonical_rows, output)

    summary: Dict[str, Any] = {
        "metadata_path": str(source),
        "output_metadata": str(output),
        "num_input_rows": int(len(rows)),
        "num_output_rows": int(len(canonical_rows)),
        "num_skipped_rows": int(skipped),
        "keys": {
            "video_path": video_key,
            "episode_id": episode_key,
            "frame_idx": frame_key,
            "timestamp": timestamp_key,
            "gaze_x": gaze_x_key,
            "gaze_y": gaze_y_key,
            "image_width": width_key,
            "image_height": height_key,
        },
    }
    if output_manifest is not None:
        _ensure_metadata_export_runtime()
        export_summary = export_video_gaze_manifest(
            metadata_path=str(output),
            output_manifest=output_manifest,
            frames_dir=frames_dir,
            root_dir=root_dir,
            image_size=image_size,
            gaze_is_normalized=gaze_is_normalized,
            overwrite=overwrite,
            output_zarr=output_zarr,
            zarr_image_size=zarr_image_size,
        )
        summary["output_manifest"] = output_manifest
        summary["export"] = export_summary
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Adapt dataset-specific video gaze metadata into the generic Gaze-WAM video metadata "
            "schema used by export_video_gaze_manifest.py."
        )
    )
    parser.add_argument("--metadata", required=True, help="Input CSV, JSON, or JSONL metadata.")
    parser.add_argument("--output-metadata", required=True, help="Output canonical CSV metadata.")
    parser.add_argument("--key-map", default=None, help="JSON string or path mapping canonical names to source keys.")
    parser.add_argument("--video-key", default=None, help="Source key for video path.")
    parser.add_argument("--episode-key", default=None, help="Optional source key for episode id.")
    parser.add_argument("--frame-key", default=None, help="Optional source key for frame index.")
    parser.add_argument("--timestamp-key", default=None, help="Optional source key for timestamp seconds.")
    parser.add_argument("--gaze-x-key", default=None, help="Source key for gaze x.")
    parser.add_argument("--gaze-y-key", default=None, help="Source key for gaze y.")
    parser.add_argument("--width-key", default=None, help="Optional source key for image width.")
    parser.add_argument("--height-key", default=None, help="Optional source key for image height.")
    parser.add_argument("--filter", action="append", default=[], help="Keep rows matching KEY=VALUE. Can repeat.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--output-manifest", default=None, help="Optional generic image manifest output.")
    parser.add_argument("--frames-dir", default=None, help="Directory for extracted frames when exporting manifest.")
    parser.add_argument("--root-dir", default=None, help="Root for relative video paths during frame export.")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-zarr", default=None, help="Optional direct open zarr output.")
    parser.add_argument("--zarr-image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    key_map = _load_key_map(args.key_map)
    video_key = _resolve_key(args.video_key, key_map, "video_path")
    gaze_x_key = _resolve_key(args.gaze_x_key, key_map, "gaze_x")
    gaze_y_key = _resolve_key(args.gaze_y_key, key_map, "gaze_y")
    if video_key is None or gaze_x_key is None or gaze_y_key is None:
        raise ValueError(
            "video, gaze_x, and gaze_y mappings are required. Provide --video-key/"
            "--gaze-x-key/--gaze-y-key or a --key-map JSON object."
        )
    summary = adapt_open_video_gaze_metadata(
        metadata_path=args.metadata,
        output_metadata=args.output_metadata,
        video_key=video_key,
        gaze_x_key=gaze_x_key,
        gaze_y_key=gaze_y_key,
        episode_key=args.episode_key,
        frame_key=args.frame_key,
        timestamp_key=args.timestamp_key,
        width_key=args.width_key,
        height_key=args.height_key,
        key_map=key_map,
        filters=args.filter,
        limit=args.limit,
        drop_missing=args.drop_missing,
        output_manifest=args.output_manifest,
        frames_dir=args.frames_dir,
        root_dir=args.root_dir,
        image_size=args.image_size,
        gaze_is_normalized=args.gaze_is_normalized,
        output_zarr=args.output_zarr,
        zarr_image_size=args.zarr_image_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
