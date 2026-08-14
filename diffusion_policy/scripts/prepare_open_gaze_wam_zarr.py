from __future__ import annotations

import argparse
import json
import pathlib
import shlex
from typing import Dict, Optional, Sequence


def _load_key_map_arg(value: Optional[str]):
    if value is None:
        return {}
    candidate = pathlib.Path(value)
    try:
        is_path = candidate.exists()
    except OSError:
        is_path = False
    if is_path:
        return json.loads(candidate.read_text(encoding="utf-8"))
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("key_map must be a JSON object.")
    return {str(key): str(item) for key, item in data.items() if item is not None}


def _ensure_open_prepare_runtime():
    global adapt_open_video_gaze_metadata
    global as_optional_gaze_wam_key
    global convert_open_gaze_manifest
    global export_video_gaze_manifest
    global inspect_open_video_gaze_metadata
    global preview_gaze_wam_dataset
    global validate_gaze_wam_zarr
    try:
        return convert_open_gaze_manifest
    except NameError:
        from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key as _as_optional_gaze_wam_key
        from diffusion_policy.scripts.adapt_open_video_gaze_metadata import (
            adapt_open_video_gaze_metadata as _adapt_open_video_gaze_metadata,
        )
        from diffusion_policy.scripts.convert_open_gaze_manifest import (
            convert_open_gaze_manifest as _convert_open_gaze_manifest,
        )
        from diffusion_policy.scripts.export_video_gaze_manifest import (
            export_video_gaze_manifest as _export_video_gaze_manifest,
        )
        from diffusion_policy.scripts.inspect_open_video_gaze_metadata import (
            inspect_open_video_gaze_metadata as _inspect_open_video_gaze_metadata,
        )
        from diffusion_policy.scripts.preview_gaze_wam_dataset import (
            preview_gaze_wam_dataset as _preview_gaze_wam_dataset,
        )
        from diffusion_policy.scripts.validate_gaze_wam_zarr import (
            validate_gaze_wam_zarr as _validate_gaze_wam_zarr,
        )

        adapt_open_video_gaze_metadata = _adapt_open_video_gaze_metadata
        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        convert_open_gaze_manifest = _convert_open_gaze_manifest
        export_video_gaze_manifest = _export_video_gaze_manifest
        inspect_open_video_gaze_metadata = _inspect_open_video_gaze_metadata
        preview_gaze_wam_dataset = _preview_gaze_wam_dataset
        validate_gaze_wam_zarr = _validate_gaze_wam_zarr
        return convert_open_gaze_manifest


def _json_write(path: str, payload: Dict[str, object]) -> None:
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_optional_open_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ("", "none", "null"):
        return None
    return value


def _shell_join(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _append_optional(command, flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_optional_key(command, flag: str, value) -> None:
    command.extend([flag, str(value) if value is not None else "null"])


def _open_adapt_command(
    metadata_path: str,
    output_metadata: str,
    key_map: Optional[Dict[str, str]],
    video_key: Optional[str],
    gaze_x_key: Optional[str],
    gaze_y_key: Optional[str],
    episode_key: Optional[str],
    frame_key: Optional[str],
    timestamp_key: Optional[str],
    width_key: Optional[str],
    height_key: Optional[str],
    filters: Sequence[str],
    limit: Optional[int],
    drop_missing: bool,
    overwrite: bool,
) -> str:
    command = [
        "py",
        "scripts/adapt_open_video_gaze_metadata.py",
        "--metadata",
        metadata_path,
        "--output-metadata",
        output_metadata,
    ]
    if key_map:
        command.extend(
            [
                "--key-map",
                json.dumps(key_map, sort_keys=True, separators=(",", ":")),
            ]
        )
    for flag, value in (
        ("--video-key", video_key),
        ("--gaze-x-key", gaze_x_key),
        ("--gaze-y-key", gaze_y_key),
        ("--episode-key", episode_key),
        ("--frame-key", frame_key),
        ("--timestamp-key", timestamp_key),
        ("--width-key", width_key),
        ("--height-key", height_key),
    ):
        _append_optional(command, flag, value)
    for item in filters:
        command.extend(["--filter", str(item)])
    _append_optional(command, "--limit", limit)
    if drop_missing:
        command.append("--drop-missing")
    if overwrite:
        command.append("--overwrite")
    return _shell_join(command)


def _open_export_command(
    metadata_path: str,
    output_manifest: str,
    frames_dir: str,
    root_dir: Optional[str],
    image_size: Optional[Sequence[int]],
    image_resize_mode: str,
    gaze_is_normalized: bool,
    gaze_bounds_policy: str,
    overwrite: bool,
) -> str:
    command = [
        "py",
        "scripts/export_video_gaze_manifest.py",
        "--metadata",
        metadata_path,
        "--output-manifest",
        output_manifest,
        "--frames-dir",
        frames_dir,
    ]
    _append_optional(command, "--root-dir", root_dir)
    if image_size is not None:
        command.extend(["--image-size", *[str(value) for value in image_size]])
    command.extend(["--image-resize-mode", str(image_resize_mode)])
    if not gaze_is_normalized:
        command.append("--no-gaze-is-normalized")
    command.extend(["--gaze-bounds-policy", str(gaze_bounds_policy)])
    if overwrite:
        command.append("--overwrite")
    return _shell_join(command)


def _open_convert_command(
    manifest_path: str,
    output_zarr: str,
    image_size: Sequence[int],
    image_resize_mode: str,
    gaze_is_normalized: bool,
    label_mode: str,
    root_dir: Optional[str],
    gaze_key: Optional[str],
    heatmap_key: Optional[str],
    timestamp_key: Optional[str],
    gaze_bounds_policy: str,
    overwrite: bool,
) -> str:
    command = [
        "py",
        "scripts/convert_open_gaze_manifest.py",
        "--manifest",
        manifest_path,
        "--output",
        output_zarr,
        "--image-size",
        *[str(value) for value in image_size],
        "--image-resize-mode",
        str(image_resize_mode),
    ]
    if not gaze_is_normalized:
        command.append("--no-gaze-is-normalized")
    command.extend(["--label-mode", str(label_mode)])
    _append_optional(command, "--root-dir", root_dir)
    _append_optional_key(command, "--gaze-key", gaze_key)
    _append_optional_key(command, "--heatmap-key", heatmap_key)
    _append_optional(command, "--timestamp-key", timestamp_key)
    command.extend(["--gaze-bounds-policy", str(gaze_bounds_policy)])
    if overwrite:
        command.append("--overwrite")
    return _shell_join(command)


def _open_validation_command(
    output_zarr: str,
    n_obs_steps: int,
    action_horizon: int,
    n_latency_steps: int,
    image_size: Sequence[int],
    image_resize_mode: str,
    heatmap_token_grid: Sequence[int],
    heatmap_dim: int,
    require_timestamps: bool,
    timestamp_max_step: Optional[float],
    gaze_key: Optional[str],
    heatmap_key: Optional[str],
) -> str:
    command = [
        "py",
        "scripts/validate_gaze_wam_zarr.py",
        "--dataset-path",
        output_zarr,
        "--dataset-type",
        "open",
        "--n-obs-steps",
        str(int(n_obs_steps)),
        "--action-horizon",
        str(int(action_horizon)),
        "--n-latency-steps",
        str(int(n_latency_steps)),
        "--image-size",
        *[str(value) for value in image_size],
        "--image-resize-mode",
        str(image_resize_mode),
        "--heatmap-token-grid",
        *[str(value) for value in heatmap_token_grid],
        "--heatmap-dim",
        str(int(heatmap_dim)),
    ]
    _append_optional_key(command, "--gaze-key", gaze_key)
    _append_optional_key(command, "--heatmap-key", heatmap_key)
    if require_timestamps:
        command.append("--require-timestamps")
    _append_optional(command, "--timestamp-max-step", timestamp_max_step)
    return _shell_join(command)


def _open_preview_command(
    output_zarr: str,
    preview_output: str,
    preview_sample_index: int,
    n_obs_steps: int,
    action_horizon: int,
    n_latency_steps: int,
    image_size: Sequence[int],
    image_resize_mode: str,
    heatmap_token_grid: Sequence[int],
    gaze_key: Optional[str],
    heatmap_key: Optional[str],
) -> str:
    command = [
        "py",
        "scripts/preview_gaze_wam_dataset.py",
        "--dataset-path",
        output_zarr,
        "--dataset-type",
        "open",
        "--output-dir",
        preview_output,
        "--sample-index",
        str(int(preview_sample_index)),
        "--n-obs-steps",
        str(int(n_obs_steps)),
        "--action-horizon",
        str(int(action_horizon)),
        "--n-latency-steps",
        str(int(n_latency_steps)),
        "--image-size",
        *[str(value) for value in image_size],
        "--image-resize-mode",
        str(image_resize_mode),
        "--heatmap-token-grid",
        *[str(value) for value in heatmap_token_grid],
    ]
    _append_optional_key(command, "--gaze-key", gaze_key)
    _append_optional_key(command, "--heatmap-key", heatmap_key)
    return _shell_join(command)


def prepare_open_gaze_wam_zarr(
    output_zarr: str,
    report_json: Optional[str] = None,
    preview_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    video_metadata_path: Optional[str] = None,
    adapted_metadata_path: Optional[str] = None,
    metadata_inspect_json: Optional[str] = None,
    metadata_inspect_sample_rows: int = 200,
    output_manifest: Optional[str] = None,
    frames_dir: Optional[str] = None,
    root_dir: Optional[str] = None,
    key_map: Optional[Dict[str, str]] = None,
    video_key: Optional[str] = None,
    gaze_x_key: Optional[str] = None,
    gaze_y_key: Optional[str] = None,
    episode_key: Optional[str] = None,
    frame_key: Optional[str] = None,
    timestamp_key: Optional[str] = None,
    width_key: Optional[str] = None,
    height_key: Optional[str] = None,
    filters: Sequence[str] = (),
    limit: Optional[int] = None,
    drop_missing: bool = False,
    image_size: Sequence[int] = (256, 256),
    image_resize_mode: str = "stretch",
    gaze_is_normalized: bool = True,
    gaze_bounds_policy: str = "error",
    label_mode: str = "auto",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = "gaze_heatmap",
    overwrite: bool = False,
    n_obs_steps: int = 2,
    action_horizon: int = 48,
    n_latency_steps: int = 0,
    heatmap_token_grid: Sequence[int] = (16, 16),
    require_timestamps: bool = False,
    timestamp_max_step: Optional[float] = None,
    preview_sample_index: int = 0,
    skip_preview: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Prepare an open gaze dataset as a canonical Gaze-WAM open zarr."""
    metadata_inspect_fn = None
    if dry_run:
        gaze_key = _as_optional_open_key(gaze_key)
        heatmap_key = _as_optional_open_key(heatmap_key)
    else:
        _ensure_open_prepare_runtime()
        metadata_inspect_fn = inspect_open_video_gaze_metadata
        gaze_key = as_optional_gaze_wam_key(gaze_key)
        heatmap_key = as_optional_gaze_wam_key(heatmap_key)
    if (manifest_path is None) == (video_metadata_path is None):
        raise ValueError("Exactly one of manifest_path or video_metadata_path must be provided.")

    summary: Dict[str, object] = {
        "output_zarr": str(output_zarr),
        "mode": "manifest" if manifest_path is not None else "video_metadata",
        "dry_run": bool(dry_run),
    }
    manifest_for_conversion = manifest_path
    manifest_root = root_dir
    video_metadata_for_export = video_metadata_path
    needs_metadata_adapt = False
    planned_commands = {}

    if video_metadata_path is not None:
        if output_manifest is None:
            raise ValueError("output_manifest is required for video metadata preparation.")
        if frames_dir is None:
            raise ValueError("frames_dir is required for video metadata preparation.")
        if metadata_inspect_json is not None:
            if metadata_inspect_fn is None:
                from diffusion_policy.scripts.inspect_open_video_gaze_metadata import (
                    inspect_open_video_gaze_metadata as _inspect_open_video_gaze_metadata,
                )

                metadata_inspect_fn = _inspect_open_video_gaze_metadata
            inspect_output_metadata = adapted_metadata_path or str(
                pathlib.Path(output_manifest).with_suffix(".canonical.csv")
            )
            inspect_summary = metadata_inspect_fn(
                metadata_path=video_metadata_path,
                sample_rows=metadata_inspect_sample_rows,
                output_metadata_template=inspect_output_metadata,
                frames_dir_template=frames_dir,
                output_manifest_template=output_manifest,
                output_zarr_template=output_zarr,
                root_dir_template=root_dir,
            )
            _json_write(metadata_inspect_json, inspect_summary)
            inspect_summary["report_json"] = str(metadata_inspect_json)
            summary["metadata_inspect"] = inspect_summary
        needs_metadata_adapt = (
            adapted_metadata_path is not None
            or bool(key_map)
            or any(
                value is not None
                for value in (
                    video_key,
                    gaze_x_key,
                    gaze_y_key,
                    episode_key,
                    frame_key,
                    timestamp_key,
                    width_key,
                    height_key,
                )
            )
            or len(filters) > 0
            or limit is not None
            or drop_missing
        )
        if needs_metadata_adapt:
            if adapted_metadata_path is None:
                adapted_metadata_path = str(pathlib.Path(output_manifest).with_suffix(".canonical.csv"))
            if dry_run:
                video_metadata_for_export = adapted_metadata_path
                summary["adapt"] = None
            else:
                adapt_summary = adapt_open_video_gaze_metadata(
                    metadata_path=video_metadata_path,
                    output_metadata=adapted_metadata_path,
                    video_key=video_key,
                    gaze_x_key=gaze_x_key,
                    gaze_y_key=gaze_y_key,
                    episode_key=episode_key,
                    frame_key=frame_key,
                    timestamp_key=timestamp_key,
                    width_key=width_key,
                    height_key=height_key,
                    key_map=key_map,
                    filters=filters,
                    limit=limit,
                    drop_missing=drop_missing,
                    overwrite=overwrite,
                )
                summary["adapt"] = adapt_summary
                video_metadata_for_export = adapted_metadata_path
        else:
            video_metadata_for_export = video_metadata_path
        manifest_for_conversion = output_manifest
        manifest_root = str(pathlib.Path(output_manifest).parent)

    if video_metadata_path is not None and needs_metadata_adapt:
        planned_commands["adapt"] = _open_adapt_command(
            metadata_path=str(video_metadata_path),
            output_metadata=str(adapted_metadata_path),
            key_map=key_map,
            video_key=video_key,
            gaze_x_key=gaze_x_key,
            gaze_y_key=gaze_y_key,
            episode_key=episode_key,
            frame_key=frame_key,
            timestamp_key=timestamp_key,
            width_key=width_key,
            height_key=height_key,
            filters=filters,
            limit=limit,
            drop_missing=drop_missing,
            overwrite=overwrite,
        )
    if video_metadata_path is not None:
        planned_commands["export"] = _open_export_command(
            metadata_path=str(video_metadata_for_export),
            output_manifest=str(output_manifest),
            frames_dir=str(frames_dir),
            root_dir=root_dir,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            gaze_is_normalized=gaze_is_normalized,
            gaze_bounds_policy=gaze_bounds_policy,
            overwrite=overwrite,
        )
    planned_commands["convert"] = _open_convert_command(
        manifest_path=str(manifest_for_conversion),
        output_zarr=str(output_zarr),
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        gaze_is_normalized=True if video_metadata_path is not None else gaze_is_normalized,
        label_mode=label_mode,
        root_dir=manifest_root,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
        timestamp_key=timestamp_key if video_metadata_path is None else None,
        gaze_bounds_policy=gaze_bounds_policy,
        overwrite=overwrite,
    )
    planned_commands["validation"] = _open_validation_command(
        output_zarr=str(output_zarr),
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
        n_latency_steps=n_latency_steps,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        heatmap_token_grid=heatmap_token_grid,
        heatmap_dim=int(image_size[0] // heatmap_token_grid[0])
        * int(image_size[1] // heatmap_token_grid[1]),
        require_timestamps=require_timestamps,
        timestamp_max_step=timestamp_max_step,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
    )
    if not skip_preview:
        preview_output = preview_dir or str(pathlib.Path(output_zarr).with_suffix("")) + "_preview"
        planned_commands["preview"] = _open_preview_command(
            output_zarr=str(output_zarr),
            preview_output=preview_output,
            preview_sample_index=preview_sample_index,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            heatmap_token_grid=heatmap_token_grid,
            gaze_key=gaze_key,
            heatmap_key=heatmap_key,
        )
    summary["planned_commands"] = planned_commands
    if dry_run:
        summary["manifest_for_conversion"] = str(manifest_for_conversion)
        summary["manifest_root"] = str(manifest_root) if manifest_root is not None else None
        summary["convert"] = None
        summary["validation"] = None
        summary["preview"] = None
        summary["ok"] = True
        if report_json is not None:
            _json_write(report_json, summary)
            summary["report_json"] = str(report_json)
        return summary

    if video_metadata_path is not None:
        if needs_metadata_adapt:
            pass
        export_summary = export_video_gaze_manifest(
            metadata_path=video_metadata_for_export,
            output_manifest=output_manifest,
            frames_dir=frames_dir,
            root_dir=root_dir,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            gaze_is_normalized=gaze_is_normalized,
            overwrite=overwrite,
            gaze_bounds_policy=gaze_bounds_policy,
        )
        summary["export"] = export_summary

    convert_summary = convert_open_gaze_manifest(
        manifest_path=manifest_for_conversion,
        output_path=output_zarr,
        image_size=image_size,
        gaze_is_normalized=True if video_metadata_path is not None else gaze_is_normalized,
        label_mode=label_mode,
        root_dir=manifest_root,
        overwrite=overwrite,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
        timestamp_key=timestamp_key if video_metadata_path is None else None,
        gaze_bounds_policy=gaze_bounds_policy,
        image_resize_mode=image_resize_mode,
    )
    summary["convert"] = convert_summary

    validation = validate_gaze_wam_zarr(
        dataset_path=output_zarr,
        dataset_type="open",
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
        n_latency_steps=n_latency_steps,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        heatmap_token_grid=heatmap_token_grid,
        require_timestamps=require_timestamps,
        timestamp_max_step=timestamp_max_step,
        check_dataset_sample=True,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
    )
    summary["validation"] = validation

    preview = None
    if not skip_preview:
        preview_output = preview_dir or str(pathlib.Path(output_zarr).with_suffix("")) + "_preview"
        preview = preview_gaze_wam_dataset(
            dataset_path=output_zarr,
            dataset_type="open",
            output_dir=preview_output,
            sample_index=preview_sample_index,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            heatmap_token_grid=heatmap_token_grid,
            gaze_key=gaze_key,
            heatmap_key=heatmap_key,
        )
    summary["preview"] = preview
    summary["ok"] = bool(validation["valid"])

    if report_json is not None:
        _json_write(report_json, summary)
        summary["report_json"] = str(report_json)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare open-source gaze data for Gaze-WAM from either a generic image manifest "
            "or video-gaze metadata."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", default=None, help="Generic image/heatmap gaze manifest.")
    source.add_argument("--video-metadata", default=None, help="Raw or canonical video gaze metadata.")
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--adapted-metadata", default=None)
    parser.add_argument(
        "--metadata-inspect-json",
        default=None,
        help="Optional JSON report with suggested raw video metadata key mappings.",
    )
    parser.add_argument("--metadata-inspect-sample-rows", type=int, default=200)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--key-map", default=None, help="JSON object or path for raw metadata field mapping.")
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--gaze-x-key", default=None)
    parser.add_argument("--gaze-y-key", default=None)
    parser.add_argument("--episode-key", default=None)
    parser.add_argument("--frame-key", default=None)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--width-key", default=None)
    parser.add_argument("--height-key", default=None)
    parser.add_argument("--filter", dest="filters", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-missing", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument(
        "--image-resize-mode",
        choices=("stretch",),
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gaze-bounds-policy",
        choices=("error", "drop", "clip"),
        default="error",
        help="How to handle point labels outside [0,1] after normalization.",
    )
    parser.add_argument("--label-mode", choices=("auto", "point", "heatmap"), default="auto")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--heatmap-key", default="gaze_heatmap")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--n-latency-steps", type=int, default=0)
    parser.add_argument(
        "--heatmap-token-grid",
        type=int,
        nargs=2,
        default=(16, 16),
        metavar=("H", "W"),
    )
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--preview-sample-index", type=int, default=0)
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the open-data preparation plan and write optional reports without extracting "
            "frames, writing zarr, running validation, or generating previews."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = prepare_open_gaze_wam_zarr(
        output_zarr=args.output_zarr,
        report_json=args.report_json,
        preview_dir=args.preview_dir,
        manifest_path=args.manifest,
        video_metadata_path=args.video_metadata,
        adapted_metadata_path=args.adapted_metadata,
        metadata_inspect_json=args.metadata_inspect_json,
        metadata_inspect_sample_rows=args.metadata_inspect_sample_rows,
        output_manifest=args.output_manifest,
        frames_dir=args.frames_dir,
        root_dir=args.root_dir,
        key_map=_load_key_map_arg(args.key_map),
        video_key=args.video_key,
        gaze_x_key=args.gaze_x_key,
        gaze_y_key=args.gaze_y_key,
        episode_key=args.episode_key,
        frame_key=args.frame_key,
        timestamp_key=args.timestamp_key,
        width_key=args.width_key,
        height_key=args.height_key,
        filters=args.filters,
        limit=args.limit,
        drop_missing=args.drop_missing,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        gaze_is_normalized=args.gaze_is_normalized,
        gaze_bounds_policy=args.gaze_bounds_policy,
        label_mode=args.label_mode,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        overwrite=args.overwrite,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        n_latency_steps=args.n_latency_steps,
        heatmap_token_grid=args.heatmap_token_grid,
        require_timestamps=args.require_timestamps,
        timestamp_max_step=args.timestamp_max_step,
        preview_sample_index=args.preview_sample_index,
        skip_preview=args.skip_preview,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
