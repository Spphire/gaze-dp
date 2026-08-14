from __future__ import annotations

import argparse
import json
import pathlib
import shlex
from typing import Dict, Optional, Sequence


def _ensure_robot_prepare_runtime():
    global as_optional_gaze_wam_key
    global canonicalize_robot_gaze_wam_zarr
    global inspect_gaze_wam_zarr
    global preview_gaze_wam_dataset
    global validate_gaze_wam_zarr
    try:
        return canonicalize_robot_gaze_wam_zarr
    except NameError:
        from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key as _as_optional_gaze_wam_key
        from diffusion_policy.scripts.canonicalize_robot_gaze_wam_zarr import (
            canonicalize_robot_gaze_wam_zarr as _canonicalize_robot_gaze_wam_zarr,
        )
        from diffusion_policy.scripts.inspect_gaze_wam_zarr import inspect_gaze_wam_zarr as _inspect_gaze_wam_zarr
        from diffusion_policy.scripts.preview_gaze_wam_dataset import (
            preview_gaze_wam_dataset as _preview_gaze_wam_dataset,
        )
        from diffusion_policy.scripts.validate_gaze_wam_zarr import (
            validate_gaze_wam_zarr as _validate_gaze_wam_zarr,
        )

        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        canonicalize_robot_gaze_wam_zarr = _canonicalize_robot_gaze_wam_zarr
        inspect_gaze_wam_zarr = _inspect_gaze_wam_zarr
        preview_gaze_wam_dataset = _preview_gaze_wam_dataset
        validate_gaze_wam_zarr = _validate_gaze_wam_zarr
        return canonicalize_robot_gaze_wam_zarr


def _suggested_key(inspect_summary: Dict[str, object], role: str) -> Optional[str]:
    suggestions = inspect_summary.get("suggestions", {})
    if not isinstance(suggestions, dict):
        return None
    candidates = suggestions.get(role) or []
    if not candidates:
        return None
    return candidates[0].get("key")


def _resolve_key(explicit: Optional[str], inspect_summary: Dict[str, object], role: str) -> str:
    _ensure_robot_prepare_runtime()
    explicit = as_optional_gaze_wam_key(explicit)
    key = explicit or _suggested_key(inspect_summary, role)
    if key is None:
        raise ValueError(
            f"Could not infer a key for role '{role}'. Pass --{role.replace('_', '-')}-key."
        )
    return key


def _resolve_robot_label_keys(
    gaze_key: Optional[str],
    heatmap_key: Optional[str],
    inspect_summary: Dict[str, object],
) -> Dict[str, Optional[str]]:
    _ensure_robot_prepare_runtime()
    gaze_key = as_optional_gaze_wam_key(gaze_key)
    heatmap_key = as_optional_gaze_wam_key(heatmap_key)
    resolved_gaze_key = gaze_key if gaze_key is not None else _suggested_key(inspect_summary, "gaze")
    resolved_heatmap_key = heatmap_key if heatmap_key is not None else _suggested_key(inspect_summary, "heatmap")
    if resolved_gaze_key is None:
        raise ValueError(
            "Could not infer the required robot point gaze key. Pass --gaze-key for "
            "normalized or pixel point labels; a dense heatmap alone is not a canonical "
            "Gaze-WAM training input."
        )
    return {
        "gaze_key": resolved_gaze_key,
        "heatmap_key": resolved_heatmap_key,
    }


def _resolve_optional_key(
    explicit: Optional[str],
    inspect_summary: Dict[str, object],
    role: str,
) -> Optional[str]:
    _ensure_robot_prepare_runtime()
    explicit = as_optional_gaze_wam_key(explicit)
    if explicit is not None:
        return explicit
    return _suggested_key(inspect_summary, role)


def _has_inspected_key(inspect_summary: Dict[str, object], key: str) -> bool:
    arrays = inspect_summary.get("arrays", [])
    if not isinstance(arrays, list):
        return False
    for summary in arrays:
        if not isinstance(summary, dict):
            continue
        if summary.get("key") == key or summary.get("path") == key:
            return True
    return False


def _resolve_named_optional_key(
    explicit: Optional[str],
    inspect_summary: Dict[str, object],
    canonical_key: str,
) -> Optional[str]:
    _ensure_robot_prepare_runtime()
    explicit = as_optional_gaze_wam_key(explicit)
    if explicit is not None:
        return explicit
    if _has_inspected_key(inspect_summary, canonical_key):
        return canonical_key
    return None


def _robot_canonicalizer_command(
    input_path: str,
    output_path: str,
    key_map: Dict[str, Optional[str]],
    *,
    gaze_is_normalized: bool,
    gaze_bounds_policy: str,
    image_size_for_pixel_gaze: Optional[Sequence[int]],
    keep_tcp_dim: int,
    overwrite: bool,
) -> str:
    command = [
        "py",
        "scripts/canonicalize_robot_gaze_wam_zarr.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--camera-key",
        str(key_map["camera_key"]),
        "--action-key",
        str(key_map["action_key"]),
        "--tcp-pose-key",
        str(key_map["tcp_pose_key"]),
        "--gripper-key",
        str(key_map["gripper_key"]),
    ]
    optional_key_args = (
        ("gaze_key", "--gaze-key"),
        ("heatmap_key", "--heatmap-key"),
        ("timestamp_key", "--timestamp-key"),
        ("image_timestamp_key", "--image-timestamp-key"),
        ("robot_state_timestamp_key", "--robot-state-timestamp-key"),
        ("action_timestamp_key", "--action-timestamp-key"),
        ("gaze_timestamp_key", "--gaze-timestamp-key"),
    )
    for key_name, flag in optional_key_args:
        value = key_map.get(key_name)
        if value is not None:
            command.extend([flag, str(value)])
    command.extend(["--gaze-bounds-policy", str(gaze_bounds_policy)])
    if not gaze_is_normalized:
        command.append("--no-gaze-is-normalized")
    if image_size_for_pixel_gaze is not None:
        command.extend(["--image-size-for-pixel-gaze", *[str(v) for v in image_size_for_pixel_gaze]])
    command.extend(["--keep-tcp-dim", str(int(keep_tcp_dim))])
    if overwrite:
        command.append("--overwrite")
    return " ".join(shlex.quote(str(part)) for part in command)


def prepare_robot_gaze_wam_zarr(
    input_path: str,
    output_path: str,
    report_json: Optional[str] = None,
    preview_dir: Optional[str] = None,
    camera_key: Optional[str] = None,
    action_key: Optional[str] = None,
    tcp_pose_key: Optional[str] = None,
    gripper_key: Optional[str] = None,
    gaze_key: Optional[str] = None,
    heatmap_key: Optional[str] = None,
    timestamp_key: Optional[str] = None,
    output_timestamp_key: str = "timestamp",
    image_timestamp_key: Optional[str] = None,
    robot_state_timestamp_key: Optional[str] = None,
    action_timestamp_key: Optional[str] = None,
    gaze_timestamp_key: Optional[str] = None,
    output_image_timestamp_key: str = "image_timestamp",
    output_robot_state_timestamp_key: str = "robot_state_timestamp",
    output_action_timestamp_key: str = "action_timestamp",
    output_gaze_timestamp_key: str = "gaze_timestamp",
    gaze_is_normalized: bool = True,
    gaze_bounds_policy: str = "error",
    image_size_for_pixel_gaze: Optional[Sequence[int]] = None,
    keep_tcp_dim: int = 9,
    overwrite: bool = False,
    inspect_max_items: int = 32,
    inspect_top_k: int = 3,
    n_obs_steps: int = 2,
    action_horizon: int = 48,
    n_latency_steps: int = 0,
    image_size: Sequence[int] = (256, 256),
    image_resize_mode: str = "stretch",
    heatmap_token_grid: Sequence[int] = (16, 16),
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    preview_sample_index: int = 0,
    skip_preview: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Inspect, canonicalize, validate, and optionally preview a robot zarr."""
    _ensure_robot_prepare_runtime()
    inspect_summary = inspect_gaze_wam_zarr(
        dataset_path=input_path,
        dataset_type="robot",
        max_items=inspect_max_items,
        top_k=inspect_top_k,
    )
    label_key_map = _resolve_robot_label_keys(gaze_key, heatmap_key, inspect_summary)
    key_map = {
        "camera_key": _resolve_key(camera_key, inspect_summary, "camera"),
        "action_key": _resolve_key(action_key, inspect_summary, "action"),
        "tcp_pose_key": _resolve_key(tcp_pose_key, inspect_summary, "tcp_pose"),
        "gripper_key": _resolve_key(gripper_key, inspect_summary, "gripper"),
        "gaze_key": label_key_map["gaze_key"],
        "heatmap_key": label_key_map["heatmap_key"],
        "timestamp_key": _resolve_optional_key(timestamp_key, inspect_summary, "timestamp"),
        "image_timestamp_key": _resolve_named_optional_key(
            image_timestamp_key,
            inspect_summary,
            "image_timestamp",
        ),
        "robot_state_timestamp_key": _resolve_named_optional_key(
            robot_state_timestamp_key,
            inspect_summary,
            "robot_state_timestamp",
        ),
        "action_timestamp_key": _resolve_named_optional_key(
            action_timestamp_key,
            inspect_summary,
            "action_timestamp",
        ),
        "gaze_timestamp_key": _resolve_named_optional_key(
            gaze_timestamp_key,
            inspect_summary,
            "gaze_timestamp",
        ),
    }
    canonicalizer_command = _robot_canonicalizer_command(
        input_path=input_path,
        output_path=output_path,
        key_map=key_map,
        gaze_is_normalized=gaze_is_normalized,
        gaze_bounds_policy=gaze_bounds_policy,
        image_size_for_pixel_gaze=image_size_for_pixel_gaze,
        keep_tcp_dim=keep_tcp_dim,
        overwrite=overwrite,
    )
    if dry_run:
        summary: Dict[str, object] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "key_map": key_map,
            "canonicalizer_command": canonicalizer_command,
            "inspect": inspect_summary,
            "canonicalize": None,
            "validation": None,
            "preview": None,
            "dry_run": True,
            "ok": True,
        }
        if report_json is not None:
            report_path = pathlib.Path(report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary["report_json"] = str(report_path)
        return summary

    canonicalize_summary = canonicalize_robot_gaze_wam_zarr(
        input_path=input_path,
        output_path=output_path,
        camera_key=key_map["camera_key"],
        action_key=key_map["action_key"],
        tcp_pose_key=key_map["tcp_pose_key"],
        gripper_key=key_map["gripper_key"],
        gaze_key=key_map["gaze_key"],
        heatmap_key=key_map["heatmap_key"],
        timestamp_key=key_map["timestamp_key"],
        output_timestamp_key=output_timestamp_key,
        image_timestamp_key=key_map["image_timestamp_key"],
        robot_state_timestamp_key=key_map["robot_state_timestamp_key"],
        action_timestamp_key=key_map["action_timestamp_key"],
        gaze_timestamp_key=key_map["gaze_timestamp_key"],
        output_image_timestamp_key=output_image_timestamp_key,
        output_robot_state_timestamp_key=output_robot_state_timestamp_key,
        output_action_timestamp_key=output_action_timestamp_key,
        output_gaze_timestamp_key=output_gaze_timestamp_key,
        gaze_is_normalized=gaze_is_normalized,
        gaze_bounds_policy=gaze_bounds_policy,
        image_size_for_pixel_gaze=image_size_for_pixel_gaze,
        keep_tcp_dim=keep_tcp_dim,
        overwrite=overwrite,
        validate_output=False,
    )
    validation = validate_gaze_wam_zarr(
        dataset_path=output_path,
        dataset_type="robot",
        gaze_key="gaze_xy",
        heatmap_key="gaze_heatmap",
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
        n_latency_steps=n_latency_steps,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        heatmap_token_grid=heatmap_token_grid,
        heatmap_dim=int(image_size[0] // heatmap_token_grid[0])
        * int(image_size[1] // heatmap_token_grid[1]),
        timestamp_key=output_timestamp_key if key_map["timestamp_key"] is not None else None,
        image_timestamp_key=(
            output_image_timestamp_key if key_map["image_timestamp_key"] is not None else None
        ),
        robot_state_timestamp_key=(
            output_robot_state_timestamp_key if key_map["robot_state_timestamp_key"] is not None else None
        ),
        action_timestamp_key=(
            output_action_timestamp_key if key_map["action_timestamp_key"] is not None else None
        ),
        gaze_timestamp_key=(
            output_gaze_timestamp_key if key_map["gaze_timestamp_key"] is not None else None
        ),
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
        check_dataset_sample=True,
    )
    preview = None
    if not skip_preview:
        preview_output = preview_dir or str(pathlib.Path(output_path).with_suffix("")) + "_preview"
        preview = preview_gaze_wam_dataset(
            dataset_path=output_path,
            dataset_type="robot",
            output_dir=preview_output,
            sample_index=preview_sample_index,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            heatmap_token_grid=heatmap_token_grid,
        )

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "key_map": key_map,
        "canonicalizer_command": canonicalizer_command,
        "inspect": inspect_summary,
        "canonicalize": canonicalize_summary,
        "validation": validation,
        "preview": preview,
        "dry_run": False,
        "ok": bool(validation["valid"]),
    }
    if report_json is not None:
        report_path = pathlib.Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["report_json"] = str(report_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a raw robot zarr for Gaze-WAM: inspect key candidates, canonicalize, "
            "validate, and write a preview artifact."
        )
    )
    parser.add_argument("--input", required=True, help="Raw or non-canonical robot zarr.")
    parser.add_argument("--output", required=True, help="Canonical Gaze-WAM robot zarr output.")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--camera-key", default=None)
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--tcp-pose-key", default=None)
    parser.add_argument("--gripper-key", default=None)
    parser.add_argument("--gaze-key", default=None)
    parser.add_argument("--heatmap-key", default=None)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--output-timestamp-key", default="timestamp")
    parser.add_argument("--image-timestamp-key", default=None)
    parser.add_argument("--robot-state-timestamp-key", default=None)
    parser.add_argument("--action-timestamp-key", default=None)
    parser.add_argument("--gaze-timestamp-key", default=None)
    parser.add_argument("--output-image-timestamp-key", default="image_timestamp")
    parser.add_argument("--output-robot-state-timestamp-key", default="robot_state_timestamp")
    parser.add_argument("--output-action-timestamp-key", default="action_timestamp")
    parser.add_argument("--output-gaze-timestamp-key", default="gaze_timestamp")
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gaze-bounds-policy",
        choices=("error", "clip"),
        default="error",
        help="How to handle gaze outside [0,1] after normalization.",
    )
    parser.add_argument("--image-size-for-pixel-gaze", type=int, nargs=2, default=None)
    parser.add_argument("--keep-tcp-dim", type=int, choices=(9, 10), default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-max-items", type=int, default=32)
    parser.add_argument("--inspect-top-k", type=int, default=3)
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
    parser.add_argument(
        "--heatmap-token-grid",
        type=int,
        nargs=2,
        default=(16, 16),
        metavar=("H", "W"),
    )
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--preview-sample-index", type=int, default=0)
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect and resolve the robot key map without writing the output zarr, running "
            "validation, or generating preview images."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = prepare_robot_gaze_wam_zarr(
        input_path=args.input,
        output_path=args.output,
        report_json=args.report_json,
        preview_dir=args.preview_dir,
        camera_key=args.camera_key,
        action_key=args.action_key,
        tcp_pose_key=args.tcp_pose_key,
        gripper_key=args.gripper_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        timestamp_key=args.timestamp_key,
        output_timestamp_key=args.output_timestamp_key,
        image_timestamp_key=args.image_timestamp_key,
        robot_state_timestamp_key=args.robot_state_timestamp_key,
        action_timestamp_key=args.action_timestamp_key,
        gaze_timestamp_key=args.gaze_timestamp_key,
        output_image_timestamp_key=args.output_image_timestamp_key,
        output_robot_state_timestamp_key=args.output_robot_state_timestamp_key,
        output_action_timestamp_key=args.output_action_timestamp_key,
        output_gaze_timestamp_key=args.output_gaze_timestamp_key,
        gaze_is_normalized=args.gaze_is_normalized,
        gaze_bounds_policy=args.gaze_bounds_policy,
        image_size_for_pixel_gaze=args.image_size_for_pixel_gaze,
        keep_tcp_dim=args.keep_tcp_dim,
        overwrite=args.overwrite,
        inspect_max_items=args.inspect_max_items,
        inspect_top_k=args.inspect_top_k,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        n_latency_steps=args.n_latency_steps,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        heatmap_token_grid=args.heatmap_token_grid,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        preview_sample_index=args.preview_sample_index,
        skip_preview=args.skip_preview,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
