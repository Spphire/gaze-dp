from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Sequence

from diffusion_policy.common.gaze_wam_training_config import normalize_gaze_wam_bool_field


prepare_open_gaze_wam_zarr = None
prepare_robot_gaze_wam_zarr = None


def _ensure_onboarding_runtime(needs_robot: bool, needs_open: bool) -> None:
    global prepare_open_gaze_wam_zarr
    global prepare_robot_gaze_wam_zarr
    if needs_robot and prepare_robot_gaze_wam_zarr is None:
        from diffusion_policy.scripts.prepare_robot_gaze_wam_zarr import (
            prepare_robot_gaze_wam_zarr as _prepare_robot_gaze_wam_zarr,
        )

        prepare_robot_gaze_wam_zarr = _prepare_robot_gaze_wam_zarr
    if needs_open and prepare_open_gaze_wam_zarr is None:
        from diffusion_policy.scripts.prepare_open_gaze_wam_zarr import (
            prepare_open_gaze_wam_zarr as _prepare_open_gaze_wam_zarr,
        )

        prepare_open_gaze_wam_zarr = _prepare_open_gaze_wam_zarr


def _json_write(path: str, payload: Dict[str, object]) -> None:
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_object_arg(value: Optional[str]) -> Dict[str, str]:
    if value is None:
        return {}
    candidate = pathlib.Path(value)
    try:
        is_path = candidate.exists()
    except OSError:
        is_path = False
    if is_path:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must decode to an object.")
    return {str(key): str(item) for key, item in data.items() if item is not None}


def _stage_ok(summary: Optional[Dict[str, object]]) -> bool:
    if summary is None:
        return True
    try:
        return normalize_gaze_wam_bool_field(
            "data_onboarding_stage.ok",
            summary.get("ok", False),
            default=False,
        )
    except ValueError:
        return False


def review_gaze_wam_data_onboarding(
    *,
    output_json: Optional[str] = None,
    robot_input_path: Optional[str] = None,
    robot_output_path: Optional[str] = None,
    robot_report_json: Optional[str] = None,
    robot_preview_dir: Optional[str] = None,
    robot_camera_key: Optional[str] = None,
    robot_action_key: Optional[str] = None,
    robot_tcp_pose_key: Optional[str] = None,
    robot_gripper_key: Optional[str] = None,
    robot_gaze_key: Optional[str] = None,
    robot_heatmap_key: Optional[str] = None,
    robot_timestamp_key: Optional[str] = None,
    robot_gaze_is_normalized: bool = True,
    robot_gaze_bounds_policy: str = "error",
    robot_image_size_for_pixel_gaze: Optional[Sequence[int]] = None,
    robot_keep_tcp_dim: int = 9,
    robot_inspect_max_items: int = 32,
    robot_inspect_top_k: int = 3,
    open_manifest_path: Optional[str] = None,
    open_video_metadata_path: Optional[str] = None,
    open_output_zarr: Optional[str] = None,
    open_report_json: Optional[str] = None,
    open_preview_dir: Optional[str] = None,
    open_adapted_metadata_path: Optional[str] = None,
    open_metadata_inspect_json: Optional[str] = None,
    open_metadata_inspect_sample_rows: int = 200,
    open_output_manifest: Optional[str] = None,
    open_frames_dir: Optional[str] = None,
    open_root_dir: Optional[str] = None,
    open_key_map: Optional[Dict[str, str]] = None,
    open_video_key: Optional[str] = None,
    open_gaze_x_key: Optional[str] = None,
    open_gaze_y_key: Optional[str] = None,
    open_episode_key: Optional[str] = None,
    open_frame_key: Optional[str] = None,
    open_timestamp_key: Optional[str] = None,
    open_width_key: Optional[str] = None,
    open_height_key: Optional[str] = None,
    open_filters: Sequence[str] = (),
    open_limit: Optional[int] = None,
    open_drop_missing: bool = False,
    open_gaze_is_normalized: bool = True,
    open_gaze_bounds_policy: str = "error",
    open_label_mode: str = "auto",
    open_gaze_key: Optional[str] = "gaze_xy",
    open_heatmap_key: Optional[str] = "gaze_heatmap",
    image_size: Sequence[int] = (256, 256),
    image_resize_mode: str = "stretch",
    n_obs_steps: int = 2,
    action_horizon: int = 48,
    n_latency_steps: int = 0,
    heatmap_token_grid: Sequence[int] = (16, 16),
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    preview_sample_index: int = 0,
) -> Dict[str, object]:
    """Run policy-training data onboarding dry-runs for robot and open gaze sources."""
    needs_robot = robot_input_path is not None or robot_output_path is not None
    needs_open = (
        open_manifest_path is not None
        or open_video_metadata_path is not None
        or open_output_zarr is not None
    )
    summary: Dict[str, object] = {
        "ok": True,
        "dry_run": True,
        "policy_training_scope": True,
        "deployment_runner_scope": "deferred",
        "errors": [],
        "warnings": [],
        "selected": {
            "robot": bool(needs_robot),
            "open": bool(needs_open),
        },
        "contract": {
            "image_size": [int(v) for v in image_size],
            "image_resize_mode": str(image_resize_mode),
            "n_obs_steps": int(n_obs_steps),
            "action_horizon": int(action_horizon),
            "n_latency_steps": int(n_latency_steps),
            "heatmap_token_grid": [int(v) for v in heatmap_token_grid],
            "require_timestamps": bool(require_timestamps),
            "timestamp_max_delta": timestamp_max_delta,
            "timestamp_max_step": timestamp_max_step,
        },
        "robot": None,
        "open": None,
    }
    errors = summary["errors"]
    if not needs_robot and not needs_open:
        errors.append("At least one robot or open-source data source must be provided.")
    if needs_robot and (robot_input_path is None or robot_output_path is None):
        errors.append("Robot onboarding requires both robot_input_path and robot_output_path.")
    if needs_open:
        if open_output_zarr is None:
            errors.append("Open onboarding requires open_output_zarr.")
        if (open_manifest_path is None) == (open_video_metadata_path is None):
            errors.append("Open onboarding requires exactly one of open_manifest_path or open_video_metadata_path.")
    if errors:
        summary["ok"] = False
        if output_json is not None:
            _json_write(output_json, summary)
            summary["output_json"] = str(output_json)
        return summary

    _ensure_onboarding_runtime(needs_robot=needs_robot, needs_open=needs_open)

    if needs_robot:
        robot_summary = prepare_robot_gaze_wam_zarr(
            input_path=robot_input_path,
            output_path=robot_output_path,
            report_json=robot_report_json,
            preview_dir=robot_preview_dir,
            camera_key=robot_camera_key,
            action_key=robot_action_key,
            tcp_pose_key=robot_tcp_pose_key,
            gripper_key=robot_gripper_key,
            gaze_key=robot_gaze_key,
            heatmap_key=robot_heatmap_key,
            timestamp_key=robot_timestamp_key,
            gaze_is_normalized=robot_gaze_is_normalized,
            gaze_bounds_policy=robot_gaze_bounds_policy,
            image_size_for_pixel_gaze=robot_image_size_for_pixel_gaze,
            keep_tcp_dim=robot_keep_tcp_dim,
            inspect_max_items=robot_inspect_max_items,
            inspect_top_k=robot_inspect_top_k,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            heatmap_token_grid=heatmap_token_grid,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
            preview_sample_index=preview_sample_index,
            dry_run=True,
        )
        summary["robot"] = robot_summary
        if not _stage_ok(robot_summary):
            errors.append("Robot dry-run onboarding failed.")

    if needs_open:
        open_summary = prepare_open_gaze_wam_zarr(
            output_zarr=open_output_zarr,
            report_json=open_report_json,
            preview_dir=open_preview_dir,
            manifest_path=open_manifest_path,
            video_metadata_path=open_video_metadata_path,
            adapted_metadata_path=open_adapted_metadata_path,
            metadata_inspect_json=open_metadata_inspect_json,
            metadata_inspect_sample_rows=open_metadata_inspect_sample_rows,
            output_manifest=open_output_manifest,
            frames_dir=open_frames_dir,
            root_dir=open_root_dir,
            key_map=open_key_map or {},
            video_key=open_video_key,
            gaze_x_key=open_gaze_x_key,
            gaze_y_key=open_gaze_y_key,
            episode_key=open_episode_key,
            frame_key=open_frame_key,
            timestamp_key=open_timestamp_key,
            width_key=open_width_key,
            height_key=open_height_key,
            filters=open_filters,
            limit=open_limit,
            drop_missing=open_drop_missing,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            gaze_is_normalized=open_gaze_is_normalized,
            gaze_bounds_policy=open_gaze_bounds_policy,
            label_mode=open_label_mode,
            gaze_key=open_gaze_key,
            heatmap_key=open_heatmap_key,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            heatmap_token_grid=heatmap_token_grid,
            require_timestamps=require_timestamps,
            timestamp_max_step=timestamp_max_step,
            preview_sample_index=preview_sample_index,
            dry_run=True,
        )
        summary["open"] = open_summary
        if not _stage_ok(open_summary):
            errors.append("Open dry-run onboarding failed.")

    summary["ok"] = len(errors) == 0
    if output_json is not None:
        _json_write(output_json, summary)
        summary["output_json"] = str(output_json)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Review Gaze-WAM robot/open data onboarding with dry-run preparation reports before "
            "writing zarrs or launching policy training."
        )
    )
    parser.add_argument("--output-json", default=None)

    parser.add_argument("--robot-input-zarr", default=None)
    parser.add_argument("--robot-output-zarr", default=None)
    parser.add_argument("--robot-report-json", default=None)
    parser.add_argument("--robot-preview-dir", default=None)
    parser.add_argument("--robot-camera-key", default=None)
    parser.add_argument("--robot-action-key", default=None)
    parser.add_argument("--robot-tcp-pose-key", default=None)
    parser.add_argument("--robot-gripper-key", default=None)
    parser.add_argument("--robot-gaze-key", default=None)
    parser.add_argument("--robot-heatmap-key", default=None)
    parser.add_argument("--robot-timestamp-key", default=None)
    parser.add_argument("--robot-gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-gaze-bounds-policy", choices=("error", "clip"), default="error")
    parser.add_argument("--robot-image-size-for-pixel-gaze", type=int, nargs=2, default=None)
    parser.add_argument("--robot-keep-tcp-dim", type=int, choices=(9, 10), default=9)
    parser.add_argument("--robot-inspect-max-items", type=int, default=32)
    parser.add_argument("--robot-inspect-top-k", type=int, default=3)

    open_source = parser.add_mutually_exclusive_group()
    open_source.add_argument("--open-manifest", default=None)
    open_source.add_argument("--open-video-metadata", default=None)
    parser.add_argument("--open-output-zarr", default=None)
    parser.add_argument("--open-report-json", default=None)
    parser.add_argument("--open-preview-dir", default=None)
    parser.add_argument("--open-adapted-metadata", default=None)
    parser.add_argument("--open-metadata-inspect-json", default=None)
    parser.add_argument("--open-metadata-inspect-sample-rows", type=int, default=200)
    parser.add_argument("--open-output-manifest", default=None)
    parser.add_argument("--open-frames-dir", default=None)
    parser.add_argument("--open-root-dir", default=None)
    parser.add_argument("--open-key-map", default=None, help="JSON object or path for raw open metadata mapping.")
    parser.add_argument("--open-video-key", default=None)
    parser.add_argument("--open-gaze-x-key", default=None)
    parser.add_argument("--open-gaze-y-key", default=None)
    parser.add_argument("--open-episode-key", default=None)
    parser.add_argument("--open-frame-key", default=None)
    parser.add_argument("--open-timestamp-key", default=None)
    parser.add_argument("--open-width-key", default=None)
    parser.add_argument("--open-height-key", default=None)
    parser.add_argument("--open-filter", dest="open_filters", action="append", default=[])
    parser.add_argument("--open-limit", type=int, default=None)
    parser.add_argument("--open-drop-missing", action="store_true")
    parser.add_argument("--open-gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-gaze-bounds-policy", choices=("error", "drop", "clip"), default="error")
    parser.add_argument("--open-label-mode", choices=("auto", "point", "heatmap"), default="auto")
    parser.add_argument("--open-gaze-key", default="gaze_xy")
    parser.add_argument("--open-heatmap-key", default="gaze_heatmap")

    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--image-resize-mode", choices=("stretch",), default="stretch")
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--n-latency-steps", type=int, default=0)
    parser.add_argument("--heatmap-token-grid", type=int, nargs=2, default=(16, 16), metavar=("H", "W"))
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--preview-sample-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = review_gaze_wam_data_onboarding(
        output_json=args.output_json,
        robot_input_path=args.robot_input_zarr,
        robot_output_path=args.robot_output_zarr,
        robot_report_json=args.robot_report_json,
        robot_preview_dir=args.robot_preview_dir,
        robot_camera_key=args.robot_camera_key,
        robot_action_key=args.robot_action_key,
        robot_tcp_pose_key=args.robot_tcp_pose_key,
        robot_gripper_key=args.robot_gripper_key,
        robot_gaze_key=args.robot_gaze_key,
        robot_heatmap_key=args.robot_heatmap_key,
        robot_timestamp_key=args.robot_timestamp_key,
        robot_gaze_is_normalized=args.robot_gaze_is_normalized,
        robot_gaze_bounds_policy=args.robot_gaze_bounds_policy,
        robot_image_size_for_pixel_gaze=args.robot_image_size_for_pixel_gaze,
        robot_keep_tcp_dim=args.robot_keep_tcp_dim,
        robot_inspect_max_items=args.robot_inspect_max_items,
        robot_inspect_top_k=args.robot_inspect_top_k,
        open_manifest_path=args.open_manifest,
        open_video_metadata_path=args.open_video_metadata,
        open_output_zarr=args.open_output_zarr,
        open_report_json=args.open_report_json,
        open_preview_dir=args.open_preview_dir,
        open_adapted_metadata_path=args.open_adapted_metadata,
        open_metadata_inspect_json=args.open_metadata_inspect_json,
        open_metadata_inspect_sample_rows=args.open_metadata_inspect_sample_rows,
        open_output_manifest=args.open_output_manifest,
        open_frames_dir=args.open_frames_dir,
        open_root_dir=args.open_root_dir,
        open_key_map=_load_json_object_arg(args.open_key_map),
        open_video_key=args.open_video_key,
        open_gaze_x_key=args.open_gaze_x_key,
        open_gaze_y_key=args.open_gaze_y_key,
        open_episode_key=args.open_episode_key,
        open_frame_key=args.open_frame_key,
        open_timestamp_key=args.open_timestamp_key,
        open_width_key=args.open_width_key,
        open_height_key=args.open_height_key,
        open_filters=args.open_filters,
        open_limit=args.open_limit,
        open_drop_missing=args.open_drop_missing,
        open_gaze_is_normalized=args.open_gaze_is_normalized,
        open_gaze_bounds_policy=args.open_gaze_bounds_policy,
        open_label_mode=args.open_label_mode,
        open_gaze_key=args.open_gaze_key,
        open_heatmap_key=args.open_heatmap_key,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        n_latency_steps=args.n_latency_steps,
        heatmap_token_grid=args.heatmap_token_grid,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        preview_sample_index=args.preview_sample_index,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
