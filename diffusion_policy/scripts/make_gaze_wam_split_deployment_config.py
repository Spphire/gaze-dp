import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _json_safe_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _drop_none(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if value is not None}


def _as_finite_flat_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_finite_float(name: str, value: Any) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    value = _validate_finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def _validate_nonnegative_float(name: str, value: Any) -> float:
    value = _validate_finite_float(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return value


def _validate_optional_nonnegative_float(name: str, value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    return _validate_nonnegative_float(name, value)


def _validate_optional_positive_int(name: str, value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_value != value and not (isinstance(value, np.integer) and int_value == int(value)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int_value


def _normalize_static_gripper_width(gripper_width: Optional[Any]) -> Optional[float]:
    if gripper_width is None:
        return None
    arr = _as_finite_flat_array("static_gripper_width", gripper_width)
    if arr.size != 1:
        raise ValueError(
            f"static_gripper_width must contain exactly one scalar, got {arr.size} value(s)."
        )
    return float(arr[0])


def _validate_static_state_config(
    static_action_base_abs: Optional[Any],
    static_tcp_pose: Optional[Any],
    static_gripper_width: Optional[Any],
) -> Optional[float]:
    gripper_width = _normalize_static_gripper_width(static_gripper_width)
    if static_action_base_abs is not None:
        base = _as_finite_flat_array("static_action_base_abs", static_action_base_abs)
        if base.size not in (9, 10):
            raise ValueError(
                "static_action_base_abs must contain 9 pose-only or 10 pose+gripper values, "
                f"got {base.size}."
            )
        if base.size == 9 and gripper_width is None:
            raise ValueError(
                "9D static_action_base_abs requires static_gripper_width so deployment can "
                "build the 10D Gaze-WAM action base."
            )
    if static_tcp_pose is not None:
        tcp_pose = _as_finite_flat_array("static_tcp_pose", static_tcp_pose)
        if tcp_pose.size not in (6, 9, 10):
            raise ValueError(f"static_tcp_pose must contain 6, 9, or 10 values, got {tcp_pose.size}.")
    return gripper_width


def build_gaze_wam_split_deployment_config(
    image_source: str,
    state_path: Optional[str] = None,
    gaze_path: Optional[str] = None,
    command_output_jsonl: Optional[str] = None,
    image_provider_type: str = "opencv_video",
    image_loop: bool = True,
    convert_bgr_to_rgb: bool = True,
    camera_backend: Optional[int] = None,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
    camera_fps: Optional[float] = None,
    camera_warmup_reads: int = 0,
    state_provider_type: str = "jsonl_replay",
    state_eof: str = "hold_last",
    action_base_abs_key: str = "action_base_abs",
    tcp_pose_key: str = "tcp_pose",
    tcp_pose_6d_key: str = "tcp_pose_6d",
    gripper_width_key: str = "gripper_width",
    static_action_base_abs: Optional[Any] = None,
    static_tcp_pose: Optional[Any] = None,
    static_gripper_width: Optional[float] = None,
    gaze_provider_type: str = "jsonl_replay",
    gaze_eof: str = "hold_last",
    gaze_key: Optional[str] = "gaze_xy",
    gaze_x_key: str = "gaze_x",
    gaze_y_key: str = "gaze_y",
    gaze_is_normalized: bool = True,
    gaze_missing: str = "none",
    gaze_clip: bool = True,
    gaze_image_width: Optional[float] = None,
    gaze_image_height: Optional[float] = None,
    command_sink_type: str = "jsonl",
    command_append: bool = False,
    command_flush_each_batch: bool = True,
    command_include_batch_index: bool = True,
    command_dt: float = 0.1,
    command_start_delay: float = 0.0,
    max_commands_per_step: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    dry_run: bool = True,
    safety: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON shape consumed by build_gaze_wam_deployment_runner_from_config."""
    if not image_source:
        raise ValueError("image_source is required.")
    command_dt = _validate_positive_float("command_dt", command_dt)
    command_start_delay = _validate_nonnegative_float(
        "command_start_delay",
        command_start_delay,
    )
    max_commands_per_step = _validate_optional_positive_int(
        "max_commands_per_step",
        max_commands_per_step,
    )
    cfg_scale = _validate_optional_nonnegative_float("cfg_scale", cfg_scale)
    if state_provider_type not in ("jsonl_replay", "jsonl", "static"):
        raise ValueError("state_provider_type must be one of: jsonl_replay, jsonl, static.")
    if state_provider_type in ("jsonl_replay", "jsonl") and not state_path:
        raise ValueError("state_path is required for JSONL state providers.")
    if state_provider_type == "static" and static_action_base_abs is None and static_tcp_pose is None:
        raise ValueError("static state provider requires static_action_base_abs or static_tcp_pose.")
    if state_provider_type == "static":
        static_gripper_width = _validate_static_state_config(
            static_action_base_abs=static_action_base_abs,
            static_tcp_pose=static_tcp_pose,
            static_gripper_width=static_gripper_width,
        )
    if gaze_provider_type not in ("jsonl_replay", "jsonl", "none"):
        raise ValueError("gaze_provider_type must be one of: jsonl_replay, jsonl, none.")
    if gaze_provider_type in ("jsonl_replay", "jsonl") and not gaze_path:
        raise ValueError("gaze_path is required for JSONL gaze providers.")
    if command_sink_type in ("jsonl", "jsonl_queue") and not command_output_jsonl:
        raise ValueError("command_output_jsonl is required for JSONL command sinks.")

    image_provider = _drop_none(
        {
            "type": image_provider_type,
            "source": _json_safe_path(image_source),
            "loop": bool(image_loop),
            "convert_bgr_to_rgb": bool(convert_bgr_to_rgb),
            "backend": camera_backend,
            "width": camera_width,
            "height": camera_height,
            "fps": camera_fps,
            "warmup_reads": int(camera_warmup_reads),
        }
    )

    if state_provider_type == "static":
        state_provider = _drop_none(
            {
                "type": "static",
                "action_base_abs": static_action_base_abs,
                "tcp_pose": static_tcp_pose,
                "gripper_width": static_gripper_width,
            }
        )
    else:
        state_provider = _drop_none(
            {
                "type": state_provider_type,
                "path": _json_safe_path(state_path),
                "eof": state_eof,
                "action_base_abs_key": action_base_abs_key,
                "tcp_pose_key": tcp_pose_key,
                "tcp_pose_6d_key": tcp_pose_6d_key,
                "gripper_width_key": gripper_width_key,
            }
        )

    if gaze_provider_type == "none":
        gaze_provider = {"type": "none"}
    else:
        gaze_provider = _drop_none(
            {
                "type": gaze_provider_type,
                "path": _json_safe_path(gaze_path),
                "eof": gaze_eof,
                "gaze_key": gaze_key,
                "x_key": gaze_x_key,
                "y_key": gaze_y_key,
                "gaze_is_normalized": bool(gaze_is_normalized),
                "missing_gaze": gaze_missing,
                "clip": bool(gaze_clip),
                "image_width": gaze_image_width,
                "image_height": gaze_image_height,
            }
        )

    command_sink = _drop_none(
        {
            "type": command_sink_type,
            "output_jsonl": _json_safe_path(command_output_jsonl),
            "append": bool(command_append),
            "flush_each_batch": bool(command_flush_each_batch),
            "include_batch_index": bool(command_include_batch_index),
        }
    )

    config = _drop_none(
        {
            "image_provider": image_provider,
            "state_provider": state_provider,
            "gaze_provider": gaze_provider,
            "command_sink": command_sink,
            "command_dt": command_dt,
            "command_start_delay": command_start_delay,
            "max_commands_per_step": max_commands_per_step,
            "cfg_scale": cfg_scale,
            "dry_run": bool(dry_run),
            "safety": safety,
        }
    )
    return config


def _parse_json_value(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    maybe_path = Path(value)
    if maybe_path.exists():
        return json.loads(maybe_path.read_text(encoding="utf-8"))
    return json.loads(value)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Gaze-WAM split-provider deployment config JSON."
    )
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--image-provider-type", default="opencv_video")
    parser.add_argument("--image-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--convert-bgr-to-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-backend", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument("--camera-warmup-reads", type=int, default=0)

    parser.add_argument("--state-provider-type", default="jsonl_replay")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--state-eof", default="hold_last")
    parser.add_argument("--action-base-abs-key", default="action_base_abs")
    parser.add_argument("--tcp-pose-key", default="tcp_pose")
    parser.add_argument("--tcp-pose-6d-key", default="tcp_pose_6d")
    parser.add_argument("--gripper-width-key", default="gripper_width")
    parser.add_argument("--static-action-base-abs", default=None)
    parser.add_argument("--static-tcp-pose", default=None)
    parser.add_argument("--static-gripper-width", type=float, default=None)

    parser.add_argument("--gaze-provider-type", default="jsonl_replay")
    parser.add_argument("--gaze-path", default=None)
    parser.add_argument("--gaze-eof", default="hold_last")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--gaze-x-key", default="gaze_x")
    parser.add_argument("--gaze-y-key", default="gaze_y")
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaze-missing", default="none")
    parser.add_argument("--gaze-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaze-image-width", type=float, default=None)
    parser.add_argument("--gaze-image-height", type=float, default=None)

    parser.add_argument("--command-sink-type", default="jsonl")
    parser.add_argument("--command-output-jsonl", default=None)
    parser.add_argument("--command-append", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--command-flush-each-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--command-include-batch-index",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--command-dt", type=float, default=0.1)
    parser.add_argument("--command-start-delay", type=float, default=0.0)
    parser.add_argument("--max-commands-per-step", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--safety-json",
        default=None,
        help="Safety config as inline JSON or a path to a JSON file.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    config = build_gaze_wam_split_deployment_config(
        image_source=args.image_source,
        state_path=args.state_path,
        gaze_path=args.gaze_path,
        command_output_jsonl=args.command_output_jsonl,
        image_provider_type=args.image_provider_type,
        image_loop=args.image_loop,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        camera_backend=args.camera_backend,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_warmup_reads=args.camera_warmup_reads,
        state_provider_type=args.state_provider_type,
        state_eof=args.state_eof,
        action_base_abs_key=args.action_base_abs_key,
        tcp_pose_key=args.tcp_pose_key,
        tcp_pose_6d_key=args.tcp_pose_6d_key,
        gripper_width_key=args.gripper_width_key,
        static_action_base_abs=_parse_json_value(args.static_action_base_abs),
        static_tcp_pose=_parse_json_value(args.static_tcp_pose),
        static_gripper_width=args.static_gripper_width,
        gaze_provider_type=args.gaze_provider_type,
        gaze_eof=args.gaze_eof,
        gaze_key=args.gaze_key,
        gaze_x_key=args.gaze_x_key,
        gaze_y_key=args.gaze_y_key,
        gaze_is_normalized=args.gaze_is_normalized,
        gaze_missing=args.gaze_missing,
        gaze_clip=args.gaze_clip,
        gaze_image_width=args.gaze_image_width,
        gaze_image_height=args.gaze_image_height,
        command_sink_type=args.command_sink_type,
        command_append=args.command_append,
        command_flush_each_batch=args.command_flush_each_batch,
        command_include_batch_index=args.command_include_batch_index,
        command_dt=args.command_dt,
        command_start_delay=args.command_start_delay,
        max_commands_per_step=args.max_commands_per_step,
        cfg_scale=args.cfg_scale,
        dry_run=args.dry_run,
        safety=_parse_json_value(args.safety_json),
    )
    _write_json(args.output_json, config)
    print(json.dumps({"output_json": args.output_json, "config": config}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
