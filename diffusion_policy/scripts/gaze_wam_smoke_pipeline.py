from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Sequence

build_gaze_wam_split_deployment_config = None
cv2 = None
np = None
preflight_gaze_wam = None
run_gaze_wam_config_split_deployment_rehearsal = None
run_gaze_wam_config_zarr_deployment_rehearsal = None
validate_gaze_wam_zarr = None
write_dataset = None
zarr = None


def _ensure_smoke_core_runtime(needs_debug_writer: bool = False):
    global preflight_gaze_wam
    global validate_gaze_wam_zarr
    global write_dataset
    if needs_debug_writer and write_dataset is None:
        from diffusion_policy.scripts.generate_gaze_wam_debug_data import write_dataset as _write_dataset

        write_dataset = _write_dataset
    if preflight_gaze_wam is None:
        from diffusion_policy.scripts.preflight_gaze_wam import preflight_gaze_wam as _preflight_gaze_wam

        preflight_gaze_wam = _preflight_gaze_wam
    if validate_gaze_wam_zarr is None:
        from diffusion_policy.scripts.validate_gaze_wam_zarr import (
            validate_gaze_wam_zarr as _validate_gaze_wam_zarr,
        )

        validate_gaze_wam_zarr = _validate_gaze_wam_zarr


def _ensure_zarr_deployment_runtime():
    global run_gaze_wam_config_zarr_deployment_rehearsal
    if run_gaze_wam_config_zarr_deployment_rehearsal is None:
        from diffusion_policy.real_world.gaze_wam_zarr_replay import (
            run_gaze_wam_config_zarr_deployment_rehearsal as _run_gaze_wam_config_zarr_deployment_rehearsal,
        )

        run_gaze_wam_config_zarr_deployment_rehearsal = _run_gaze_wam_config_zarr_deployment_rehearsal


def _ensure_split_deployment_runtime():
    global build_gaze_wam_split_deployment_config
    global cv2
    global np
    global run_gaze_wam_config_split_deployment_rehearsal
    global zarr
    if cv2 is None or np is None or zarr is None:
        import cv2 as _cv2
        import numpy as _np
        import zarr as _zarr

        cv2 = _cv2
        np = _np
        zarr = _zarr
    if build_gaze_wam_split_deployment_config is None:
        from diffusion_policy.scripts.make_gaze_wam_split_deployment_config import (
            build_gaze_wam_split_deployment_config as _build_gaze_wam_split_deployment_config,
        )

        build_gaze_wam_split_deployment_config = _build_gaze_wam_split_deployment_config
    if run_gaze_wam_config_split_deployment_rehearsal is None:
        from diffusion_policy.scripts.rehearse_gaze_wam_split_deployment import (
            run_gaze_wam_config_split_deployment_rehearsal as _run_gaze_wam_config_split_deployment_rehearsal,
        )

        run_gaze_wam_config_split_deployment_rehearsal = _run_gaze_wam_config_split_deployment_rehearsal


def _json_safe_path(path: pathlib.Path) -> str:
    return str(path).replace("\\", "/")


def generate_debug_gaze_wam_data(
    output_dir: str,
    num_episodes: int = 2,
    episode_length: int = 24,
    image_size: int = 256,
    image_resize_mode: str = "stretch",
    seed: int = 42,
) -> Dict[str, str]:
    _ensure_smoke_core_runtime(needs_debug_writer=True)
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    robot_path = output / "robot.zarr"
    open_path = output / "open.zarr"
    write_dataset(
        robot_path,
        include_action=True,
        num_episodes=num_episodes,
        episode_length=episode_length,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        seed=seed,
    )
    write_dataset(
        open_path,
        include_action=False,
        num_episodes=num_episodes,
        episode_length=episode_length,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        seed=seed + 1,
    )
    return {
        "robot_path": _json_safe_path(robot_path),
        "open_path": _json_safe_path(open_path),
    }


def _default_overrides(robot_path: str, open_path: str) -> Sequence[str]:
    return [
        f"task.robot_dataset_path={robot_path}",
        f"task.open_dataset_path={open_path}",
        "policy.obs_encoder.pretrained=false",
    ]


def _write_split_provider_inputs(
    robot_dataset_path: str,
    output_dir: pathlib.Path,
    max_steps: int,
    image_size: int,
) -> Dict[str, str]:
    _ensure_split_deployment_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(robot_dataset_path), mode="r")
    data = root["data"] if "data" in root else root
    num_rows = int(min(max_steps, data["camera0_rgb"].shape[0]))
    if num_rows <= 0:
        raise ValueError("Cannot build split-provider inputs from an empty robot dataset.")

    video_path = output_dir / "camera.mp4"
    first_frame = np.asarray(data["camera0_rgb"][0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}.")
    try:
        for i in range(num_rows):
            frame_rgb = np.asarray(data["camera0_rgb"][i])
            if frame_rgb.shape[:2] != (height, width):
                frame_rgb = cv2.resize(frame_rgb, (int(width), int(height)))
            writer.write(cv2.cvtColor(frame_rgb[..., :3], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    gaze_path = output_dir / "gaze.jsonl"
    with gaze_path.open("w", encoding="utf-8") as f:
        gaze = np.asarray(data["gaze_xy"][:num_rows], dtype=np.float32)
        for row in gaze:
            f.write(json.dumps({"gaze_xy": row.reshape(-1).tolist()}) + "\n")

    state_path = output_dir / "state.jsonl"
    with state_path.open("w", encoding="utf-8") as f:
        if "tcp_pose_abs" in data and "gripper_width" in data:
            tcp_pose = np.asarray(data["tcp_pose_abs"][:num_rows], dtype=np.float32)
            gripper = np.asarray(data["gripper_width"][:num_rows], dtype=np.float32)
            for i in range(num_rows):
                row = {
                    "tcp_pose": tcp_pose[i].reshape(-1).tolist(),
                    "gripper_width": float(gripper[i].reshape(-1)[0]),
                }
                f.write(json.dumps(row) + "\n")
        elif "action_abs_tcp" in data:
            action_base = np.asarray(data["action_abs_tcp"][:num_rows], dtype=np.float32)
            for row in action_base:
                f.write(json.dumps({"action_base_abs": row.reshape(-1).tolist()}) + "\n")
        else:
            raise KeyError("Robot dataset must contain tcp_pose_abs/gripper_width or action_abs_tcp.")

    return {
        "video_path": _json_safe_path(video_path),
        "gaze_jsonl": _json_safe_path(gaze_path),
        "state_jsonl": _json_safe_path(state_path),
        "num_rows": int(num_rows),
        "image_size": int(image_size),
    }


def run_gaze_wam_smoke_pipeline(
    config_name: str = "train_gaze_wam_debug_workspace",
    output_dir: str = "data/outputs/gaze_wam_smoke_pipeline",
    generate_debug_data: bool = True,
    debug_data_dir: str = "data/debug_gaze_wam_smoke_pipeline",
    robot_dataset_path: Optional[str] = None,
    open_dataset_path: Optional[str] = None,
    num_episodes: int = 2,
    episode_length: int = 24,
    image_size: int = 256,
    image_resize_mode: str = "stretch",
    seed: int = 42,
    device: str = "cpu",
    max_rehearsal_steps: int = 1,
    max_commands_per_step: int = 1,
    num_inference_steps: int = 1,
    missing_gaze_rehearsal: bool = False,
    run_deployment_rehearsal: bool = False,
    run_split_rehearsal: bool = False,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    fail_on_zarr_warning: bool = False,
    extra_overrides: Optional[Sequence[str]] = None,
    output_json: Optional[str] = None,
) -> Dict[str, object]:
    _ensure_smoke_core_runtime(needs_debug_writer=generate_debug_data)
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "config_name": config_name,
        "output_dir": _json_safe_path(output),
        "device": device,
        "generated_debug_data": bool(generate_debug_data),
        "image_resize_mode": image_resize_mode,
        "run_deployment_rehearsal": bool(run_deployment_rehearsal),
        "run_split_rehearsal": bool(run_split_rehearsal),
        "timestamp_validation_options": {
            "require_timestamps": bool(require_timestamps),
            "timestamp_max_delta": timestamp_max_delta,
            "timestamp_max_step": timestamp_max_step,
            "fail_on_zarr_warning": bool(fail_on_zarr_warning),
        },
    }

    if generate_debug_data:
        generated = generate_debug_gaze_wam_data(
            output_dir=debug_data_dir,
            num_episodes=num_episodes,
            episode_length=episode_length,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            seed=seed,
        )
        robot_dataset_path = generated["robot_path"]
        open_dataset_path = generated["open_path"]
        summary["debug_data"] = {
            **generated,
            "num_episodes": int(num_episodes),
            "episode_length": int(episode_length),
            "image_size": int(image_size),
            "image_resize_mode": image_resize_mode,
            "seed": int(seed),
        }
    else:
        if robot_dataset_path is None or open_dataset_path is None:
            raise ValueError(
                "robot_dataset_path and open_dataset_path are required when "
                "generate_debug_data=False."
            )
        summary["debug_data"] = None

    overrides = list(_default_overrides(robot_dataset_path, open_dataset_path))
    overrides.extend(list(extra_overrides or []))
    summary["overrides"] = overrides
    summary["robot_dataset_path"] = robot_dataset_path
    summary["open_dataset_path"] = open_dataset_path

    robot_validation = validate_gaze_wam_zarr(
        dataset_path=robot_dataset_path,
        dataset_type="robot",
        image_size=(image_size, image_size),
        image_resize_mode=image_resize_mode,
        heatmap_dim=int(image_size // 16) * int(image_size // 16),
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
        check_dataset_sample=True,
    )
    open_validation = validate_gaze_wam_zarr(
        dataset_path=open_dataset_path,
        dataset_type="open",
        image_size=(image_size, image_size),
        image_resize_mode=image_resize_mode,
        heatmap_dim=int(image_size // 16) * int(image_size // 16),
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
        check_dataset_sample=True,
    )
    summary["robot_zarr_validation"] = robot_validation
    summary["open_zarr_validation"] = open_validation
    if not robot_validation["valid"]:
        summary["errors"].append("Robot zarr validation failed.")
    if not open_validation["valid"]:
        summary["errors"].append("Open zarr validation failed.")
    for warning in robot_validation.get("warnings", []) or []:
        summary["errors" if fail_on_zarr_warning else "warnings"].append(
            f"Robot zarr warning: {warning}"
        )
    for warning in open_validation.get("warnings", []) or []:
        summary["errors" if fail_on_zarr_warning else "warnings"].append(
            f"Open zarr warning: {warning}"
        )

    preflight_json = output / "preflight.json"
    preflight = preflight_gaze_wam(
        config_name=config_name,
        overrides=overrides,
        device=device,
        validate_zarr=True,
        run_loss_smoke=True,
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
        fail_on_zarr_warning=fail_on_zarr_warning,
    )
    preflight_json.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["preflight"] = preflight
    summary["preflight_json"] = _json_safe_path(preflight_json)
    if not preflight["ok"]:
        summary["errors"].append("Preflight failed.")

    if run_deployment_rehearsal:
        _ensure_zarr_deployment_runtime()
        rehearsal_json = output / "rehearsal.json"
        rehearsal = run_gaze_wam_config_zarr_deployment_rehearsal(
            config_name=config_name,
            dataset_path=robot_dataset_path,
            output_json=str(rehearsal_json),
            device=device,
            overrides=overrides,
            num_inference_steps=num_inference_steps,
            max_steps=max_rehearsal_steps,
            max_commands_per_step=max_commands_per_step,
            missing_gaze=missing_gaze_rehearsal,
            include_prediction_summary=True,
        )
        summary["rehearsal"] = {
            key: value
            for key, value in rehearsal.items()
            if key != "records"
        }
        summary["rehearsal_record_count"] = len(rehearsal.get("records", []))
        summary["rehearsal_json"] = _json_safe_path(rehearsal_json)
        if rehearsal.get("num_steps", 0) <= 0 or rehearsal.get("num_commands", 0) <= 0:
            summary["errors"].append("Deployment rehearsal produced no commands.")
    else:
        summary["rehearsal"] = None
        summary["rehearsal_record_count"] = 0
        summary["rehearsal_json"] = None
        summary["warnings"].append("Deployment rehearsals skipped for policy-only smoke gate.")

    if run_deployment_rehearsal and run_split_rehearsal:
        _ensure_split_deployment_runtime()
        split_dir = output / "split_rehearsal_inputs"
        split_inputs = _write_split_provider_inputs(
            robot_dataset_path=robot_dataset_path,
            output_dir=split_dir,
            max_steps=max(1, int(max_rehearsal_steps)),
            image_size=image_size,
        )
        split_config_path = output / "split_rehearsal_config.json"
        split_command_jsonl = output / "split_commands.jsonl"
        split_rehearsal_json = output / "split_rehearsal.json"
        split_config = build_gaze_wam_split_deployment_config(
            image_source=split_inputs["video_path"],
            state_path=split_inputs["state_jsonl"],
            gaze_path=split_inputs["gaze_jsonl"],
            command_output_jsonl=_json_safe_path(split_command_jsonl),
            max_commands_per_step=max_commands_per_step,
            dry_run=True,
        )
        split_config_path.write_text(
            json.dumps(split_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_rehearsal = run_gaze_wam_config_split_deployment_rehearsal(
            config_name=config_name,
            robot_dataset_path=robot_dataset_path,
            deployment_config=split_config,
            output_json=str(split_rehearsal_json),
            device=device,
            overrides=overrides,
            num_inference_steps=num_inference_steps,
            max_steps=max_rehearsal_steps,
            dispatch=True,
        )
        summary["split_rehearsal_inputs"] = split_inputs
        summary["split_rehearsal_config_json"] = _json_safe_path(split_config_path)
        summary["split_rehearsal_json"] = _json_safe_path(split_rehearsal_json)
        summary["split_command_jsonl"] = _json_safe_path(split_command_jsonl)
        summary["split_rehearsal"] = {
            key: value
            for key, value in split_rehearsal.items()
            if key != "records"
        }
        if (
            split_rehearsal.get("num_steps", 0) <= 0
            or split_rehearsal.get("num_commands", 0) <= 0
        ):
            summary["errors"].append("Split-provider deployment rehearsal produced no commands.")
    else:
        summary["split_rehearsal_inputs"] = None
        summary["split_rehearsal_config_json"] = None
        summary["split_rehearsal_json"] = None
        summary["split_command_jsonl"] = None
        summary["split_rehearsal"] = None

    summary["ok"] = len(summary["errors"]) == 0
    final_json = pathlib.Path(output_json) if output_json is not None else output / "summary.json"
    final_json.parent.mkdir(parents=True, exist_ok=True)
    final_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary["summary_json"] = _json_safe_path(final_json)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the Gaze-WAM debug data -> validation -> preflight policy-training smoke gate."
        )
    )
    parser.add_argument("--config-name", default="train_gaze_wam_debug_workspace")
    parser.add_argument("--output-dir", default="data/outputs/gaze_wam_smoke_pipeline")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--debug-data-dir", default="data/debug_gaze_wam_smoke_pipeline")
    parser.add_argument("--skip-generate-debug-data", action="store_true")
    parser.add_argument("--robot-dataset-path", default=None)
    parser.add_argument("--open-dataset-path", default=None)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--image-resize-mode",
        choices=("stretch",),
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rehearsal-steps", type=int, default=1)
    parser.add_argument("--max-commands-per-step", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--missing-gaze-rehearsal", action="store_true")
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help=(
            "Run only debug-data generation, zarr validation, and preflight/loss smoke. "
            "This is the default; the flag is kept for explicit policy-training commands."
        ),
    )
    parser.add_argument(
        "--with-deployment-rehearsal",
        action="store_true",
        help=(
            "Also run the reference-only deployment rehearsal stages. "
            "These are deferred for the current policy-training milestone."
        ),
    )
    parser.add_argument("--skip-split-rehearsal", action="store_true")
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--fail-on-zarr-warning", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = run_gaze_wam_smoke_pipeline(
        config_name=args.config_name,
        output_dir=args.output_dir,
        generate_debug_data=not args.skip_generate_debug_data,
        debug_data_dir=args.debug_data_dir,
        robot_dataset_path=args.robot_dataset_path,
        open_dataset_path=args.open_dataset_path,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        seed=args.seed,
        device=args.device,
        max_rehearsal_steps=args.max_rehearsal_steps,
        max_commands_per_step=args.max_commands_per_step,
        num_inference_steps=args.num_inference_steps,
        missing_gaze_rehearsal=args.missing_gaze_rehearsal,
        run_deployment_rehearsal=bool(args.with_deployment_rehearsal and not args.policy_only),
        run_split_rehearsal=bool(args.with_deployment_rehearsal and not args.skip_split_rehearsal),
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        fail_on_zarr_warning=args.fail_on_zarr_warning,
        extra_overrides=args.override,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    if not summary["ok"]:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
