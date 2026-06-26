import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from diffusion_policy.real_world.gaze_wam_deployment_bindings import (
    build_gaze_wam_deployment_runner_from_config,
)
from diffusion_policy.real_world.gaze_wam_zarr_replay import (
    build_config_rehearsal_adapter,
    scheduled_command_to_dict,
    validate_config_rehearsal_robot_zarr,
)


def _read_json(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _close_provider_handles(runner) -> None:
    for handle in getattr(runner, "provider_handles", []):
        for method_name in ("close", "release"):
            if hasattr(handle, method_name):
                getattr(handle, method_name)()
                break


def _command_summary(commands) -> Dict[str, Any]:
    return {
        "num_commands": len(commands),
        "num_clipped_commands": int(sum(bool(command.was_clipped) for command in commands)),
        "commands": [scheduled_command_to_dict(command) for command in commands],
    }


def run_gaze_wam_split_deployment_rehearsal(
    adapter: Any,
    deployment_config: Dict[str, Any],
    output_json: Optional[str] = None,
    max_steps: int = 1,
    start_time: float = 0.0,
    step_dt: float = 0.1,
    dispatch: bool = False,
    clock=None,
) -> Dict[str, Any]:
    """Run split-provider deployment rehearsal without requiring a canonical zarr source."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if step_dt <= 0:
        raise ValueError("step_dt must be positive.")

    cfg = dict(deployment_config)
    cfg["dry_run"] = not bool(dispatch)
    runner = build_gaze_wam_deployment_runner_from_config(
        adapter=adapter,
        config=cfg,
        clock=clock,
    )
    records = []
    try:
        for step_idx in range(int(max_steps)):
            now = float(start_time + step_idx * step_dt)
            output = runner.step(now=now)
            records.append(
                {
                    "step": int(step_idx),
                    "now": now,
                    "dry_run": bool(output["dry_run"]),
                    **_command_summary(output.get("commands", [])),
                }
            )
    finally:
        _close_provider_handles(runner)

    sink = getattr(runner, "command_sink_handle", None)
    payload = {
        "num_steps": len(records),
        "num_commands": int(sum(record["num_commands"] for record in records)),
        "num_clipped_commands": int(sum(record["num_clipped_commands"] for record in records)),
        "dispatch": bool(dispatch),
        "records": records,
        "command_sink": sink.to_dict() if hasattr(sink, "to_dict") else None,
    }
    _write_json(output_json, payload)
    return payload


def run_gaze_wam_config_split_deployment_rehearsal(
    config_name: str,
    robot_dataset_path: str,
    deployment_config: Dict[str, Any],
    output_json: Optional[str] = None,
    device: str = "cpu",
    overrides: Optional[Sequence[str]] = None,
    num_inference_steps: Optional[int] = None,
    adapter_camera_key: str = "camera0_rgb",
    cfg_scale: float = 1.0,
    max_steps: int = 1,
    start_time: float = 0.0,
    step_dt: float = 0.1,
    dispatch: bool = False,
    validate_zarr: bool = True,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
) -> Dict[str, Any]:
    validation = None
    if validate_zarr:
        validation = validate_config_rehearsal_robot_zarr(
            config_name=config_name,
            dataset_path=robot_dataset_path,
            overrides=overrides,
            timestamp_key=timestamp_key,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
        )
    adapter, cfg = build_config_rehearsal_adapter(
        config_name=config_name,
        dataset_path=robot_dataset_path,
        device=device,
        overrides=overrides,
        num_inference_steps=num_inference_steps,
        adapter_camera_key=adapter_camera_key,
        cfg_scale=cfg_scale,
    )
    payload = run_gaze_wam_split_deployment_rehearsal(
        adapter=adapter,
        deployment_config=deployment_config,
        output_json=output_json,
        max_steps=max_steps,
        start_time=start_time,
        step_dt=step_dt,
        dispatch=dispatch,
    )
    payload["policy_source"] = "config"
    payload["config_name"] = config_name
    payload["checkpoint_path"] = None
    payload["resolved_task_name"] = str(cfg.task.name)
    payload["zarr_validation"] = validation if validation is not None else {"skipped": True}
    _write_json(output_json, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Gaze-WAM deployment rehearsal from split providers such as OpenCV video, "
            "JSONL gaze, JSONL robot state, and JSONL command sinks."
        )
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="Hydra workspace config for untrained/config-mode smoke rehearsal.",
    )
    parser.add_argument(
        "--robot-dataset-path",
        required=True,
        help="Canonical robot zarr used only to fit the policy normalizer in config mode.",
    )
    parser.add_argument(
        "--deployment-config",
        required=True,
        help="JSON file containing image_provider/state_provider/gaze_provider/command_sink config.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override for config mode. Repeat for multiple overrides.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--adapter-camera-key", default="camera0_rgb")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--step-dt", type=float, default=0.1)
    parser.add_argument("--skip-zarr-validation", action="store_true")
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument(
        "--dispatch",
        action="store_true",
        help="Dispatch commands to the configured sink. Default is dry-run only.",
    )
    args = parser.parse_args()

    deployment_config = _read_json(args.deployment_config)
    summary = run_gaze_wam_config_split_deployment_rehearsal(
        config_name=args.config_name,
        robot_dataset_path=args.robot_dataset_path,
        deployment_config=deployment_config,
        output_json=args.output_json,
        device=args.device,
        overrides=args.override,
        num_inference_steps=args.num_inference_steps,
        adapter_camera_key=args.adapter_camera_key,
        cfg_scale=args.cfg_scale,
        max_steps=args.max_steps,
        start_time=args.start_time,
        step_dt=args.step_dt,
        dispatch=args.dispatch,
        validate_zarr=not args.skip_zarr_validation,
        timestamp_key=args.timestamp_key,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
    )
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "policy_source": summary["policy_source"],
                "num_steps": summary["num_steps"],
                "num_commands": summary["num_commands"],
                "num_clipped_commands": summary["num_clipped_commands"],
                "dispatch": summary["dispatch"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
