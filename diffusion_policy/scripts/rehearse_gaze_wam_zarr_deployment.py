import argparse
import json

from diffusion_policy.real_world.gaze_wam_zarr_replay import (
    run_gaze_wam_checkpoint_zarr_deployment_rehearsal,
    run_gaze_wam_config_zarr_deployment_rehearsal,
    safety_config_from_json,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a canonical robot zarr through the hardware-agnostic Gaze-WAM "
            "deployment runner and record scheduled absolute commands."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default=None, help="Workspace checkpoint or output dir.")
    source.add_argument(
        "--config-name",
        default=None,
        help=(
            "Hydra workspace config for untrained smoke rehearsal when no checkpoint exists, "
            "for example train_gaze_wam_debug_workspace."
        ),
    )
    parser.add_argument("--dataset-path", required=True, help="Canonical robot zarr path.")
    parser.add_argument("--output-json", required=True, help="Where to write the rehearsal record.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override for --config-name mode. Repeat for multiple overrides.",
    )
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--adapter-camera-key", default="camera0_rgb")

    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--tcp-pose-key", default="tcp_pose_abs")
    parser.add_argument("--gripper-key", default="gripper_width")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--heatmap-key", default="gaze_heatmap")
    parser.add_argument("--action-base-abs-key", default=None)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--skip-zarr-validation", action="store_true")
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)

    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--missing-gaze", action="store_true")

    parser.add_argument("--command-dt", type=float, default=0.1)
    parser.add_argument("--command-start-delay", type=float, default=0.05)
    parser.add_argument("--max-commands-per-step", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--replay-base-time", type=float, default=0.0)
    parser.add_argument("--replay-dt", type=float, default=0.1)
    parser.add_argument(
        "--safety-json",
        default=None,
        help=(
            "Optional JSON file with GazeWamSafetyConfig fields: position_min, position_max, "
            "gripper_min, gripper_max, and max_position_step."
        ),
    )
    parser.add_argument("--send-to-sink", action="store_true", help="Reserved for future hardware sinks.")
    parser.add_argument(
        "--skip-prediction-summary",
        action="store_true",
        help="Do not include min/max/mean summaries of prediction arrays.",
    )
    parser.add_argument(
        "--fail-on-late-commands",
        action="store_true",
        help=(
            "Fail if timing diagnostics show any scheduled command target time is already "
            "in the past when the prediction becomes available."
        ),
    )
    args = parser.parse_args()

    if args.send_to_sink:
        raise NotImplementedError(
            "This CLI is intentionally offline-only for now. Bind real hardware through "
            "GazeWamDeploymentRunner directly."
        )

    common_kwargs = dict(
        dataset_path=args.dataset_path,
        output_json=args.output_json,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        adapter_camera_key=args.adapter_camera_key,
        cfg_scale=args.cfg_scale,
        camera_key=args.camera_key,
        tcp_pose_key=args.tcp_pose_key,
        gripper_key=args.gripper_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        action_base_abs_key=args.action_base_abs_key,
        timestamp_key=args.timestamp_key,
        validate_zarr=not args.skip_zarr_validation,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        episode_index=args.episode_index,
        start_offset=args.start_offset,
        max_steps=args.max_steps,
        stride=args.stride,
        missing_gaze=args.missing_gaze,
        command_dt=args.command_dt,
        command_start_delay=args.command_start_delay,
        max_commands_per_step=args.max_commands_per_step,
        dry_run=True,
        replay_base_time=args.replay_base_time,
        replay_dt=args.replay_dt,
        include_prediction_summary=not args.skip_prediction_summary,
        safety=safety_config_from_json(args.safety_json),
        fail_on_late_commands=args.fail_on_late_commands,
    )
    if args.checkpoint is not None:
        summary = run_gaze_wam_checkpoint_zarr_deployment_rehearsal(
            checkpoint_path=args.checkpoint,
            use_ema=not args.no_ema,
            **common_kwargs,
        )
    else:
        summary = run_gaze_wam_config_zarr_deployment_rehearsal(
            config_name=args.config_name,
            overrides=args.override,
            **common_kwargs,
        )
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "policy_source": summary["policy_source"],
                "num_steps": summary["num_steps"],
                "num_commands": summary["num_commands"],
                "num_clipped_commands": summary["num_clipped_commands"],
                "timing_summary": summary["timing_summary"],
                "missing_gaze": summary["missing_gaze"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
