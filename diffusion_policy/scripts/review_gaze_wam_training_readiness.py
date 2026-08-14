from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Sequence

from diffusion_policy.common.gaze_wam_training_config import normalize_gaze_wam_bool_field


launch_gaze_wam_training = None
review_gaze_wam_data_onboarding = None
verify_gaze_wam_dino_source = None


def _json_write(path: Optional[str], payload: Dict[str, object]) -> None:
    if not path:
        return
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_load_object(path: str) -> Dict[str, object]:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must decode to an object: {path}")
    return payload


def _load_json_object_arg(value: Optional[str]) -> Dict[str, str]:
    if value is None:
        return {}
    candidate = pathlib.Path(value)
    try:
        is_path = candidate.exists()
    except OSError:
        is_path = False
    payload = json.loads(candidate.read_text(encoding="utf-8") if is_path else value)
    if not isinstance(payload, dict):
        raise ValueError("JSON argument must decode to an object.")
    return {str(key): str(item) for key, item in payload.items() if item is not None}


def _derive_artifact_path(output_json: Optional[str], suffix: str) -> Optional[str]:
    if output_json is None or str(output_json).strip() == "":
        return None
    path = pathlib.Path(output_json)
    stem = path.stem or "readiness"
    return str(path.with_name(f"{stem}_{suffix}.json"))


def _json_safe_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def _normalize_path(value: object) -> str:
    if value is None:
        return ""
    try:
        return pathlib.Path(str(value)).expanduser().resolve().as_posix().rstrip("/")
    except (OSError, RuntimeError, ValueError):
        return str(value).replace("\\", "/").rstrip("/")


def _signature_from_dino_report(report: Dict[str, object]) -> Dict[str, object]:
    dino_source = report.get("dino_source") or {}
    geometry = report.get("geometry") or {}
    normalization = report.get("normalization") or {}
    return {
        "model_name": dino_source.get("model_name"),
        "pretrained": normalize_gaze_wam_bool_field(
            "dino_source.pretrained",
            dino_source.get("pretrained", False),
            default=False,
        ),
        "checkpoint_path": _normalize_path(dino_source.get("checkpoint_path")),
        "cache_dir": _normalize_path(dino_source.get("cache_dir")),
        "image_size": geometry.get("image_size"),
        "patch_size": geometry.get("patch_size"),
        "expected_tokens_per_frame": geometry.get("expected_tokens_per_frame"),
        "heatmap_token_grid": geometry.get("heatmap_token_grid"),
        "heatmap_num_tokens": geometry.get("heatmap_num_tokens"),
        "mean": normalization.get("mean"),
        "std": normalization.get("std"),
    }


def _artifact_bool(name: str, value, warnings: Optional[list] = None, default: bool = False) -> bool:
    try:
        return normalize_gaze_wam_bool_field(name, value, default=default)
    except ValueError as exc:
        if warnings is not None:
            warnings.append(str(exc))
        return default


def _dino_report_match_check(
    standalone_report: Optional[Dict[str, object]],
    launch_report: Optional[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": bool(standalone_report is not None and launch_report is not None),
        "ok": True,
        "errors": [],
        "standalone_signature": None,
        "launch_signature": None,
        "mismatched_fields": [],
    }
    if standalone_report is None or launch_report is None:
        return summary
    launch_dino = (
        (launch_report.get("real_data_readiness") or {}).get("dino_source_verifier")
        if isinstance(launch_report, dict)
        else None
    )
    if not isinstance(launch_dino, dict):
        summary["ok"] = False
        summary["errors"].append(
            "Launcher report does not contain real_data_readiness.dino_source_verifier."
        )
        return summary
    standalone_sig = _signature_from_dino_report(standalone_report)
    launch_sig = _signature_from_dino_report(launch_dino)
    summary["standalone_signature"] = standalone_sig
    summary["launch_signature"] = launch_sig
    mismatched = [
        key
        for key in standalone_sig
        if standalone_sig.get(key) != launch_sig.get(key)
    ]
    summary["mismatched_fields"] = mismatched
    if mismatched:
        summary["ok"] = False
        summary["errors"].append(
            "Standalone DINO verifier report does not match launcher DINO verifier fields: "
            + ", ".join(mismatched)
            + "."
        )
    return summary


def _launch_preflight_routing_guardrail_check(
    launch_report: Optional[Dict[str, object]],
    *,
    run_launch_preflight: bool,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": bool(run_launch_preflight and launch_report is not None),
        "ok": True,
        "errors": [],
        "preflight_routing_validation_guardrails_ok": None,
    }
    if not summary["enabled"]:
        return summary
    if not isinstance(launch_report, dict):
        summary["ok"] = False
        summary["errors"].append("Launcher report must decode to an object.")
        return summary
    value = launch_report.get("preflight_routing_validation_guardrails_ok")
    summary["preflight_routing_validation_guardrails_ok"] = value
    if value is not True:
        summary["ok"] = False
        summary["errors"].append(
            "Launcher report preflight_routing_validation_guardrails_ok is not true."
        )
    return summary


def _has_onboarding_source(
    *,
    robot_input_path: Optional[str],
    robot_output_path: Optional[str],
    open_manifest_path: Optional[str],
    open_video_metadata_path: Optional[str],
    open_output_zarr: Optional[str],
) -> bool:
    return any(
        value
        for value in (
            robot_input_path,
            robot_output_path,
            open_manifest_path,
            open_video_metadata_path,
            open_output_zarr,
        )
    )


def _ensure_onboarding_runtime() -> None:
    global review_gaze_wam_data_onboarding
    if review_gaze_wam_data_onboarding is None:
        from diffusion_policy.scripts.review_gaze_wam_data_onboarding import (
            review_gaze_wam_data_onboarding as _review_gaze_wam_data_onboarding,
        )

        review_gaze_wam_data_onboarding = _review_gaze_wam_data_onboarding


def _ensure_dino_runtime() -> None:
    global verify_gaze_wam_dino_source
    if verify_gaze_wam_dino_source is None:
        from diffusion_policy.scripts.verify_gaze_wam_dino_source import (
            verify_gaze_wam_dino_source as _verify_gaze_wam_dino_source,
        )

        verify_gaze_wam_dino_source = _verify_gaze_wam_dino_source


def _ensure_launcher_runtime() -> None:
    global launch_gaze_wam_training
    if launch_gaze_wam_training is None:
        from diffusion_policy.scripts.launch_gaze_wam_training import (
            launch_gaze_wam_training as _launch_gaze_wam_training,
        )

        launch_gaze_wam_training = _launch_gaze_wam_training


def review_gaze_wam_training_readiness(
    *,
    output_json: Optional[str] = None,
    onboarding_review_json: Optional[str] = None,
    onboarding_stage_json: Optional[str] = None,
    dino_report_json: Optional[str] = None,
    launch_report_json: Optional[str] = None,
    require_data_onboarding_review: bool = True,
    require_dino_ok: bool = True,
    require_launch_ok: bool = True,
    run_dino_verifier: bool = True,
    run_launch_dry_run: bool = True,
    run_launch_preflight: bool = True,
    config_name: str = "train_gaze_wam_workspace",
    task: str = "gaze_wam",
    overrides: Optional[Sequence[str]] = None,
    use_accelerate: bool = True,
    accelerate_config: str = "accelerate/8gpu-amp.yaml",
    python_bin: str = "py",
    preflight_device: str = "cpu",
    preflight_checkpoint: Optional[str] = None,
    trust_preflight_checkpoint: bool = False,
    preflight_require_timestamps: bool = True,
    preflight_timestamp_max_delta: Optional[float] = None,
    preflight_timestamp_max_step: Optional[float] = None,
    preflight_fail_on_zarr_warning: bool = True,
    real_data_contract: str = "main",
    use_ema: bool = True,
    dino_config_yaml: Optional[str] = None,
    dino_task_yaml: Optional[str] = None,
    dino_model_name: Optional[str] = None,
    dino_expected_model_name: str = "vit_base_patch16_dinov3",
    dino_pretrained: Optional[bool] = None,
    dino_checkpoint_path: Optional[str] = None,
    dino_cache_dir: Optional[str] = None,
    dino_require_local_source: bool = True,
    dino_require_cache_files: bool = False,
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
    preview_sample_index: int = 0,
) -> Dict[str, object]:
    """Write a one-file review bundle for the real-data policy-training gate."""
    overrides = list(overrides or [])
    onboarding_sources_configured = _has_onboarding_source(
        robot_input_path=robot_input_path,
        robot_output_path=robot_output_path,
        open_manifest_path=open_manifest_path,
        open_video_metadata_path=open_video_metadata_path,
        open_output_zarr=open_output_zarr,
    )
    explicit_onboarding_json = onboarding_stage_json or onboarding_review_json
    if explicit_onboarding_json is not None:
        onboarding_stage_json = explicit_onboarding_json
    elif onboarding_sources_configured:
        onboarding_stage_json = _derive_artifact_path(output_json, "onboarding")
    else:
        onboarding_stage_json = None
    dino_report_json = dino_report_json or _derive_artifact_path(output_json, "dino")
    launch_report_json = launch_report_json or _derive_artifact_path(output_json, "launch")

    summary: Dict[str, object] = {
        "ok": True,
        "policy_training_scope": True,
        "deployment_runner_scope": "deferred",
        "errors": [],
        "warnings": [],
        "artifacts": {
            "output_json": _json_safe_path(output_json),
            "data_onboarding_review_json": _json_safe_path(onboarding_stage_json),
            "dino_report_json": _json_safe_path(dino_report_json),
            "launch_report_json": _json_safe_path(launch_report_json),
        },
        "contract": {
            "image_size": [int(v) for v in image_size],
            "image_resize_mode": str(image_resize_mode),
            "n_obs_steps": int(n_obs_steps),
            "action_horizon": int(action_horizon),
            "n_latency_steps": int(n_latency_steps),
            "heatmap_token_grid": [int(v) for v in heatmap_token_grid],
            "preflight_require_timestamps": bool(preflight_require_timestamps),
            "preflight_timestamp_max_delta": preflight_timestamp_max_delta,
            "preflight_timestamp_max_step": preflight_timestamp_max_step,
            "preflight_fail_on_zarr_warning": bool(preflight_fail_on_zarr_warning),
            "real_data_contract": str(real_data_contract),
        },
        "stages": {
            "data_onboarding_review": {
                "enabled": bool(onboarding_sources_configured or explicit_onboarding_json),
                "ran": False,
                "path": _json_safe_path(onboarding_stage_json),
                "ok": not require_data_onboarding_review,
                "summary": None,
            },
            "dino_source_verifier": {
                "enabled": bool(run_dino_verifier),
                "ran": False,
                "path": _json_safe_path(dino_report_json),
                "ok": not require_dino_ok,
                "summary": None,
            },
            "launch_dry_run": {
                "enabled": bool(run_launch_dry_run),
                "ran": False,
                "path": _json_safe_path(launch_report_json),
                "ok": not require_launch_ok,
                "summary": None,
            },
        },
        "cross_checks": {
            "dino_matches_launch": {
                "enabled": False,
                "ok": True,
                "errors": [],
            },
            "launch_preflight_routing_guardrails": {
                "enabled": False,
                "ok": True,
                "errors": [],
                "preflight_routing_validation_guardrails_ok": None,
            },
        },
    }
    errors = summary["errors"]
    warnings = summary["warnings"]

    if onboarding_sources_configured:
        _ensure_onboarding_runtime()
        onboarding_summary = review_gaze_wam_data_onboarding(
            output_json=onboarding_stage_json,
            robot_input_path=robot_input_path,
            robot_output_path=robot_output_path,
            robot_report_json=robot_report_json,
            robot_preview_dir=robot_preview_dir,
            robot_camera_key=robot_camera_key,
            robot_action_key=robot_action_key,
            robot_tcp_pose_key=robot_tcp_pose_key,
            robot_gripper_key=robot_gripper_key,
            robot_gaze_key=robot_gaze_key,
            robot_heatmap_key=robot_heatmap_key,
            robot_timestamp_key=robot_timestamp_key,
            robot_gaze_is_normalized=robot_gaze_is_normalized,
            robot_gaze_bounds_policy=robot_gaze_bounds_policy,
            robot_image_size_for_pixel_gaze=robot_image_size_for_pixel_gaze,
            robot_keep_tcp_dim=robot_keep_tcp_dim,
            robot_inspect_max_items=robot_inspect_max_items,
            robot_inspect_top_k=robot_inspect_top_k,
            open_manifest_path=open_manifest_path,
            open_video_metadata_path=open_video_metadata_path,
            open_output_zarr=open_output_zarr,
            open_report_json=open_report_json,
            open_preview_dir=open_preview_dir,
            open_adapted_metadata_path=open_adapted_metadata_path,
            open_metadata_inspect_json=open_metadata_inspect_json,
            open_metadata_inspect_sample_rows=open_metadata_inspect_sample_rows,
            open_output_manifest=open_output_manifest,
            open_frames_dir=open_frames_dir,
            open_root_dir=open_root_dir,
            open_key_map=open_key_map or {},
            open_video_key=open_video_key,
            open_gaze_x_key=open_gaze_x_key,
            open_gaze_y_key=open_gaze_y_key,
            open_episode_key=open_episode_key,
            open_frame_key=open_frame_key,
            open_timestamp_key=open_timestamp_key,
            open_width_key=open_width_key,
            open_height_key=open_height_key,
            open_filters=open_filters,
            open_limit=open_limit,
            open_drop_missing=open_drop_missing,
            open_gaze_is_normalized=open_gaze_is_normalized,
            open_gaze_bounds_policy=open_gaze_bounds_policy,
            open_label_mode=open_label_mode,
            open_gaze_key=open_gaze_key,
            open_heatmap_key=open_heatmap_key,
            image_size=image_size,
            image_resize_mode=image_resize_mode,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            heatmap_token_grid=heatmap_token_grid,
            require_timestamps=preflight_require_timestamps,
            timestamp_max_delta=preflight_timestamp_max_delta,
            timestamp_max_step=preflight_timestamp_max_step,
            preview_sample_index=preview_sample_index,
        )
        stage = summary["stages"]["data_onboarding_review"]
        stage["ran"] = True
        stage["summary"] = onboarding_summary
        stage["ok"] = _artifact_bool(
            "data_onboarding_review.ok",
            onboarding_summary.get("ok", False),
            warnings,
        )
    elif onboarding_stage_json and pathlib.Path(onboarding_stage_json).exists():
        try:
            onboarding_summary = _json_load_object(onboarding_stage_json)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Could not load data onboarding review JSON: {type(exc).__name__}: {exc}")
        else:
            stage = summary["stages"]["data_onboarding_review"]
            stage["summary"] = onboarding_summary
            stage["ok"] = _artifact_bool(
                "data_onboarding_review.ok",
                onboarding_summary.get("ok", False),
                warnings,
            )
    elif require_data_onboarding_review:
        errors.append(
            "Training readiness review requires an onboarding dry-run artifact; provide "
            "robot/open source arguments or --onboarding-review-json."
        )

    if require_data_onboarding_review:
        onboarding_stage = summary["stages"]["data_onboarding_review"]
        if not _artifact_bool(
            "data_onboarding_review.stage.ok",
            onboarding_stage.get("ok", False),
            warnings,
        ):
            errors.append("Data onboarding review stage is not ok.")

    if run_dino_verifier:
        _ensure_dino_runtime()
        dino_kwargs = {
            "model_name": dino_model_name,
            "expected_model_name": dino_expected_model_name,
            "pretrained": dino_pretrained,
            "checkpoint_path": dino_checkpoint_path,
            "cache_dir": dino_cache_dir,
            "image_size": image_size,
            "heatmap_token_grid": heatmap_token_grid,
            "require_local_source": dino_require_local_source,
            "require_cache_files": dino_require_cache_files,
        }
        if dino_config_yaml is not None:
            dino_kwargs["config_yaml"] = dino_config_yaml
        if dino_task_yaml is not None:
            dino_kwargs["task_yaml"] = dino_task_yaml
        dino_summary = verify_gaze_wam_dino_source(**dino_kwargs)
        _json_write(dino_report_json, dino_summary)
        stage = summary["stages"]["dino_source_verifier"]
        stage["ran"] = True
        stage["summary"] = dino_summary
        stage["ok"] = _artifact_bool(
            "dino_source_verifier.ok",
            dino_summary.get("ok", False),
            warnings,
        )
        warnings.extend(str(warning) for warning in dino_summary.get("warnings", []) or [])
        if require_dino_ok and not stage["ok"]:
            errors.append("DINO source verifier stage is not ok.")

    if run_launch_dry_run:
        _ensure_launcher_runtime()
        launch_onboarding_stage = summary["stages"]["data_onboarding_review"]
        launch_onboarding_json = (
            onboarding_stage_json
            if onboarding_stage_json and launch_onboarding_stage.get("summary") is not None
            else None
        )
        launch_summary = launch_gaze_wam_training(
            config_name=config_name,
            task=task,
            overrides=overrides,
            use_accelerate=use_accelerate,
            accelerate_config=accelerate_config,
            python_bin=python_bin,
            preflight_device=preflight_device,
            preflight_checkpoint=preflight_checkpoint,
            trust_preflight_checkpoint=trust_preflight_checkpoint,
            skip_preflight=not run_launch_preflight,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=preflight_require_timestamps,
            preflight_timestamp_max_delta=preflight_timestamp_max_delta,
            preflight_timestamp_max_step=preflight_timestamp_max_step,
            preflight_fail_on_zarr_warning=preflight_fail_on_zarr_warning,
            real_data=True,
            real_data_contract=real_data_contract,
            data_onboarding_review_json=launch_onboarding_json,
            require_data_onboarding_review=require_data_onboarding_review,
            use_ema=use_ema,
            output_json=launch_report_json,
            run=False,
        )
        stage = summary["stages"]["launch_dry_run"]
        stage["ran"] = True
        stage["summary"] = launch_summary
        stage["ok"] = _artifact_bool(
            "launch_dry_run.ok",
            launch_summary.get("ok", False),
            warnings,
        )
        warnings.extend(str(warning) for warning in launch_summary.get("warnings", []) or [])
        if require_launch_ok and not stage["ok"]:
            errors.append("Launcher real-data dry-run stage is not ok.")

    dino_stage = summary["stages"]["dino_source_verifier"]
    launch_stage = summary["stages"]["launch_dry_run"]
    launch_routing_check = _launch_preflight_routing_guardrail_check(
        launch_stage.get("summary"),
        run_launch_preflight=run_launch_preflight,
    )
    summary["cross_checks"]["launch_preflight_routing_guardrails"] = launch_routing_check
    if not launch_routing_check["ok"]:
        errors.extend(str(error) for error in launch_routing_check.get("errors", []))

    if dino_stage.get("summary") is not None and launch_stage.get("summary") is not None:
        dino_match = _dino_report_match_check(
            dino_stage.get("summary"),
            launch_stage.get("summary"),
        )
        summary["cross_checks"]["dino_matches_launch"] = dino_match
        if not dino_match["ok"]:
            errors.extend(str(error) for error in dino_match.get("errors", []))

    summary["errors"] = list(dict.fromkeys(str(error) for error in errors))
    summary["warnings"] = list(dict.fromkeys(str(warning) for warning in warnings))
    summary["ok"] = len(summary["errors"]) == 0
    _json_write(output_json, summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gaze-WAM policy-training readiness bundle from data onboarding, "
            "DINO source verification, and launcher real-data dry-run reports."
        )
    )
    parser.add_argument("--output-json", default="data/outputs/gaze_wam_readiness/summary.json")
    parser.add_argument("--onboarding-review-json", default=None)
    parser.add_argument("--onboarding-stage-json", default=None)
    parser.add_argument("--dino-report-json", default=None)
    parser.add_argument("--launch-report-json", default=None)
    parser.add_argument(
        "--require-data-onboarding-review",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-dino-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-launch-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dino-verifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-launch-dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-launch-preflight", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--config-name", default="train_gaze_wam_workspace")
    parser.add_argument("--task", default="gaze_wam")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--accelerate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accelerate-config", default="accelerate/8gpu-amp.yaml")
    parser.add_argument("--python-bin", default="py")
    parser.add_argument("--preflight-device", default="cpu")
    parser.add_argument("--preflight-checkpoint", default=None)
    parser.add_argument(
        "--trust-preflight-checkpoint",
        action="store_true",
        help="Acknowledge that the preflight dill checkpoint is trusted and may execute code.",
    )
    parser.add_argument("--preflight-require-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-timestamp-max-delta", type=float, default=None)
    parser.add_argument("--preflight-timestamp-max-step", type=float, default=None)
    parser.add_argument("--preflight-fail-on-zarr-warning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-data-contract", choices=("main", "ablation"), default="main")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--dino-config-yaml", default=None)
    parser.add_argument("--dino-task-yaml", default=None)
    parser.add_argument("--dino-model-name", default=None)
    parser.add_argument("--dino-expected-model-name", default="vit_base_patch16_dinov3")
    parser.add_argument("--dino-pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dino-checkpoint-path", default=None)
    parser.add_argument("--dino-cache-dir", default=None)
    parser.add_argument("--dino-require-local-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dino-require-cache-files", action="store_true")

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
    parser.add_argument("--open-key-map", default=None)
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
    parser.add_argument("--preview-sample-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = review_gaze_wam_training_readiness(
        output_json=args.output_json,
        onboarding_review_json=args.onboarding_review_json,
        onboarding_stage_json=args.onboarding_stage_json,
        dino_report_json=args.dino_report_json,
        launch_report_json=args.launch_report_json,
        require_data_onboarding_review=args.require_data_onboarding_review,
        require_dino_ok=args.require_dino_ok,
        require_launch_ok=args.require_launch_ok,
        run_dino_verifier=args.run_dino_verifier,
        run_launch_dry_run=args.run_launch_dry_run,
        run_launch_preflight=args.run_launch_preflight,
        config_name=args.config_name,
        task=args.task,
        overrides=args.override,
        use_accelerate=args.accelerate,
        accelerate_config=args.accelerate_config,
        python_bin=args.python_bin,
        preflight_device=args.preflight_device,
        preflight_checkpoint=args.preflight_checkpoint,
        trust_preflight_checkpoint=args.trust_preflight_checkpoint,
        preflight_require_timestamps=args.preflight_require_timestamps,
        preflight_timestamp_max_delta=args.preflight_timestamp_max_delta,
        preflight_timestamp_max_step=args.preflight_timestamp_max_step,
        preflight_fail_on_zarr_warning=args.preflight_fail_on_zarr_warning,
        real_data_contract=args.real_data_contract,
        use_ema=args.use_ema,
        dino_config_yaml=args.dino_config_yaml,
        dino_task_yaml=args.dino_task_yaml,
        dino_model_name=args.dino_model_name,
        dino_expected_model_name=args.dino_expected_model_name,
        dino_pretrained=args.dino_pretrained,
        dino_checkpoint_path=args.dino_checkpoint_path,
        dino_cache_dir=args.dino_cache_dir,
        dino_require_local_source=args.dino_require_local_source,
        dino_require_cache_files=args.dino_require_cache_files,
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
        preview_sample_index=args.preview_sample_index,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
