import argparse
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _write_launch_report(output_json: Optional[str], summary: Dict[str, object]) -> None:
    if output_json is None:
        return
    output = pathlib.Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _launch_report_path_is_writable_candidate(output_json: Optional[str]) -> bool:
    if output_json is None or str(output_json).strip() == "":
        return False
    path = pathlib.Path(output_json)
    if path.exists() and path.is_dir():
        return False
    parent = path.parent
    probe = parent
    while not probe.exists():
        if probe == probe.parent:
            return False
        probe = probe.parent
    return probe.is_dir()


def _preflight_loss_routing_validation_guardrails_ok(preflight_summary) -> bool:
    if not isinstance(preflight_summary, dict):
        return False
    policy_contract = preflight_summary.get("policy_contract")
    if not isinstance(policy_contract, dict):
        return False
    loss_routing_contract = policy_contract.get("loss_routing_contract")
    from diffusion_policy.common.gaze_wam_training_config import (
        gaze_wam_loss_routing_validation_guardrails_ok,
    )

    return gaze_wam_loss_routing_validation_guardrails_ok(loss_routing_contract)


def build_gaze_wam_train_command(
    config_name: str = "train_gaze_wam_workspace",
    task: str = "gaze_wam",
    overrides: Optional[Sequence[str]] = None,
    use_accelerate: bool = True,
    accelerate_config: str = "accelerate/8gpu-amp.yaml",
    python_bin: str = "py",
) -> Sequence[str]:
    command = []
    if use_accelerate:
        command = [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "train.py",
        ]
    else:
        command = [python_bin, "train.py"]
    command.extend(["--config-name", config_name, f"task={task}"])
    command.extend(list(overrides or []))
    return command


def _load_accelerate_config(path: str) -> Dict[str, object]:
    from omegaconf import OmegaConf

    config_path = pathlib.Path(path)
    summary: Dict[str, object] = {
        "path": str(path),
        "exists": config_path.exists(),
        "distributed_type": None,
        "mixed_precision": None,
        "num_processes": 1,
        "gpu_ids": "",
        "use_cpu": None,
    }
    if not config_path.exists():
        return summary
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    summary.update(
        {
            "distributed_type": cfg.get("distributed_type"),
            "mixed_precision": cfg.get("mixed_precision"),
            "num_processes": int(cfg.get("num_processes", 1) or 1),
            "gpu_ids": str(cfg.get("gpu_ids", "")),
            "use_cpu": cfg.get("use_cpu"),
        }
    )
    return summary


def _training_acceleration_summary(
    config_name: str,
    task: str,
    overrides: Sequence[str],
    use_accelerate: bool,
    accelerate_config: str,
) -> Dict[str, object]:
    from diffusion_policy.scripts.eval_gaze_wam_metrics import load_cfg
    from diffusion_policy.common.gaze_wam_training_config import (
        validate_gaze_wam_training_config,
        validate_gaze_wam_task_routing_config,
    )

    cfg = load_cfg(config_name, overrides=[f"task={task}"] + list(overrides))
    training_config = validate_gaze_wam_training_config(cfg)
    task_routing_config = validate_gaze_wam_task_routing_config(cfg)
    robot_batch_size = int(training_config.get("robot_batch_size", 0))
    open_batch_size = int(training_config.get("open_batch_size", 0))
    grad_accum = int(training_config.get("gradient_accumulate_every", 0))
    per_process_batch_size = int(training_config.get("train_batch_size_per_process", 0))
    accelerator_cfg = (
        _load_accelerate_config(accelerate_config)
        if use_accelerate
        else {
            "path": "",
            "exists": False,
            "distributed_type": "NO",
            "mixed_precision": "no",
            "num_processes": 1,
            "gpu_ids": "",
            "use_cpu": None,
        }
    )
    num_processes = int(accelerator_cfg.get("num_processes") or 1)
    effective_batch_size = per_process_batch_size * num_processes * grad_accum
    effective_robot_batch_size = robot_batch_size * num_processes * grad_accum
    effective_open_batch_size = open_batch_size * num_processes * grad_accum
    warnings = []
    if use_accelerate and not accelerator_cfg["exists"]:
        warnings.append(f"Accelerate config does not exist: {accelerate_config}")
    if use_accelerate and str(accelerator_cfg.get("mixed_precision")).lower() != "bf16":
        warnings.append(
            "Gaze-WAM recommends bf16 mixed precision for DINO/DiT stability; "
            f"got {accelerator_cfg.get('mixed_precision')!r}."
        )
    if use_accelerate and num_processes <= 1:
        warnings.append("Accelerate launch is enabled but num_processes <= 1.")
    errors = list(training_config.get("errors", []))
    errors.extend(task_routing_config.get("errors", []))
    return {
        "use_accelerate": bool(use_accelerate),
        "accelerate_config": accelerator_cfg,
        "training_config": training_config,
        "task_routing_config": task_routing_config,
        "robot_batch_size_per_process": robot_batch_size,
        "open_batch_size_per_process": open_batch_size,
        "train_batch_size_per_process": per_process_batch_size,
        "gradient_accumulate_every": grad_accum,
        "num_processes": num_processes,
        "effective_robot_batch_size_per_optimizer_step": effective_robot_batch_size,
        "effective_open_batch_size_per_optimizer_step": effective_open_batch_size,
        "effective_train_batch_size_per_optimizer_step": effective_batch_size,
        "effective_train_batch_size": effective_batch_size,
        "mixed_precision": accelerator_cfg.get("mixed_precision"),
        "errors": errors,
        "warnings": warnings,
    }


def _as_bool(value, default: bool = False, name: str = "value") -> bool:
    from diffusion_policy.common.gaze_wam_training_config import (
        normalize_gaze_wam_bool_field,
    )

    return normalize_gaze_wam_bool_field(name, value, default=default)


def _as_float(value, default: float = 0.0, name: str = "value") -> float:
    from diffusion_policy.common.gaze_wam_training_config import (
        normalize_gaze_wam_nonnegative_float_field,
    )

    return normalize_gaze_wam_nonnegative_float_field(name, value, default=default)


def _as_positive_float(value, default: float = 1.0, name: str = "value") -> float:
    from diffusion_policy.common.gaze_wam_training_config import (
        normalize_gaze_wam_positive_float_field,
    )

    return normalize_gaze_wam_positive_float_field(name, value, default=default)


def _is_positive_finite(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _looks_like_debug_data_path(value) -> bool:
    text = str(value).replace("\\", "/").lower().rstrip("/")
    parts = [part for part in text.split("/") if part]
    if not parts:
        return False
    store_idx = next(
        (idx for idx, part in enumerate(parts) if part.endswith(".zarr")),
        len(parts) - 1,
    )
    start_idx = max(0, store_idx - 1)
    parts = parts[start_idx : store_idx + 1]
    debug_markers = ("debug", "smoke", "synthetic")
    exact_markers = {"tmp", "temp"}
    return any(any(marker in part for marker in debug_markers) for part in parts) or any(
        part in exact_markers for part in parts
    )


def _looks_like_zarr_path(value) -> bool:
    return str(value).replace("\\", "/").lower().rstrip("/").endswith(".zarr")


def _cfg_get_str(container, key: str, default: str) -> str:
    if container is None:
        return default
    try:
        value = container.get(key, default)
    except AttributeError:
        value = getattr(container, key, default)
    if value is None:
        return default
    return str(value)


def _cfg_get_path_str(container, key: str) -> str:
    value = _cfg_get_str(container, key, "")
    return value.strip()


def _configured_path_exists(value: str) -> bool:
    return bool(value) and pathlib.Path(value).exists()


def _configured_path_is_file(value: str) -> bool:
    return bool(value) and pathlib.Path(value).is_file()


def _configured_path_is_dir(value: str) -> bool:
    return bool(value) and pathlib.Path(value).is_dir()


def _jsonable_attr(value):
    if hasattr(value, "tolist"):
        return _jsonable_attr(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        return [_jsonable_attr(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_attr(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_zarr_metadata_attrs(path: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "path": str(path),
        "exists": pathlib.Path(path).exists(),
        "readable": False,
        "metadata_group": None,
        "metadata_attrs": {},
        "errors": [],
    }
    if not summary["exists"]:
        summary["errors"].append(f"Zarr path does not exist: {path!r}.")
        return summary
    try:
        import zarr
    except ImportError as exc:
        summary["errors"].append(f"Could not import zarr to read metadata: {exc}.")
        return summary
    store = None
    try:
        if str(path).endswith(".zip"):
            store = zarr.ZipStore(path, mode="r")
            root = zarr.group(store=store)
        else:
            root = zarr.open(path, mode="r")
        if "meta" in root:
            attrs = root["meta"].attrs
            summary["metadata_group"] = "meta"
        else:
            attrs = root.attrs
            summary["metadata_group"] = "root"
        summary["metadata_attrs"] = {
            str(key): _jsonable_attr(value)
            for key, value in attrs.items()
        }
        summary["readable"] = True
    except Exception as exc:
        summary["errors"].append(f"Could not read zarr metadata from {path!r}: {type(exc).__name__}: {exc}")
    finally:
        if store is not None:
            store.close()
    return summary


def _metadata_image_size_matches(attrs: Dict[str, object], expected: Sequence[int]) -> bool:
    value = attrs.get("image_size")
    if value is None or isinstance(value, str):
        return False
    try:
        values = [int(item) for item in list(value)]
    except (TypeError, ValueError):
        return False
    return values == [int(item) for item in expected]


def _cfg_list(value) -> List[int]:
    return [int(item) for item in list(value)]


def _cfg_float_list(value) -> List[float]:
    return [float(item) for item in list(value)]


def _obs_encoder_normalize_stats(obs_encoder_cfg) -> Dict[str, Optional[List[float]]]:
    stats: Dict[str, Optional[List[float]]] = {"mean": None, "std": None}
    if obs_encoder_cfg is None:
        return stats
    try:
        transforms = obs_encoder_cfg.get("transforms", [])
    except AttributeError:
        transforms = getattr(obs_encoder_cfg, "transforms", [])
    for transform in transforms or []:
        target = _cfg_get_str(transform, "_target_", "")
        if not target.endswith("Normalize"):
            continue
        try:
            mean = transform.get("mean", None)
            std = transform.get("std", None)
        except AttributeError:
            mean = getattr(transform, "mean", None)
            std = getattr(transform, "std", None)
        stats["mean"] = _cfg_float_list(mean) if mean is not None else None
        stats["std"] = _cfg_float_list(std) if std is not None else None
        break
    return stats


def _normalize_path_for_compare(value: object) -> str:
    if value is None:
        return ""
    try:
        return pathlib.Path(str(value)).expanduser().resolve().as_posix().rstrip("/")
    except (OSError, RuntimeError, ValueError):
        return str(value).replace("\\", "/").rstrip("/")


def _same_configured_path(a: object, b: object) -> bool:
    return _normalize_path_for_compare(a) == _normalize_path_for_compare(b)


def _read_data_onboarding_review(
    *,
    path: Optional[str],
    required: bool,
    robot_dataset_path: str,
    open_dataset_path: str,
    image_size: Sequence[int],
    image_resize_mode: str,
    task_sampling: Dict[str, int],
    heatmap_token_grid: Sequence[int],
    preflight_require_timestamps: bool,
    preflight_timestamp_max_delta: Optional[float],
    preflight_timestamp_max_step: Optional[float],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": bool(path),
        "required": bool(required),
        "path": str(path) if path else None,
        "exists": False,
        "loaded": False,
        "ok": not bool(required) if not path else False,
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    def add_check(name: str, ok: bool, message: str, *, severity: str = "error"):
        summary["checks"].append(
            {
                "name": name,
                "ok": bool(ok),
                "severity": severity,
                "message": message,
            }
        )

    add_check(
        "data_onboarding_review_path_configured",
        (not required) or bool(path),
        "Real-data launch requires --data-onboarding-review-json before training.",
    )
    if not path:
        summary["errors"] = [
            check["message"]
            for check in summary["checks"]
            if check["severity"] == "error" and not check["ok"]
        ]
        summary["ok"] = len(summary["errors"]) == 0
        return summary

    review_path = pathlib.Path(path)
    summary["exists"] = review_path.exists()
    add_check(
        "data_onboarding_review_path_exists",
        review_path.exists(),
        f"Data onboarding review JSON does not exist: {path!r}.",
    )
    report = None
    if review_path.exists():
        try:
            report = json.loads(review_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary["warnings"].append(f"Could not load data onboarding review JSON: {type(exc).__name__}: {exc}")
        else:
            summary["loaded"] = isinstance(report, dict)
    add_check(
        "data_onboarding_review_loaded",
        isinstance(report, dict),
        "Data onboarding review JSON must decode to a JSON object.",
    )

    if isinstance(report, dict):
        selected = report.get("selected") if isinstance(report.get("selected"), dict) else {}
        contract = report.get("contract") if isinstance(report.get("contract"), dict) else {}
        robot = report.get("robot") if isinstance(report.get("robot"), dict) else {}
        open_data = report.get("open") if isinstance(report.get("open"), dict) else {}

        def report_bool(container, key: str, default: bool = False) -> bool:
            try:
                return _as_bool(
                    container.get(key, default),
                    default=default,
                    name=f"data_onboarding_review.{key}",
                )
            except ValueError as exc:
                summary["warnings"].append(str(exc))
                return default

        report_ok = report_bool(report, "ok")
        report_dry_run = report_bool(report, "dry_run")
        report_policy_training_scope = report_bool(report, "policy_training_scope")
        selected_robot = report_bool(selected, "robot")
        selected_open = report_bool(selected, "open")
        contract_require_timestamps = report_bool(contract, "require_timestamps")
        summary["report"] = {
            "ok": report_ok,
            "dry_run": report_dry_run,
            "policy_training_scope": report_policy_training_scope,
            "deployment_runner_scope": report.get("deployment_runner_scope"),
            "selected": selected,
            "contract": contract,
            "robot_output_path": robot.get("output_path") if isinstance(robot, dict) else None,
            "open_output_zarr": open_data.get("output_zarr") if isinstance(open_data, dict) else None,
        }
        add_check(
            "data_onboarding_review_ok",
            report_ok,
            "Data onboarding review report must have ok=true.",
        )
        add_check(
            "data_onboarding_review_dry_run",
            report_dry_run,
            "Data onboarding review report must be a dry-run artifact.",
        )
        add_check(
            "data_onboarding_review_policy_scope",
            report_policy_training_scope,
            "Data onboarding review report must mark policy_training_scope=true.",
        )
        add_check(
            "data_onboarding_review_runner_scope_deferred",
            str(report.get("deployment_runner_scope", "")).lower() == "deferred",
            "Data onboarding review report must keep deployment_runner_scope='deferred'.",
        )
        add_check(
            "data_onboarding_review_selected_robot",
            selected_robot and isinstance(robot, dict),
            "Data onboarding review report must include the robot dry-run stage.",
        )
        add_check(
            "data_onboarding_review_selected_open",
            selected_open and isinstance(open_data, dict),
            "Data onboarding review report must include the open-data dry-run stage.",
        )
        add_check(
            "data_onboarding_review_robot_output_matches_config",
            isinstance(robot, dict)
            and _same_configured_path(robot.get("output_path"), robot_dataset_path),
            (
                "Data onboarding review robot output_path must match the configured robot "
                f"dataset path {robot_dataset_path!r}."
            ),
        )
        add_check(
            "data_onboarding_review_open_output_matches_config",
            isinstance(open_data, dict)
            and _same_configured_path(open_data.get("output_zarr"), open_dataset_path),
            (
                "Data onboarding review open output_zarr must match the configured open "
                f"dataset path {open_dataset_path!r}."
            ),
        )
        add_check(
            "data_onboarding_review_image_size_matches_config",
            [int(v) for v in contract.get("image_size", [])] == [int(v) for v in image_size],
            (
                "Data onboarding review image_size must match the launch config; "
                f"expected {[int(v) for v in image_size]!r}."
            ),
        )
        add_check(
            "data_onboarding_review_resize_mode_matches_config",
            str(contract.get("image_resize_mode", "")) == str(image_resize_mode),
            (
                "Data onboarding review image_resize_mode must match the launch config; "
                f"expected {image_resize_mode!r}."
            ),
        )
        add_check(
            "data_onboarding_review_sampling_matches_config",
            {
                "n_obs_steps": int(contract.get("n_obs_steps", -1)),
                "action_horizon": int(contract.get("action_horizon", -1)),
                "n_latency_steps": int(contract.get("n_latency_steps", -1)),
            }
            == task_sampling,
            (
                "Data onboarding review temporal sampling must match the launch config; "
                f"expected {task_sampling!r}."
            ),
        )
        add_check(
            "data_onboarding_review_heatmap_grid_matches_config",
            [int(v) for v in contract.get("heatmap_token_grid", [])]
            == [int(v) for v in heatmap_token_grid],
            (
                "Data onboarding review heatmap_token_grid must match the launch config; "
                f"expected {[int(v) for v in heatmap_token_grid]!r}."
            ),
        )
        add_check(
            "data_onboarding_review_timestamp_requirement_matches_launch",
            contract_require_timestamps == bool(preflight_require_timestamps),
            (
                "Data onboarding review require_timestamps must match the launch timestamp gate; "
                f"expected {bool(preflight_require_timestamps)!r}."
            ),
        )
        if preflight_timestamp_max_step is not None:
            add_check(
                "data_onboarding_review_timestamp_max_step_matches_launch",
                contract.get("timestamp_max_step") == preflight_timestamp_max_step,
                (
                    "Data onboarding review timestamp_max_step must match the launch timestamp "
                    f"gate; expected {preflight_timestamp_max_step!r}."
                ),
            )
        if preflight_timestamp_max_delta is not None:
            add_check(
                "data_onboarding_review_timestamp_max_delta_matches_launch",
                contract.get("timestamp_max_delta") == preflight_timestamp_max_delta,
                (
                    "Data onboarding review timestamp_max_delta must match the launch timestamp "
                    f"gate; expected {preflight_timestamp_max_delta!r}."
                ),
            )

    summary["errors"] = [
        check["message"]
        for check in summary["checks"]
        if check["severity"] == "error" and not check["ok"]
    ]
    summary["ok"] = len(summary["errors"]) == 0
    return summary


def check_real_data_readiness(
    *,
    config_name: str,
    cfg,
    acceleration: Dict[str, object],
    output_json: Optional[str],
    skip_preflight: bool,
    skip_zarr_validation: bool,
    skip_loss_smoke: bool,
    preflight_require_timestamps: bool,
    preflight_timestamp_max_delta: Optional[float],
    preflight_timestamp_max_step: Optional[float],
    preflight_fail_on_zarr_warning: bool,
    use_accelerate: bool,
    contract: str = "main",
    data_onboarding_review_json: Optional[str] = None,
    require_data_onboarding_review: bool = False,
) -> Dict[str, object]:
    """Return launch-blocking checks for a real Gaze-WAM training run."""
    from diffusion_policy.scripts.verify_gaze_wam_dino_source import (
        verify_gaze_wam_dino_source,
    )
    from diffusion_policy.common.gaze_wam_training_config import (
        normalize_gaze_wam_nonnegative_int_field,
        normalize_gaze_wam_positive_int_field,
        normalize_gaze_wam_positive_int_sequence,
        validate_gaze_wam_task_routing_config,
        validate_gaze_wam_training_config,
    )

    contract = str(contract).strip().lower()
    if contract not in {"main", "ablation"}:
        raise ValueError(f"Unknown real-data launch contract {contract!r}; expected 'main' or 'ablation'.")
    enforce_main_contract = contract == "main"
    checks: List[Dict[str, object]] = []

    def add_check(name: str, ok: bool, message: str, *, severity: str = "error"):
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "severity": severity,
                "message": message,
            }
        )

    config_parse_errors: List[str] = []

    def parse_positive_int(name: str, value, *, default: int = 1, fallback: int = -1) -> int:
        try:
            return normalize_gaze_wam_positive_int_field(name, value, default=default)
        except ValueError as exc:
            config_parse_errors.append(str(exc))
            return int(fallback)

    def parse_nonnegative_int(name: str, value, *, default: int = 0, fallback: int = -1) -> int:
        try:
            return normalize_gaze_wam_nonnegative_int_field(name, value, default=default)
        except ValueError as exc:
            config_parse_errors.append(str(exc))
            return int(fallback)

    def parse_positive_int_sequence(
        name: str,
        value,
        *,
        length: Optional[int] = None,
        fallback: Optional[Sequence[int]] = None,
    ) -> List[int]:
        try:
            return normalize_gaze_wam_positive_int_sequence(name, value, length=length)
        except ValueError as exc:
            config_parse_errors.append(str(exc))
            return [int(item) for item in (fallback or [])]

    lower_config_name = config_name.lower()
    task_resize_mode = _cfg_get_str(getattr(cfg, "task", None), "image_resize_mode", "stretch")
    robot_resize_mode = _cfg_get_str(cfg.task.get("robot_dataset", None), "image_resize_mode", task_resize_mode)
    open_resize_mode = _cfg_get_str(cfg.task.get("open_dataset", None), "image_resize_mode", task_resize_mode)
    resize_modes = {
        "task": task_resize_mode,
        "robot_dataset": robot_resize_mode,
        "open_dataset": open_resize_mode,
    }
    image_shape = parse_positive_int_sequence(
        "task.image_shape",
        cfg.task.image_shape,
        length=3,
        fallback=[3, 256, 256],
    )
    task_image_size = image_shape[-2:]
    robot_image_size = parse_positive_int_sequence(
        "task.robot_dataset.image_size",
        cfg.task.robot_dataset.image_size,
        length=2,
        fallback=task_image_size,
    )
    open_image_size = parse_positive_int_sequence(
        "task.open_dataset.image_size",
        cfg.task.open_dataset.image_size,
        length=2,
        fallback=task_image_size,
    )
    image_sizes = {
        "task": task_image_size,
        "robot_dataset": robot_image_size,
        "open_dataset": open_image_size,
    }
    task_sampling = {
        "n_obs_steps": parse_positive_int("task.n_obs_steps", cfg.task.n_obs_steps),
        "action_horizon": parse_positive_int("task.action_horizon", cfg.task.action_horizon),
        "n_latency_steps": parse_nonnegative_int(
            "task.n_latency_steps",
            cfg.task.get("n_latency_steps", 0),
        ),
    }
    robot_sampling = {
        "n_obs_steps": parse_positive_int(
            "task.robot_dataset.n_obs_steps",
            cfg.task.robot_dataset.get("n_obs_steps", 0),
        ),
        "action_horizon": parse_positive_int(
            "task.robot_dataset.action_horizon",
            cfg.task.robot_dataset.get("action_horizon", 0),
        ),
        "n_latency_steps": parse_nonnegative_int(
            "task.robot_dataset.n_latency_steps",
            cfg.task.robot_dataset.get("n_latency_steps", 0),
        ),
    }
    open_sampling = {
        "n_obs_steps": parse_positive_int(
            "task.open_dataset.n_obs_steps",
            cfg.task.open_dataset.get("n_obs_steps", 0),
        ),
        "action_horizon": parse_positive_int(
            "task.open_dataset.action_horizon",
            cfg.task.open_dataset.get("action_horizon", 0),
        ),
        "n_latency_steps": parse_nonnegative_int(
            "task.open_dataset.n_latency_steps",
            cfg.task.open_dataset.get("n_latency_steps", 0),
        ),
    }
    action_dim = parse_positive_int("task.action_dim", cfg.task.action_dim)
    heatmap_token_grid = parse_positive_int_sequence(
        "task.heatmap_token_grid",
        cfg.task.heatmap_token_grid,
        length=2,
        fallback=[16, 16],
    )
    heatmap_num_tokens = parse_positive_int(
        "task.heatmap_num_tokens",
        cfg.task.heatmap_num_tokens,
    )
    heatmap_dim = parse_positive_int(
        "task.heatmap_dim",
        cfg.task.heatmap_dim,
    )
    heatmap_spatial_decoder = _cfg_get_str(
        cfg.policy,
        "heatmap_spatial_decoder",
        "cosmos_tokenizer",
    )
    heatmap_patch_area = 0
    if len(heatmap_token_grid) == 2:
        if (
            task_image_size[0] % heatmap_token_grid[0] == 0
            and task_image_size[1] % heatmap_token_grid[1] == 0
        ):
            heatmap_patch_area = (
                task_image_size[0]
                // heatmap_token_grid[0]
                * task_image_size[1]
                // heatmap_token_grid[1]
            )
    obs_encoder_model_name = _cfg_get_str(cfg.policy.obs_encoder, "model_name", "")
    obs_encoder_pretrained = _as_bool(cfg.policy.obs_encoder.get("pretrained", False))
    obs_encoder_checkpoint_path = _cfg_get_path_str(cfg.policy.obs_encoder, "checkpoint_path")
    obs_encoder_cache_dir = _cfg_get_path_str(cfg.policy.obs_encoder, "cache_dir")
    obs_encoder_downsample_ratio = parse_positive_int(
        "policy.obs_encoder.downsample_ratio",
        cfg.policy.obs_encoder.get("downsample_ratio", 16),
        default=16,
    )
    obs_encoder_normalize = _obs_encoder_normalize_stats(cfg.policy.obs_encoder)
    dino_source_verifier = verify_gaze_wam_dino_source(
        model_name=obs_encoder_model_name,
        pretrained=obs_encoder_pretrained,
        checkpoint_path=obs_encoder_checkpoint_path,
        cache_dir=obs_encoder_cache_dir,
        image_size=task_image_size,
        patch_size=obs_encoder_downsample_ratio,
        heatmap_token_grid=heatmap_token_grid,
        heatmap_num_tokens=heatmap_num_tokens,
        normalize_mean=obs_encoder_normalize["mean"],
        normalize_std=obs_encoder_normalize["std"],
        require_local_source=True,
    )
    obs_encoder_checkpoint_path_is_file = (
        (not obs_encoder_checkpoint_path)
        or _configured_path_is_file(obs_encoder_checkpoint_path)
    )
    obs_encoder_cache_dir_is_dir = (
        (not obs_encoder_cache_dir)
        or _configured_path_is_dir(obs_encoder_cache_dir)
    )
    obs_encoder_local_weight_source_valid = (
        obs_encoder_checkpoint_path_is_file and obs_encoder_cache_dir_is_dir
    )
    dino_cache_contains_files = (
        (dino_source_verifier.get("dino_source") or {}).get("cache_dir_contains_files") is True
    )
    dino_cache_only_source = bool(obs_encoder_cache_dir) and not bool(obs_encoder_checkpoint_path)
    robot_dataset_path = str(cfg.task.robot_dataset.dataset_path)
    open_dataset_path = str(cfg.task.open_dataset.dataset_path)
    robot_zarr_metadata = _read_zarr_metadata_attrs(robot_dataset_path)
    open_zarr_metadata = _read_zarr_metadata_attrs(open_dataset_path)
    robot_zarr_attrs = robot_zarr_metadata.get("metadata_attrs") or {}
    open_zarr_attrs = open_zarr_metadata.get("metadata_attrs") or {}
    data_onboarding_review = _read_data_onboarding_review(
        path=data_onboarding_review_json,
        required=require_data_onboarding_review,
        robot_dataset_path=robot_dataset_path,
        open_dataset_path=open_dataset_path,
        image_size=task_image_size,
        image_resize_mode=task_resize_mode,
        task_sampling=task_sampling,
        heatmap_token_grid=heatmap_token_grid,
        preflight_require_timestamps=preflight_require_timestamps,
        preflight_timestamp_max_delta=preflight_timestamp_max_delta,
        preflight_timestamp_max_step=preflight_timestamp_max_step,
    )
    training_config = validate_gaze_wam_training_config(cfg)
    task_routing_config = validate_gaze_wam_task_routing_config(cfg)
    robot_batch_size = int(training_config.get("robot_batch_size", 0))
    open_batch_size = int(training_config.get("open_batch_size", 0))
    total_batch_size = robot_batch_size + open_batch_size
    robot_ratio = robot_batch_size / total_batch_size if total_batch_size > 0 else 0.0
    open_ratio = open_batch_size / total_batch_size if total_batch_size > 0 else 0.0
    robot_gaze_dropout_prob = task_routing_config.get("robot_gaze_dropout_prob", 0.0)
    robot_heatmap_on_gaze_dropout = task_routing_config.get(
        "robot_heatmap_on_gaze_dropout",
        True,
    )
    action_loss_weight = _as_float(
        cfg.policy.get("action_loss_weight", 0.0),
        name="policy.action_loss_weight",
    )
    heatmap_loss_weight = _as_float(
        cfg.policy.get("heatmap_loss_weight", 0.0),
        name="policy.heatmap_loss_weight",
    )
    heatmap_token_kl_loss_weight = _as_float(
        cfg.policy.get("heatmap_token_kl_loss_weight", 0.0),
        name="policy.heatmap_token_kl_loss_weight",
    )
    heatmap_xy_loss_weight = _as_float(
        cfg.policy.get("heatmap_xy_loss_weight", 1.0),
        name="policy.heatmap_xy_loss_weight",
    )
    heatmap_point_nll_loss_weight = _as_float(
        cfg.policy.get("heatmap_point_nll_loss_weight", 0.0),
        name="policy.heatmap_point_nll_loss_weight",
    )
    heatmap_js_loss_weight = _as_float(
        cfg.policy.get("heatmap_js_loss_weight", 1.0),
        name="policy.heatmap_js_loss_weight",
    )
    heatmap_dsnt_temperature = _as_positive_float(
        cfg.policy.get("heatmap_dsnt_temperature", 0.1),
        default=0.1,
        name="policy.heatmap_dsnt_temperature",
    )
    heatmap_distribution_mode = _cfg_get_str(
        cfg.policy,
        "heatmap_distribution_mode",
        "intensity_softplus",
    )
    heatmap_objective = _cfg_get_str(cfg.policy, "heatmap_objective", "")
    add_check(
        "non_debug_config_name",
        "debug" not in lower_config_name,
        f"Real-data launch should not use a debug config name; got {config_name!r}.",
    )
    add_check(
        "training_debug_false",
        not _as_bool(cfg.training.get("debug", False)),
        "Real-data launch requires training.debug=false.",
    )
    add_check(
        "preflight_enabled",
        not skip_preflight,
        "Real-data launch requires preflight; remove --skip-preflight.",
    )
    add_check(
        "zarr_validation_enabled",
        not skip_zarr_validation,
        "Real-data launch requires zarr validation; remove --skip-zarr-validation.",
    )
    add_check(
        "loss_smoke_enabled",
        not skip_loss_smoke,
        "Real-data launch requires the preflight loss smoke; remove --skip-loss-smoke.",
    )
    add_check(
        "timestamps_required",
        bool(preflight_require_timestamps),
        "Real-data launch requires --preflight-require-timestamps.",
    )
    add_check(
        "timestamp_threshold_configured",
        _is_positive_finite(preflight_timestamp_max_step)
        or _is_positive_finite(preflight_timestamp_max_delta),
        (
            "Real-data launch requires a finite positive timestamp threshold; pass "
            "--preflight-timestamp-max-step or --preflight-timestamp-max-delta."
        ),
    )
    add_check(
        "zarr_warnings_block",
        bool(preflight_fail_on_zarr_warning),
        "Real-data launch requires --preflight-fail-on-zarr-warning.",
    )
    add_check(
        "launch_report_path",
        output_json is not None and str(output_json).strip() != "",
        "Real-data launch requires --output-json so the launch report is saved for review.",
    )
    add_check(
        "launch_report_path_writable_candidate",
        _launch_report_path_is_writable_candidate(output_json),
        (
            "Real-data launch --output-json must point to a writable report file path; "
            f"got {output_json!r}."
        ),
    )
    add_check(
        "obs_encoder_pretrained",
        obs_encoder_pretrained,
        "Real-data launch requires policy.obs_encoder.pretrained=true.",
    )
    add_check(
        "obs_encoder_local_weight_source_configured",
        bool(obs_encoder_checkpoint_path or obs_encoder_cache_dir),
        (
            "Real-data launch requires an explicit local DINO weight source: set "
            "policy.obs_encoder.checkpoint_path or policy.obs_encoder.cache_dir."
        ),
    )
    if obs_encoder_checkpoint_path:
        add_check(
            "obs_encoder_checkpoint_path_exists",
            _configured_path_exists(obs_encoder_checkpoint_path),
            (
                "Real-data launch policy.obs_encoder.checkpoint_path does not exist: "
                f"{obs_encoder_checkpoint_path!r}."
            ),
        )
        add_check(
            "obs_encoder_checkpoint_path_is_file",
            obs_encoder_checkpoint_path_is_file,
            (
                "Real-data launch policy.obs_encoder.checkpoint_path must point to a file, "
                f"got {obs_encoder_checkpoint_path!r}."
            ),
        )
    if obs_encoder_cache_dir:
        add_check(
            "obs_encoder_cache_dir_exists",
            _configured_path_exists(obs_encoder_cache_dir),
            (
                "Real-data launch policy.obs_encoder.cache_dir does not exist: "
                f"{obs_encoder_cache_dir!r}."
            ),
        )
        add_check(
            "obs_encoder_cache_dir_is_dir",
            obs_encoder_cache_dir_is_dir,
            (
                "Real-data launch policy.obs_encoder.cache_dir must point to a directory, "
                f"got {obs_encoder_cache_dir!r}."
            ),
        )
    add_check(
        "obs_encoder_local_weight_source_valid",
        obs_encoder_local_weight_source_valid,
        (
            "Real-data launch local DINO weight source is not structurally valid; "
            "checkpoint_path must be a file and cache_dir must be a directory."
        ),
    )
    add_check(
        "obs_encoder_cache_dir_contains_files_when_cache_only",
        (not dino_cache_only_source) or dino_cache_contains_files,
        (
            "Real-data launch uses policy.obs_encoder.cache_dir as the only local DINO source, "
            "but that cache directory contains no files; provide a non-empty cache directory or "
            "set policy.obs_encoder.checkpoint_path to a non-empty checkpoint file."
        ),
    )
    add_check(
        "dino_source_verifier_ok",
        bool(dino_source_verifier["ok"]),
        (
            "Real-data launch DINO source/preprocess verifier failed: "
            + "; ".join(str(error) for error in dino_source_verifier["errors"])
        ),
    )
    add_check(
        "robot_dataset_path_zarr",
        _looks_like_zarr_path(robot_dataset_path),
        f"Real-data launch robot dataset path should point to a .zarr store; got {robot_dataset_path!r}.",
    )
    add_check(
        "robot_dataset_path_exists",
        pathlib.Path(robot_dataset_path).exists(),
        f"Real-data launch robot dataset path does not exist: {robot_dataset_path!r}.",
    )
    add_check(
        "robot_zarr_metadata_readable",
        bool(robot_zarr_metadata.get("readable", False)),
        (
            "Real-data launch must be able to read robot zarr metadata attrs before training; "
            + "; ".join(str(error) for error in robot_zarr_metadata.get("errors", []))
        ),
    )
    add_check(
        "robot_zarr_metadata_dataset_type",
        str(robot_zarr_attrs.get("dataset_type", "")) == "robot",
        (
            "Real-data launch robot zarr meta.attrs.dataset_type must be 'robot'; "
            f"got {robot_zarr_attrs.get('dataset_type')!r}."
        ),
    )
    add_check(
        "robot_zarr_metadata_image_resize_mode",
        str(robot_zarr_attrs.get("image_resize_mode", "")) == str(task_resize_mode),
        (
            "Real-data launch robot zarr meta.attrs.image_resize_mode must match the task "
            f"resize mode {task_resize_mode!r}; got {robot_zarr_attrs.get('image_resize_mode')!r}."
        ),
    )
    add_check(
        "robot_zarr_metadata_image_size",
        _metadata_image_size_matches(robot_zarr_attrs, task_image_size),
        (
            "Real-data launch robot zarr meta.attrs.image_size must match task image H/W "
            f"{task_image_size!r}; got {robot_zarr_attrs.get('image_size')!r}."
        ),
    )
    add_check(
        "open_dataset_path_zarr",
        _looks_like_zarr_path(open_dataset_path),
        f"Real-data launch open dataset path should point to a .zarr store; got {open_dataset_path!r}.",
    )
    add_check(
        "open_dataset_path_exists",
        pathlib.Path(open_dataset_path).exists(),
        f"Real-data launch open dataset path does not exist: {open_dataset_path!r}.",
    )
    add_check(
        "open_zarr_metadata_readable",
        bool(open_zarr_metadata.get("readable", False)),
        (
            "Real-data launch must be able to read open-source zarr metadata attrs before training; "
            + "; ".join(str(error) for error in open_zarr_metadata.get("errors", []))
        ),
    )
    add_check(
        "open_zarr_metadata_dataset_type",
        str(open_zarr_attrs.get("dataset_type", "")) == "open",
        (
            "Real-data launch open-source zarr meta.attrs.dataset_type must be 'open'; "
            f"got {open_zarr_attrs.get('dataset_type')!r}."
        ),
    )
    add_check(
        "open_zarr_metadata_image_resize_mode",
        str(open_zarr_attrs.get("image_resize_mode", "")) == str(task_resize_mode),
        (
            "Real-data launch open-source zarr meta.attrs.image_resize_mode must match the task "
            f"resize mode {task_resize_mode!r}; got {open_zarr_attrs.get('image_resize_mode')!r}."
        ),
    )
    add_check(
        "open_zarr_metadata_image_size",
        _metadata_image_size_matches(open_zarr_attrs, task_image_size),
        (
            "Real-data launch open-source zarr meta.attrs.image_size must match task image H/W "
            f"{task_image_size!r}; got {open_zarr_attrs.get('image_size')!r}."
        ),
    )
    add_check(
        "robot_open_dataset_paths_distinct",
        robot_dataset_path != open_dataset_path,
        (
            "Real-data launch requires distinct robot and open-source dataset paths; "
            f"both resolved to {robot_dataset_path!r}."
        ),
    )
    add_check(
        "robot_dataset_path_not_debug",
        not _looks_like_debug_data_path(robot_dataset_path),
        (
            "Real-data launch robot dataset path looks like debug/smoke/synthetic/temp data; "
            f"got {robot_dataset_path!r}."
        ),
    )
    add_check(
        "open_dataset_path_not_debug",
        not _looks_like_debug_data_path(open_dataset_path),
        (
            "Real-data launch open dataset path looks like debug/smoke/synthetic/temp data; "
            f"got {open_dataset_path!r}."
        ),
    )
    add_check(
        "image_resize_mode_stretch",
        all(value == "stretch" for value in resize_modes.values()),
        (
            "Real-data launch requires direct-stretch image geometry for task, robot dataset, "
            f"and open dataset; got {resize_modes!r}."
        ),
    )
    add_check(
        "image_resize_mode_consistent",
        len(set(resize_modes.values())) == 1,
        (
            "Real-data launch requires task, robot dataset, and open dataset to share the same "
            f"image_resize_mode; got {resize_modes!r}."
        ),
    )
    add_check(
        "image_shape_256",
        image_shape == [3, 256, 256],
        f"Real-data launch requires task.image_shape=[3, 256, 256]; got {image_shape!r}.",
    )
    add_check(
        "image_size_consistent",
        robot_image_size == task_image_size and open_image_size == task_image_size,
        (
            "Real-data launch requires task.image_shape H/W, robot dataset image_size, "
            f"and open dataset image_size to match; got {image_sizes!r}."
        ),
    )
    add_check(
        "n_obs_steps_2",
        task_sampling["n_obs_steps"] == 2,
        f"Real-data launch requires task.n_obs_steps=2; got {cfg.task.n_obs_steps!r}.",
    )
    add_check(
        "action_horizon_16",
        task_sampling["action_horizon"] == 16,
        f"Real-data launch requires task.action_horizon=16; got {cfg.task.action_horizon!r}.",
    )
    add_check(
        "n_latency_steps_0",
        (not enforce_main_contract) or task_sampling["n_latency_steps"] == 0,
        (
            "Real-data main-contract launch keeps latency compensation disabled for the "
            "first policy-training run; set task.n_latency_steps=0 unless a later data-sync "
            f"contract explicitly enables latency, got {cfg.task.get('n_latency_steps', 0)!r}."
        ),
    )
    add_check(
        "robot_sampling_matches_task",
        robot_sampling == task_sampling,
        (
            "Real-data launch requires robot dataset n_obs_steps, action_horizon, and "
            f"n_latency_steps to match task sampling; got task={task_sampling!r}, "
            f"robot_dataset={robot_sampling!r}."
        ),
    )
    add_check(
        "open_sampling_matches_task",
        open_sampling == task_sampling,
        (
            "Real-data launch requires open dataset n_obs_steps, action_horizon, and "
            f"n_latency_steps to match task sampling; got task={task_sampling!r}, "
            f"open_dataset={open_sampling!r}."
        ),
    )
    add_check(
        "action_dim_10",
        action_dim == 10,
        f"Real-data launch requires task.action_dim=10; got {cfg.task.action_dim!r}.",
    )
    add_check(
        "heatmap_token_grid_16x16",
        heatmap_token_grid == [16, 16],
        f"Real-data launch requires task.heatmap_token_grid=[16, 16]; got {heatmap_token_grid!r}.",
    )
    add_check(
        "heatmap_num_tokens_256",
        heatmap_num_tokens == 256,
        f"Real-data launch requires task.heatmap_num_tokens=256; got {cfg.task.heatmap_num_tokens!r}.",
    )
    add_check(
        "heatmap_dim_16",
        heatmap_dim == 16,
        (
            "Real-data launch requires the canonical FastWAM/LDM-aligned heatmap "
            f"latent channel dim 16; got {cfg.task.heatmap_dim!r}."
        ),
    )
    add_check(
        "heatmap_dim_positive_latent_channels",
        heatmap_dim > 0,
        f"task.heatmap_dim must be a positive latent channel count; got {cfg.task.heatmap_dim!r}.",
    )
    add_check(
        "heatmap_dim_not_lossless_patch_area",
        heatmap_dim != heatmap_patch_area,
        (
            "Real-data launch must not use the old lossless heatmap patch "
            f"area ({heatmap_patch_area}) as task.heatmap_dim."
        ),
    )
    add_check(
        "heatmap_spatial_decoder_cosmos_tokenizer",
        heatmap_spatial_decoder == "cosmos_tokenizer",
        (
            "Real-data launch requires "
            "policy.heatmap_spatial_decoder='cosmos_tokenizer'."
        ),
    )
    add_check(
        "heatmap_grid_product",
        len(heatmap_token_grid) == 2
        and heatmap_token_grid[0] * heatmap_token_grid[1] == heatmap_num_tokens,
        (
            "Real-data launch requires heatmap_token_grid product to match heatmap_num_tokens; "
            f"got grid={heatmap_token_grid!r}, tokens={cfg.task.heatmap_num_tokens!r}."
        ),
    )
    add_check(
        "obs_encoder_dinov3_vit16",
        obs_encoder_model_name == "vit_base_patch16_dinov3",
        (
            "Real-data launch requires policy.obs_encoder.model_name='vit_base_patch16_dinov3'; "
            f"got {obs_encoder_model_name!r}."
        ),
    )
    add_check(
        "block_attention_mask_enabled",
        _as_bool(cfg.policy.get("use_block_attention_mask", False)),
        "Real-data launch requires policy.use_block_attention_mask=true.",
    )
    add_check(
        "core_config_parse_valid",
        len(config_parse_errors) == 0,
        (
            "Real-data launch core geometry/sampling config is invalid: "
            + "; ".join(config_parse_errors)
        ),
    )
    add_check(
        "training_config_valid",
        bool(training_config.get("valid", False)),
        (
            "Real-data launch training-loop config is invalid: "
            + "; ".join(str(error) for error in training_config.get("errors", []))
        ),
    )
    add_check(
        "task_routing_config_valid",
        bool(task_routing_config.get("valid", False)),
        (
            "Real-data launch task routing config is invalid: "
            + "; ".join(str(error) for error in task_routing_config.get("errors", []))
        ),
    )
    add_check(
        "robot_batch_positive",
        robot_batch_size > 0,
        f"Real-data main launch requires robot_dataloader.batch_size > 0; got {robot_batch_size}.",
    )
    add_check(
        "open_batch_positive",
        open_batch_size > 0,
        f"Real-data main launch requires open_dataloader.batch_size > 0; got {open_batch_size}.",
    )
    add_check(
        "source_ratio_75_25",
        (not enforce_main_contract)
        or (
            total_batch_size > 0
            and abs(robot_ratio - 0.75) < 1e-9
            and abs(open_ratio - 0.25) < 1e-9
        ),
        (
            "Real-data main-contract launch requires a 75% robot / 25% open-source gaze batch ratio; "
            f"got robot={robot_batch_size}, open={open_batch_size}, "
            f"robot_ratio={robot_ratio:.6g}, open_ratio={open_ratio:.6g}."
        ),
    )
    add_check(
        "robot_gaze_dropout_prob_0p2",
        (not enforce_main_contract) or abs(robot_gaze_dropout_prob - 0.2) < 1e-9,
        (
            "Real-data main-contract launch requires task.robot_gaze_dropout_prob=0.2; "
            f"got {robot_gaze_dropout_prob!r}."
        ),
    )
    add_check(
        "robot_heatmap_on_gaze_dropout_enabled",
        (not enforce_main_contract) or robot_heatmap_on_gaze_dropout,
        "Real-data main-contract launch requires task.robot_heatmap_on_gaze_dropout=true.",
    )
    add_check(
        "heatmap_objective_dsnt_js",
        (not enforce_main_contract) or heatmap_objective == "dsnt_js",
        (
            "Real-data main-contract launch requires policy.heatmap_objective='dsnt_js'; "
            f"got {heatmap_objective!r}."
        ),
    )
    add_check(
        "action_loss_weight_1",
        abs(action_loss_weight - 1.0) < 1e-9,
        f"Real-data main launch requires policy.action_loss_weight=1.0; got {action_loss_weight!r}.",
    )
    add_check(
        "heatmap_loss_weight_1",
        abs(heatmap_loss_weight - 1.0) < 1e-9,
        f"Real-data main launch requires policy.heatmap_loss_weight=1.0; got {heatmap_loss_weight!r}.",
    )
    add_check(
        "heatmap_token_kl_loss_weight_0",
        (not enforce_main_contract) or abs(heatmap_token_kl_loss_weight - 0.0) < 1e-9,
        (
            "Real-data main-contract launch replaces token-KL heatmap supervision with "
            "DSNT+JS; set policy.heatmap_token_kl_loss_weight=0.0 for the main method; "
            f"got {heatmap_token_kl_loss_weight!r}."
        ),
    )
    add_check(
        "heatmap_xy_loss_weight_1",
        (not enforce_main_contract) or abs(heatmap_xy_loss_weight - 1.0) < 1e-9,
        (
            "Real-data main-contract launch requires policy.heatmap_xy_loss_weight=1.0; "
            f"got {heatmap_xy_loss_weight!r}."
        ),
    )
    add_check(
        "heatmap_point_nll_loss_weight_0",
        (not enforce_main_contract) or abs(heatmap_point_nll_loss_weight) < 1e-9,
        (
            "Real-data main-contract launch requires "
            "policy.heatmap_point_nll_loss_weight=0.0; "
            f"got {heatmap_point_nll_loss_weight!r}."
        ),
    )
    add_check(
        "heatmap_js_loss_weight_1",
        (not enforce_main_contract) or abs(heatmap_js_loss_weight - 1.0) < 1e-9,
        (
            "Real-data main-contract launch requires policy.heatmap_js_loss_weight=1.0; "
            f"got {heatmap_js_loss_weight!r}."
        ),
    )
    add_check(
        "heatmap_distribution_mode_intensity_softplus",
        (not enforce_main_contract)
        or heatmap_distribution_mode == "intensity_softplus",
        (
            "Real-data main-contract launch with frozen Cosmos heatmap decoder "
            "must interpret decoded outputs as intensity; set "
            "policy.heatmap_distribution_mode='intensity_softplus', got "
            f"{heatmap_distribution_mode!r}."
        ),
    )
    add_check(
        "heatmap_dsnt_temperature_0p1",
        (not enforce_main_contract)
        or abs(heatmap_dsnt_temperature - 0.1) < 1e-9,
        (
            "Real-data main-contract launch uses calibrated "
            "policy.heatmap_dsnt_temperature=0.1 for intensity_softplus; "
            f"got {heatmap_dsnt_temperature!r}."
        ),
    )
    add_check(
        "accelerate_enabled",
        bool(use_accelerate),
        "Real-data launch should use Accelerate multi-GPU; pass --accelerate.",
    )
    add_check(
        "accelerate_config_exists",
        bool(acceleration.get("accelerate_config", {}).get("exists", False)) if use_accelerate else False,
        "Real-data launch requires an existing Accelerate config.",
    )
    add_check(
        "accelerate_bf16",
        str(acceleration.get("mixed_precision")).lower() == "bf16",
        f"Real-data launch requires bf16 mixed precision; got {acceleration.get('mixed_precision')!r}.",
    )
    add_check(
        "multi_process",
        int(acceleration.get("num_processes") or 1) > 1,
        f"Real-data launch expects multi-process training; got {acceleration.get('num_processes')!r}.",
    )

    errors = [check["message"] for check in checks if check["severity"] == "error" and not check["ok"]]
    errors.extend(data_onboarding_review["errors"])
    warnings = [check["message"] for check in checks if check["severity"] == "warning" and not check["ok"]]
    warnings.extend(str(warning) for warning in dino_source_verifier["warnings"])
    warnings.extend(str(warning) for warning in data_onboarding_review["warnings"])
    return {
        "enabled": True,
        "contract": contract,
        "ok": len(errors) == 0,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "dino_source_verifier": dino_source_verifier,
        "data_onboarding_review": data_onboarding_review,
        "training_config": training_config,
        "task_routing_config": task_routing_config,
        "zarr_metadata": {
            "robot": robot_zarr_metadata,
            "open": open_zarr_metadata,
        },
    }


def launch_gaze_wam_training(
    config_name: str = "train_gaze_wam_workspace",
    task: str = "gaze_wam",
    overrides: Optional[Sequence[str]] = None,
    use_accelerate: bool = True,
    accelerate_config: str = "accelerate/8gpu-amp.yaml",
    python_bin: str = "py",
    preflight_device: str = "cpu",
    preflight_checkpoint: Optional[str] = None,
    skip_preflight: bool = False,
    skip_zarr_validation: bool = False,
    skip_loss_smoke: bool = False,
    preflight_require_timestamps: bool = False,
    preflight_timestamp_max_delta: Optional[float] = None,
    preflight_timestamp_max_step: Optional[float] = None,
    preflight_fail_on_zarr_warning: bool = False,
    real_data: bool = False,
    real_data_contract: str = "main",
    data_onboarding_review_json: Optional[str] = None,
    require_data_onboarding_review: bool = False,
    use_ema: bool = True,
    output_json: Optional[str] = None,
    run: bool = False,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, object]:
    """Preflight a Gaze-WAM config, build a train command, and optionally execute it."""
    from diffusion_policy.scripts.eval_gaze_wam_metrics import load_cfg
    from diffusion_policy.scripts.preflight_gaze_wam import preflight_gaze_wam

    overrides = list(overrides or [])
    command = list(
        build_gaze_wam_train_command(
            config_name=config_name,
            task=task,
            overrides=overrides,
            use_accelerate=use_accelerate,
            accelerate_config=accelerate_config,
            python_bin=python_bin,
        )
    )
    acceleration = _training_acceleration_summary(
        config_name=config_name,
        task=task,
        overrides=overrides,
        use_accelerate=use_accelerate,
        accelerate_config=accelerate_config,
    )
    cfg = load_cfg(config_name, overrides=[f"task={task}"] + overrides)
    summary: Dict[str, object] = {
        "config_name": config_name,
        "task": task,
        "overrides": overrides,
        "use_accelerate": bool(use_accelerate),
        "accelerate_config": accelerate_config if use_accelerate else "",
        "acceleration": acceleration,
        "command": command,
        "command_str": _command_to_string(command),
        "run": bool(run),
        "real_data": bool(real_data),
        "preflight_options": {
            "require_timestamps": bool(preflight_require_timestamps),
            "timestamp_max_delta": preflight_timestamp_max_delta,
            "timestamp_max_step": preflight_timestamp_max_step,
            "fail_on_zarr_warning": bool(preflight_fail_on_zarr_warning),
        },
        "ok": True,
        "errors": [],
        "warnings": list(acceleration["warnings"]),
        "returncode": None,
        "preflight_routing_validation_guardrails_ok": None,
    }
    if use_accelerate and not acceleration["accelerate_config"]["exists"]:
        summary["ok"] = False
        summary["errors"].append("Accelerate config missing.")
    if acceleration["errors"]:
        summary["ok"] = False
        summary["errors"].append(
            "Training acceleration config invalid: "
            + "; ".join(str(error) for error in acceleration["errors"])
        )

    if real_data:
        readiness = check_real_data_readiness(
            config_name=config_name,
            cfg=cfg,
            acceleration=acceleration,
            output_json=output_json,
            skip_preflight=skip_preflight,
            skip_zarr_validation=skip_zarr_validation,
            skip_loss_smoke=skip_loss_smoke,
            preflight_require_timestamps=preflight_require_timestamps,
            preflight_timestamp_max_delta=preflight_timestamp_max_delta,
            preflight_timestamp_max_step=preflight_timestamp_max_step,
            preflight_fail_on_zarr_warning=preflight_fail_on_zarr_warning,
            use_accelerate=use_accelerate,
            contract=real_data_contract,
            data_onboarding_review_json=data_onboarding_review_json,
            require_data_onboarding_review=require_data_onboarding_review,
        )
        summary["real_data_readiness"] = readiness
        summary["warnings"].extend(readiness["warnings"])
        if not readiness["ok"]:
            summary["ok"] = False
            summary["errors"].extend(readiness["errors"])
    else:
        summary["real_data_readiness"] = {"enabled": False}

    if not skip_preflight:
        preflight = preflight_gaze_wam(
            config_name=config_name,
            overrides=[f"task={task}"] + overrides,
            checkpoint=preflight_checkpoint,
            device=preflight_device,
            validate_zarr=not skip_zarr_validation,
            run_loss_smoke=not skip_loss_smoke,
            use_ema=use_ema,
            require_timestamps=preflight_require_timestamps,
            timestamp_max_delta=preflight_timestamp_max_delta,
            timestamp_max_step=preflight_timestamp_max_step,
            fail_on_zarr_warning=preflight_fail_on_zarr_warning,
        )
        summary["preflight"] = preflight
        preflight_routing_guardrails_ok = _preflight_loss_routing_validation_guardrails_ok(
            preflight
        )
        summary["preflight_routing_validation_guardrails_ok"] = (
            preflight_routing_guardrails_ok
        )
        if not preflight["ok"]:
            summary["ok"] = False
            summary["errors"].append("Preflight failed.")
        if not preflight_routing_guardrails_ok:
            summary["ok"] = False
            summary["errors"].append("Preflight loss-routing validation guardrails failed.")
    else:
        summary["preflight"] = {"skipped": True}

    if run and summary["ok"]:
        if output_json is not None:
            summary["pre_run_report_written"] = True
            _write_launch_report(output_json, summary)
        run_env = os.environ.copy()
        run_env.update(env or {})
        run_env.setdefault("HYDRA_FULL_ERROR", "1")
        completed = subprocess.run(command, cwd=cwd, env=run_env, check=False)
        summary["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            summary["ok"] = False
            summary["errors"].append(f"Training command failed with returncode={completed.returncode}.")
    elif run and not summary["ok"]:
        summary["run_skipped"] = "launch_checks_failed"

    _write_launch_report(output_json, summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Preflight a Gaze-WAM training config, build the launch command, and optionally run it."
    )
    parser.add_argument("--config-name", default="train_gaze_wam_workspace")
    parser.add_argument("--task", default="gaze_wam")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--accelerate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accelerate-config", default="accelerate/8gpu-amp.yaml")
    parser.add_argument("--python-bin", default="py")
    parser.add_argument("--preflight-device", default="cpu")
    parser.add_argument("--preflight-checkpoint", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-zarr-validation", action="store_true")
    parser.add_argument("--skip-loss-smoke", action="store_true")
    parser.add_argument("--preflight-require-timestamps", action="store_true")
    parser.add_argument("--preflight-timestamp-max-delta", type=float, default=None)
    parser.add_argument("--preflight-timestamp-max-step", type=float, default=None)
    parser.add_argument("--preflight-fail-on-zarr-warning", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument(
        "--real-data-contract",
        choices=["main", "ablation"],
        default="main",
        help=(
            "Real-data readiness contract. 'main' enforces the fixed main-method ratio/dropout/"
            "heatmap-objective gates; 'ablation' keeps data/DINO/timestamp/geometry gates while "
            "allowing planned ablation variants."
        ),
    )
    parser.add_argument(
        "--data-onboarding-review-json",
        default=None,
        help=(
            "Optional JSON artifact from scripts/review_gaze_wam_data_onboarding.py. "
            "When supplied for --real-data, readiness checks that it matches the configured "
            "robot/open zarr paths and fixed policy-training contract."
        ),
    )
    parser.add_argument(
        "--require-data-onboarding-review",
        action="store_true",
        help="Make --data-onboarding-review-json launch-blocking for --real-data runs.",
    )
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = launch_gaze_wam_training(
        config_name=args.config_name,
        task=args.task,
        overrides=args.override,
        use_accelerate=args.accelerate,
        accelerate_config=args.accelerate_config,
        python_bin=args.python_bin,
        preflight_device=args.preflight_device,
        preflight_checkpoint=args.preflight_checkpoint,
        skip_preflight=args.skip_preflight,
        skip_zarr_validation=args.skip_zarr_validation,
        skip_loss_smoke=args.skip_loss_smoke,
        preflight_require_timestamps=args.preflight_require_timestamps,
        preflight_timestamp_max_delta=args.preflight_timestamp_max_delta,
        preflight_timestamp_max_step=args.preflight_timestamp_max_step,
        preflight_fail_on_zarr_warning=args.preflight_fail_on_zarr_warning,
        real_data=args.real_data,
        real_data_contract=args.real_data_contract,
        data_onboarding_review_json=args.data_onboarding_review_json,
        require_data_onboarding_review=args.require_data_onboarding_review,
        use_ema=args.use_ema,
        output_json=args.output_json,
        run=args.run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
