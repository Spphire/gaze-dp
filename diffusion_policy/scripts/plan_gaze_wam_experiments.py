import argparse
import csv
import json
import pathlib
import shlex
import sys
from typing import Dict, List, Optional, Sequence

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.common.gaze_wam_training_config import parse_gaze_wam_bool_field
from diffusion_policy.scripts.gaze_wam_provenance import add_provenance_contract


DEFAULT_FULL_VARIANTS = (
    ("robot_only_baseline", "train_gaze_wam_robot_only_workspace"),
    ("mixed_main", "train_gaze_wam_workspace"),
)

DEFAULT_DEBUG_VARIANTS = (
    ("robot_only_debug", "train_gaze_wam_robot_only_debug_workspace"),
    ("mixed_debug", "train_gaze_wam_debug_workspace"),
)

GAZE_DROPOUT_SWEEP = (0.0, 0.1, 0.2, 0.3)
OPEN_RATIO_SWEEP_FULL = (
    ("100_0", 64, 0),
    ("90_10", 58, 6),
    ("75_25", 48, 16),
    ("50_50", 32, 32),
)
OPEN_RATIO_SWEEP_DEBUG = (
    ("100_0", 10, 0),
    ("90_10", 9, 1),
    ("75_25", 3, 1),
    ("50_50", 2, 2),
)

CONTRACT_SUMMARY = {
    "image_shape": "3x256x256",
    "image_resize_mode": "stretch",
    "obs_encoder_model_name": "vit_base_patch16_dinov3",
    "image_tokens_per_frame": 256,
    "n_obs_steps": 2,
    "visual_token_count": 512,
    "action_horizon": 48,
    "action_dim": 10,
    "heatmap_num_tokens": 256,
    "heatmap_token_grid": "16x16",
}

JOB_PROVENANCE_FIELDS = [
    "provenance_contract_version",
    "provenance_contract_id",
    "robot_batch_size",
    "open_batch_size",
    "robot_ratio",
    "open_ratio",
    "training_stage",
    "batch_size_source",
    "requested_batch_size_source",
    "total_batch_size_per_process",
    "requested_robot_ratio",
    "requested_open_ratio",
    "gradient_accumulate_every",
    "num_processes",
    "mixed_precision",
    "distributed_type",
    "effective_robot_batch_size_per_optimizer_step",
    "effective_open_batch_size_per_optimizer_step",
    "effective_train_batch_size_per_optimizer_step",
    "robot_gaze_dropout_prob",
    "robot_heatmap_on_gaze_dropout",
    "cfg_scale",
    "use_block_attention_mask",
    "heatmap_objective",
    "n_obs_steps",
    "action_horizon",
    "action_dim",
    "heatmap_num_tokens",
    "heatmap_token_grid",
    "image_shape",
    "image_resize_mode",
    "robot_image_resize_mode",
    "open_image_resize_mode",
    "obs_encoder_model_name",
    "obs_encoder_pretrained",
    "obs_encoder_checkpoint_path",
    "obs_encoder_checkpoint_path_exists",
    "obs_encoder_checkpoint_path_is_file",
    "obs_encoder_cache_dir",
    "obs_encoder_cache_dir_exists",
    "obs_encoder_cache_dir_is_dir",
    "obs_encoder_local_weight_source_configured",
    "obs_encoder_local_weight_source_valid",
]


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_variant(spec: str) -> Dict[str, str]:
    if "=" in spec:
        name, config = spec.split("=", maxsplit=1)
    else:
        config = spec
        name = config
    name = name.strip()
    config = config.strip()
    if not name or not config:
        raise ValueError(f"Invalid variant spec {spec!r}; use name=config.")
    return {"name": name, "config": config}


def _format_float_token(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _add_variant(
    variants: List[Dict[str, object]],
    name: str,
    config: str,
    overrides: Optional[Sequence[str]] = None,
) -> None:
    variants.append({"name": name, "config": config, "overrides": list(overrides or [])})


def _default_variants(debug: bool) -> List[Dict[str, object]]:
    variant_pairs = DEFAULT_DEBUG_VARIANTS if debug else DEFAULT_FULL_VARIANTS
    return [
        {"name": name, "config": config, "overrides": []}
        for name, config in variant_pairs
    ]


def _sweep_variants(sweep: str, debug: bool) -> List[Dict[str, object]]:
    mixed_config = "train_gaze_wam_debug_workspace" if debug else "train_gaze_wam_workspace"
    robot_only_config = (
        "train_gaze_wam_robot_only_debug_workspace"
        if debug
        else "train_gaze_wam_robot_only_workspace"
    )
    suffix = "_debug" if debug else ""
    variants: List[Dict[str, object]] = []
    if sweep == "gaze_dropout":
        for prob in GAZE_DROPOUT_SWEEP:
            token = _format_float_token(prob)
            _add_variant(
                variants,
                name=f"gaze_dropout_{token}{suffix}",
                config=mixed_config,
                overrides=[f"task.robot_gaze_dropout_prob={prob}"],
            )
        return variants
    if sweep == "open_ratio":
        ratio_rows = OPEN_RATIO_SWEEP_DEBUG if debug else OPEN_RATIO_SWEEP_FULL
        for token, robot_batch, open_batch in ratio_rows:
            ratio_total = 64 if not debug else robot_batch + open_batch
            config = robot_only_config if open_batch == 0 else mixed_config
            _add_variant(
                variants,
                name=f"open_ratio_{token}{suffix}",
                config=config,
                overrides=[
                    "data_mixing.batch_size_source=ratio",
                    f"data_mixing.total_batch_size_per_process={ratio_total}",
                    (
                        "data_mixing.robot_ratio="
                        + str(float(robot_batch / ratio_total))
                    ),
                    (
                        "data_mixing.open_ratio="
                        + str(float(open_batch / ratio_total))
                    ),
                    f"robot_dataloader.batch_size={robot_batch}",
                    f"open_dataloader.batch_size={open_batch}",
                    f"val_robot_dataloader.batch_size={robot_batch}",
                    f"val_open_dataloader.batch_size={open_batch}",
                ],
            )
        return variants
    raise ValueError(f"Unknown sweep preset {sweep!r}.")


def _resolve_variants(
    variants: Optional[Sequence[str]],
    debug: bool,
    include_sweeps: Sequence[str],
) -> List[Dict[str, object]]:
    if variants is None:
        parsed_variants = _default_variants(debug=debug)
    else:
        parsed_variants = [
            {**_parse_variant(spec), "overrides": []}
            for spec in variants
        ]
    for sweep in include_sweeps:
        parsed_variants.extend(_sweep_variants(sweep=sweep, debug=debug))
    return parsed_variants


def _command_to_string(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _override_value(overrides: Sequence[str], key: str) -> Optional[str]:
    value = None
    for override in overrides:
        if "=" not in override:
            continue
        left, right = override.split("=", maxsplit=1)
        left = left.lstrip("+").strip()
        if left == key:
            value = right.strip()
    return value


def _override_int(overrides: Sequence[str], key: str, default: int) -> int:
    value = _override_value(overrides, key)
    return default if value is None else int(value)


def _override_float(overrides: Sequence[str], key: str, default: float) -> float:
    value = _override_value(overrides, key)
    return default if value is None else float(value)


def _override_bool(overrides: Sequence[str], key: str, default: bool) -> bool:
    value = _override_value(overrides, key)
    parsed, error = parse_gaze_wam_bool_field(key, value, default=default)
    if error is not None and value is None:
        return default
    if error is not None:
        raise ValueError(error)
    return bool(parsed)


def _override_str(overrides: Sequence[str], key: str, default: str) -> str:
    value = _override_value(overrides, key)
    return default if value is None else value


def _configured_path_exists(path: str) -> bool:
    return bool(path) and pathlib.Path(path).exists()


def _configured_path_is_file(path: str) -> bool:
    return bool(path) and pathlib.Path(path).is_file()


def _configured_path_is_dir(path: str) -> bool:
    return bool(path) and pathlib.Path(path).is_dir()


def _read_simple_yaml_scalars(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path:
        return values
    yaml_path = pathlib.Path(path)
    if not yaml_path.exists():
        return values
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", maxsplit=1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _accelerate_training_scale(
    *,
    use_accelerate: bool,
    accelerate_config: str,
    gradient_accumulate_every: int,
    robot_batch: int,
    open_batch: int,
) -> Dict[str, object]:
    if use_accelerate:
        cfg = _read_simple_yaml_scalars(accelerate_config)
        num_processes = int(cfg.get("num_processes", "1") or 1)
        mixed_precision = cfg.get("mixed_precision", "")
        distributed_type = cfg.get("distributed_type", "")
    else:
        num_processes = 1
        mixed_precision = "no"
        distributed_type = "NO"
    return {
        "gradient_accumulate_every": int(gradient_accumulate_every),
        "num_processes": int(num_processes),
        "mixed_precision": str(mixed_precision),
        "distributed_type": str(distributed_type),
        "effective_robot_batch_size_per_optimizer_step": (
            int(robot_batch) * int(num_processes) * int(gradient_accumulate_every)
        ),
        "effective_open_batch_size_per_optimizer_step": (
            int(open_batch) * int(num_processes) * int(gradient_accumulate_every)
        ),
        "effective_train_batch_size_per_optimizer_step": (
            (int(robot_batch) + int(open_batch))
            * int(num_processes)
            * int(gradient_accumulate_every)
        ),
    }


def _config_is_debug(config_name: str, plan_debug: bool) -> bool:
    return plan_debug or "_debug" in config_name


def _default_training_stage_for_config(config_name: str) -> str:
    name = str(config_name).lower()
    if "open_pretrain" in name:
        return "open_pretrain"
    if "robot_finetune" in name:
        return "robot_finetune"
    if "robot_only" in name:
        return "robot_only"
    if "open_only" in name:
        return "open_only"
    return "mixed_train"


def _job_provenance(
    config_name: str,
    plan_debug: bool,
    overrides: Sequence[str],
    use_accelerate: bool = False,
    accelerate_config: str = "accelerate/8gpu-amp.yaml",
) -> Dict[str, object]:
    is_debug = _config_is_debug(config_name=config_name, plan_debug=plan_debug)
    robot_batch = 3 if is_debug else 48
    open_batch = 1 if is_debug else 16
    robot_gaze_dropout_prob = 0.2
    robot_heatmap_on_gaze_dropout = True
    cfg_scale = 1.0
    use_block_attention_mask = True
    heatmap_objective = "dsnt_js"

    default_stage = _default_training_stage_for_config(config_name)
    if default_stage == "open_pretrain":
        robot_batch = 0
        open_batch = 4 if is_debug else 64
        robot_gaze_dropout_prob = 0.0
        robot_heatmap_on_gaze_dropout = False
    elif default_stage == "open_only":
        robot_batch = 0
        robot_gaze_dropout_prob = 0.0
        robot_heatmap_on_gaze_dropout = False
    elif default_stage == "robot_only":
        robot_batch = 3 if is_debug else 64
        open_batch = 0

    canonical_robot_batch = robot_batch
    canonical_open_batch = open_batch
    legacy_robot_batch = _override_int(
        overrides,
        "robot_dataloader.batch_size",
        canonical_robot_batch,
    )
    legacy_open_batch = _override_int(
        overrides,
        "open_dataloader.batch_size",
        canonical_open_batch,
    )
    default_total_batch = canonical_robot_batch + canonical_open_batch
    default_total_batch = _override_int(
        overrides,
        "data_mixing.total_batch_size_per_process",
        default_total_batch,
    )
    canonical_total = canonical_robot_batch + canonical_open_batch
    default_robot_ratio = (
        canonical_robot_batch / canonical_total if canonical_total > 0 else 0.0
    )
    default_open_ratio = (
        canonical_open_batch / canonical_total if canonical_total > 0 else 0.0
    )
    requested_robot_ratio = _override_float(
        overrides,
        "data_mixing.robot_ratio",
        default_robot_ratio,
    )
    requested_open_ratio = _override_float(
        overrides,
        "data_mixing.open_ratio",
        default_open_ratio,
    )
    ratio_robot_batch = int(default_total_batch * requested_robot_ratio + 0.5)
    ratio_open_batch = int(default_total_batch - ratio_robot_batch)
    config_defaults = _read_simple_yaml_scalars(
        str(ROOT_DIR / "diffusion_policy" / "config" / f"{config_name}.yaml")
    )
    requested_batch_size_source = _override_str(
        overrides,
        "data_mixing.batch_size_source",
        config_defaults.get("batch_size_source", "ratio"),
    ).strip().lower()
    if requested_batch_size_source == "ratio":
        resolved_batch_size_source = "ratio"
        robot_batch = ratio_robot_batch
        open_batch = ratio_open_batch
    elif requested_batch_size_source == "auto" and (
        ratio_robot_batch == legacy_robot_batch
        and ratio_open_batch == legacy_open_batch
    ):
        resolved_batch_size_source = "ratio"
        robot_batch = ratio_robot_batch
        open_batch = ratio_open_batch
    else:
        resolved_batch_size_source = "dataloader"
        robot_batch = legacy_robot_batch
        open_batch = legacy_open_batch
    gradient_accumulate_every = _override_int(
        overrides,
        "training.gradient_accumulate_every",
        1,
    )
    total_batch = robot_batch + open_batch
    training_scale = _accelerate_training_scale(
        use_accelerate=use_accelerate,
        accelerate_config=accelerate_config,
        gradient_accumulate_every=gradient_accumulate_every,
        robot_batch=robot_batch,
        open_batch=open_batch,
    )

    robot_gaze_dropout_prob = _override_float(
        overrides,
        "task.robot_gaze_dropout_prob",
        robot_gaze_dropout_prob,
    )
    robot_heatmap_on_gaze_dropout = _override_bool(
        overrides,
        "task.robot_heatmap_on_gaze_dropout",
        robot_heatmap_on_gaze_dropout,
    )
    cfg_scale = _override_float(overrides, "policy.cfg_scale", cfg_scale)
    use_block_attention_mask = _override_bool(
        overrides,
        "policy.use_block_attention_mask",
        use_block_attention_mask,
    )
    heatmap_objective = _override_str(
        overrides,
        "policy.heatmap_objective",
        heatmap_objective,
    )

    n_obs_steps = _override_int(overrides, "task.n_obs_steps", int(CONTRACT_SUMMARY["n_obs_steps"]))
    action_horizon = _override_int(
        overrides,
        "task.action_horizon",
        int(CONTRACT_SUMMARY["action_horizon"]),
    )
    action_dim = _override_int(overrides, "task.action_dim", int(CONTRACT_SUMMARY["action_dim"]))
    heatmap_num_tokens = _override_int(
        overrides,
        "task.heatmap_num_tokens",
        int(CONTRACT_SUMMARY["heatmap_num_tokens"]),
    )
    image_resize_mode = _override_str(
        overrides,
        "task.image_resize_mode",
        str(CONTRACT_SUMMARY["image_resize_mode"]),
    )
    robot_image_resize_mode = _override_str(
        overrides,
        "task.robot_dataset.image_resize_mode",
        image_resize_mode,
    )
    open_image_resize_mode = _override_str(
        overrides,
        "task.open_dataset.image_resize_mode",
        image_resize_mode,
    )
    obs_encoder_pretrained = _override_bool(
        overrides,
        "policy.obs_encoder.pretrained",
        not is_debug,
    )
    obs_encoder_checkpoint_path = _override_str(
        overrides,
        "policy.obs_encoder.checkpoint_path",
        "",
    ).strip()
    obs_encoder_cache_dir = _override_str(
        overrides,
        "policy.obs_encoder.cache_dir",
        "",
    ).strip()
    obs_encoder_checkpoint_path_exists = _configured_path_exists(obs_encoder_checkpoint_path)
    obs_encoder_checkpoint_path_is_file = _configured_path_is_file(obs_encoder_checkpoint_path)
    obs_encoder_cache_dir_exists = _configured_path_exists(obs_encoder_cache_dir)
    obs_encoder_cache_dir_is_dir = _configured_path_is_dir(obs_encoder_cache_dir)
    obs_encoder_local_weight_source_configured = bool(
        obs_encoder_checkpoint_path or obs_encoder_cache_dir
    )
    obs_encoder_local_weight_source_valid = (
        (not obs_encoder_checkpoint_path or obs_encoder_checkpoint_path_is_file)
        and (not obs_encoder_cache_dir or obs_encoder_cache_dir_is_dir)
    )

    training_stage = _override_str(
        overrides,
        "training.stage",
        default_stage,
    )

    return add_provenance_contract({
        "robot_batch_size": robot_batch,
        "open_batch_size": open_batch,
        "robot_ratio": float(robot_batch / total_batch) if total_batch > 0 else 0.0,
        "open_ratio": float(open_batch / total_batch) if total_batch > 0 else 0.0,
        "training_stage": training_stage,
        "batch_size_source": resolved_batch_size_source,
        "requested_batch_size_source": requested_batch_size_source,
        "total_batch_size_per_process": int(total_batch),
        "requested_robot_ratio": float(requested_robot_ratio),
        "requested_open_ratio": float(requested_open_ratio),
        **training_scale,
        "robot_gaze_dropout_prob": robot_gaze_dropout_prob,
        "robot_heatmap_on_gaze_dropout": robot_heatmap_on_gaze_dropout,
        "cfg_scale": cfg_scale,
        "use_block_attention_mask": use_block_attention_mask,
        "heatmap_objective": heatmap_objective,
        "n_obs_steps": n_obs_steps,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "heatmap_num_tokens": heatmap_num_tokens,
        "heatmap_token_grid": str(CONTRACT_SUMMARY["heatmap_token_grid"]),
        "image_shape": str(CONTRACT_SUMMARY["image_shape"]),
        "image_resize_mode": image_resize_mode,
        "robot_image_resize_mode": robot_image_resize_mode,
        "open_image_resize_mode": open_image_resize_mode,
        "obs_encoder_model_name": str(CONTRACT_SUMMARY["obs_encoder_model_name"]),
        "obs_encoder_pretrained": obs_encoder_pretrained,
        "obs_encoder_checkpoint_path": obs_encoder_checkpoint_path,
        "obs_encoder_checkpoint_path_exists": obs_encoder_checkpoint_path_exists,
        "obs_encoder_checkpoint_path_is_file": obs_encoder_checkpoint_path_is_file,
        "obs_encoder_cache_dir": obs_encoder_cache_dir,
        "obs_encoder_cache_dir_exists": obs_encoder_cache_dir_exists,
        "obs_encoder_cache_dir_is_dir": obs_encoder_cache_dir_is_dir,
        "obs_encoder_local_weight_source_configured": obs_encoder_local_weight_source_configured,
        "obs_encoder_local_weight_source_valid": obs_encoder_local_weight_source_valid,
    })


def _train_command(
    config_name: str,
    task: str,
    use_accelerate: bool,
    accelerate_config: str,
    overrides: Sequence[str],
    variant_overrides: Sequence[str] = (),
    train_via_launcher: bool = False,
    real_data_launch: bool = False,
    real_data_contract: str = "ablation",
    launcher_report_path: str = "",
    data_onboarding_review_path: str = "",
    require_data_onboarding_review: bool = False,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    train_fail_on_zarr_warning: bool = False,
) -> List[str]:
    if train_via_launcher or real_data_launch:
        command = [
            "py",
            "scripts/launch_gaze_wam_training.py",
            "--config-name",
            config_name,
            "--task",
            task,
        ]
        if use_accelerate:
            command.extend(["--accelerate", "--accelerate-config", accelerate_config])
        else:
            command.append("--no-accelerate")
        for override in list(overrides) + list(variant_overrides):
            command.extend(["--override", override])
        if require_timestamps or real_data_launch:
            command.append("--preflight-require-timestamps")
        if timestamp_max_delta is not None:
            command.extend(["--preflight-timestamp-max-delta", str(float(timestamp_max_delta))])
        if timestamp_max_step is not None:
            command.extend(["--preflight-timestamp-max-step", str(float(timestamp_max_step))])
        if train_fail_on_zarr_warning or real_data_launch:
            command.append("--preflight-fail-on-zarr-warning")
        if real_data_launch:
            command.extend(["--real-data", "--real-data-contract", real_data_contract])
        if data_onboarding_review_path:
            command.extend(["--data-onboarding-review-json", data_onboarding_review_path])
        if require_data_onboarding_review:
            command.append("--require-data-onboarding-review")
        if launcher_report_path:
            command.extend(["--output-json", launcher_report_path])
        command.append("--run")
        return command

    if use_accelerate:
        command = [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "train.py",
            "--config-name",
            config_name,
            f"task={task}",
        ]
    else:
        command = ["py", "train.py", "--config-name", config_name, f"task={task}"]
    command.extend(overrides)
    command.extend(variant_overrides)
    return command


def _launcher_report_path(template: str, name: str, config: str) -> str:
    return template.format(name=name, config=config) if template else ""


def _data_onboarding_review_path(template: str, name: str, config: str) -> str:
    return template.format(name=name, config=config) if template else ""


def _eval_variant_spec(name: str, config: str, checkpoint_template: str) -> str:
    checkpoint = checkpoint_template.format(name=name, config=config)
    return f"{name}={config}:{checkpoint}" if checkpoint else f"{name}={config}"


def build_gaze_wam_experiment_plan(
    variants: Optional[Sequence[str]] = None,
    debug: bool = False,
    task: str = "gaze_wam",
    use_accelerate: bool = False,
    accelerate_config: str = "accelerate/8gpu-amp.yaml",
    train_via_launcher: bool = False,
    real_data_launch: bool = False,
    real_data_contract: str = "ablation",
    launcher_report_template: str = "data/outputs/{config}/gaze_wam_launch_report.json",
    data_onboarding_review_template: str = "",
    require_data_onboarding_review: bool = False,
    train_fail_on_zarr_warning: bool = False,
    include_sweeps: Optional[Sequence[str]] = None,
    train_overrides: Optional[Sequence[str]] = None,
    eval_overrides: Optional[Sequence[str]] = None,
    checkpoint_template: str = "data/outputs/{config}/checkpoints/latest.ckpt",
    eval_device: str = "cpu",
    eval_batch_size: int = 16,
    eval_max_batches: Optional[int] = None,
    eval_cfg_scale: Optional[float] = None,
    eval_sources: Sequence[str] = ("robot", "open"),
    skip_sampling: bool = False,
    skip_heatmap: bool = False,
    skip_gdr: bool = False,
    validate_zarr: bool = True,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    metrics_json: str = "data/outputs/gaze_wam_ablation_metrics.json",
    metrics_csv: str = "data/outputs/gaze_wam_ablation_metrics.csv",
) -> Dict[str, object]:
    train_overrides = list(train_overrides or [])
    eval_overrides = list(eval_overrides or [])
    if include_sweeps is None:
        # A default plan should expose the source-mixture ablation explicitly;
        # custom variant lists remain single-job unless the caller asks for a sweep.
        include_sweeps = ["open_ratio"] if variants is None else []
    else:
        include_sweeps = list(include_sweeps)
    parsed_variants = _resolve_variants(
        variants=variants,
        debug=debug,
        include_sweeps=include_sweeps,
    )
    eval_sources = list(eval_sources)

    train_jobs = []
    for variant in parsed_variants:
        all_train_overrides = list(train_overrides) + list(variant.get("overrides", []))
        provenance = _job_provenance(
            config_name=variant["config"],
            plan_debug=debug,
            overrides=all_train_overrides,
            use_accelerate=use_accelerate,
            accelerate_config=accelerate_config,
        )
        command = _train_command(
            config_name=variant["config"],
            task=task,
            use_accelerate=use_accelerate,
            accelerate_config=accelerate_config,
            overrides=train_overrides,
            variant_overrides=variant.get("overrides", []),
            train_via_launcher=train_via_launcher,
            real_data_launch=real_data_launch,
            real_data_contract=real_data_contract,
            launcher_report_path=_launcher_report_path(
                launcher_report_template,
                name=variant["name"],
                config=variant["config"],
            ),
            data_onboarding_review_path=_data_onboarding_review_path(
                data_onboarding_review_template,
                name=variant["name"],
                config=variant["config"],
            ),
            require_data_onboarding_review=require_data_onboarding_review,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
            train_fail_on_zarr_warning=train_fail_on_zarr_warning,
        )
        train_jobs.append(
            {
                "variant": variant["name"],
                "config_name": variant["config"],
                "overrides": list(variant.get("overrides", [])),
                "command": command,
                "command_str": _command_to_string(command),
                **provenance,
            }
        )

    eval_command = [
        "py",
        "scripts/compare_gaze_wam_ablation_metrics.py",
    ]
    for variant in parsed_variants:
        eval_command.extend(
            [
                "--variant",
                _eval_variant_spec(variant["name"], variant["config"], checkpoint_template),
            ]
        )
    for override in eval_overrides:
        eval_command.extend(["--override", override])
    for variant in parsed_variants:
        for override in variant.get("overrides", []):
            eval_command.extend(["--variant-override", variant["name"], override])
    eval_command.extend(
        [
            "--device",
            eval_device,
            "--batch-size",
            str(int(eval_batch_size)),
            "--sources",
            ",".join(eval_sources),
            "--output-json",
            metrics_json,
            "--output-csv",
            metrics_csv,
        ]
    )
    if eval_max_batches is not None:
        eval_command.extend(["--max-batches", str(int(eval_max_batches))])
    if eval_cfg_scale is not None:
        eval_command.extend(["--cfg-scale", str(float(eval_cfg_scale))])
    if skip_sampling:
        eval_command.append("--skip-sampling")
    if skip_heatmap:
        eval_command.append("--skip-heatmap")
    if skip_gdr:
        eval_command.append("--skip-gdr")
    eval_command.append("--validate-zarr" if validate_zarr else "--no-validate-zarr")
    if timestamp_key is not None:
        eval_command.extend(["--timestamp-key", timestamp_key])
    if require_timestamps:
        eval_command.append("--require-timestamps")
    if timestamp_max_delta is not None:
        eval_command.extend(["--timestamp-max-delta", str(float(timestamp_max_delta))])
    if timestamp_max_step is not None:
        eval_command.extend(["--timestamp-max-step", str(float(timestamp_max_step))])

    eval_variant_jobs = []
    for variant in parsed_variants:
        all_train_overrides = list(train_overrides) + list(variant.get("overrides", []))
        provenance = _job_provenance(
            config_name=variant["config"],
            plan_debug=debug,
            overrides=all_train_overrides,
            use_accelerate=use_accelerate,
            accelerate_config=accelerate_config,
        )
        eval_variant_jobs.append(
            {
                "variant": variant["name"],
                "config_name": variant["config"],
                "overrides": list(variant.get("overrides", [])),
                "checkpoint": checkpoint_template.format(
                    name=variant["name"],
                    config=variant["config"],
                )
                if checkpoint_template
                else "",
                **provenance,
            }
        )

    return {
        "mode": "debug" if debug else "full",
        "task": task,
        "use_accelerate": bool(use_accelerate),
        "accelerate_config": accelerate_config if use_accelerate else "",
        "train_launch": {
            "train_via_launcher": bool(train_via_launcher or real_data_launch),
            "real_data_launch": bool(real_data_launch),
            "real_data_contract": real_data_contract,
            "launcher_report_template": launcher_report_template,
            "data_onboarding_review_template": data_onboarding_review_template,
            "require_data_onboarding_review": bool(require_data_onboarding_review),
            "train_fail_on_zarr_warning": bool(train_fail_on_zarr_warning),
        },
        "include_sweeps": include_sweeps,
        "variants": parsed_variants,
        "train_overrides": train_overrides,
        "eval_overrides": eval_overrides,
        "eval_validation": {
            "validate_zarr": bool(validate_zarr),
            "timestamp_key": timestamp_key,
            "require_timestamps": bool(require_timestamps),
            "timestamp_max_delta": timestamp_max_delta,
            "timestamp_max_step": timestamp_max_step,
        },
        "contract_summary": dict(CONTRACT_SUMMARY),
        "eval_cfg_scale": eval_cfg_scale,
        "train_jobs": train_jobs,
        "eval_job": {
            "command": eval_command,
            "command_str": _command_to_string(eval_command),
            "metrics_json": metrics_json,
            "metrics_csv": metrics_csv,
            "variant_jobs": eval_variant_jobs,
        },
    }


def write_plan_csv(plan: Dict[str, object], output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in plan["train_jobs"]:
        rows.append(
            {
                "job_type": "train",
                "variant": job["variant"],
                "config_name": job["config_name"],
                "checkpoint": "",
                **{field: job.get(field, "") for field in JOB_PROVENANCE_FIELDS},
                "command": job["command_str"],
            }
        )
    eval_job = plan["eval_job"]
    for index, job in enumerate(eval_job.get("variant_jobs", [])):
        rows.append(
            {
                "job_type": "eval",
                "variant": job["variant"],
                "config_name": job["config_name"],
                "checkpoint": job.get("checkpoint", ""),
                **{field: job.get(field, "") for field in JOB_PROVENANCE_FIELDS},
                "command": eval_job["command_str"] if index == 0 else "",
            }
        )
    if not eval_job.get("variant_jobs"):
        rows.append(
            {
                "job_type": "eval",
                "variant": "all",
                "config_name": "",
                "checkpoint": "",
                **{field: "" for field in JOB_PROVENANCE_FIELDS},
                "command": eval_job["command_str"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_type",
                "variant",
                "config_name",
                "checkpoint",
                *JOB_PROVENANCE_FIELDS,
                "command",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plan_script(plan: Dict[str, object], output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated Gaze-WAM experiment plan.",
    ]
    for job in plan["train_jobs"]:
        lines.extend(
            [
                "",
                f"echo '[Gaze-WAM] training {job['variant']}'",
                job["command_str"],
            ]
        )
    lines.extend(
        [
            "",
            "echo '[Gaze-WAM] evaluating ablations'",
            plan["eval_job"]["command_str"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate a reproducible Gaze-WAM training/evaluation experiment plan."
    )
    parser.add_argument("--variant", action="append", default=None, help="Variant spec name=config.")
    parser.add_argument("--debug", action="store_true", help="Use debug smoke configs by default.")
    parser.add_argument("--task", default="gaze_wam")
    parser.add_argument("--use-accelerate", action="store_true")
    parser.add_argument("--accelerate-config", default="accelerate/8gpu-amp.yaml")
    parser.add_argument(
        "--train-via-launcher",
        action="store_true",
        help="Generate training commands through scripts/launch_gaze_wam_training.py --run.",
    )
    parser.add_argument(
        "--real-data-launch",
        action="store_true",
        help=(
            "Generate training commands through the guarded launcher with --real-data. "
            "This implies --train-via-launcher for train commands."
        ),
    )
    parser.add_argument(
        "--real-data-contract",
        choices=["main", "ablation"],
        default="ablation",
        help=(
            "Real-data readiness contract for generated launcher train commands. "
            "Use 'ablation' for multi-variant experiment plans."
        ),
    )
    parser.add_argument(
        "--launcher-report-template",
        default="data/outputs/{config}/gaze_wam_launch_report.json",
        help="Python format string with {name} and {config}; empty string omits launcher reports.",
    )
    parser.add_argument(
        "--data-onboarding-review-template",
        default="",
        help=(
            "Python format string with {name} and {config} for JSON artifacts produced by "
            "scripts/review_gaze_wam_data_onboarding.py. Empty string omits the launcher "
            "--data-onboarding-review-json flag."
        ),
    )
    parser.add_argument(
        "--require-data-onboarding-review",
        action="store_true",
        help="Add --require-data-onboarding-review to generated launcher train commands.",
    )
    parser.add_argument(
        "--train-fail-on-zarr-warning",
        action="store_true",
        help="Add --preflight-fail-on-zarr-warning to launcher train commands.",
    )
    parser.add_argument(
        "--include-sweep",
        action="append",
        choices=["gaze_dropout", "open_ratio"],
        default=[],
        help="Append a preset ablation sweep to the default or custom variants.",
    )
    parser.add_argument("--train-override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument(
        "--checkpoint-template",
        default="data/outputs/{config}/checkpoints/latest.ckpt",
        help="Python format string with {name} and {config}. Empty string omits checkpoints.",
    )
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument(
        "--eval-cfg-scale",
        type=float,
        default=None,
        help="Optional global CFG scale override for the generated eval command.",
    )
    parser.add_argument("--eval-sources", default="robot,open")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-gdr", action="store_true")
    parser.add_argument("--validate-zarr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--metrics-json", default="data/outputs/gaze_wam_ablation_metrics.json")
    parser.add_argument("--metrics-csv", default="data/outputs/gaze_wam_ablation_metrics.csv")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-script", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    plan = build_gaze_wam_experiment_plan(
        variants=args.variant,
        debug=args.debug,
        task=args.task,
        use_accelerate=args.use_accelerate,
        accelerate_config=args.accelerate_config,
        train_via_launcher=args.train_via_launcher,
        real_data_launch=args.real_data_launch,
        real_data_contract=args.real_data_contract,
        launcher_report_template=args.launcher_report_template,
        data_onboarding_review_template=args.data_onboarding_review_template,
        require_data_onboarding_review=args.require_data_onboarding_review,
        train_fail_on_zarr_warning=args.train_fail_on_zarr_warning,
        include_sweeps=args.include_sweep,
        train_overrides=args.train_override,
        eval_overrides=args.eval_override,
        checkpoint_template=args.checkpoint_template,
        eval_device=args.eval_device,
        eval_batch_size=args.eval_batch_size,
        eval_max_batches=args.eval_max_batches,
        eval_cfg_scale=args.eval_cfg_scale,
        eval_sources=_split_csv(args.eval_sources),
        skip_sampling=args.skip_sampling,
        skip_heatmap=args.skip_heatmap,
        skip_gdr=args.skip_gdr,
        validate_zarr=args.validate_zarr,
        timestamp_key=args.timestamp_key,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        metrics_json=args.metrics_json,
        metrics_csv=args.metrics_csv,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_plan_csv(plan, args.output_csv)
    if args.output_script is not None:
        write_plan_script(plan, args.output_script)
    return plan


if __name__ == "__main__":
    main()
