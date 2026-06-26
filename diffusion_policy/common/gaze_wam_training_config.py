import ast
import math


GAZE_WAM_REQUIRED_LOSS_ROUTING_VALIDATION_FLAGS = (
    "open_rows_must_not_have_action",
    "open_rows_must_have_heatmap",
    "open_rows_must_use_mask_token",
    "robot_rows_must_have_action",
    "inactive_action_rows_must_be_zero_placeholders",
    "inactive_heatmap_rows_must_be_zero_placeholders",
    "inactive_gaze_rows_must_be_zero_placeholders",
    "inactive_optional_metadata_rows_must_be_zero_placeholders",
    "robot_real_gaze_rows_must_not_have_heatmap_loss",
    "is_gaze_condition_dropped_equals_not_use_gaze_condition",
)


def gaze_wam_required_loss_routing_validation_flags() -> tuple:
    """Return the fixed loss-routing validation flags required by readiness gates."""
    return tuple(GAZE_WAM_REQUIRED_LOSS_ROUTING_VALIDATION_FLAGS)


def gaze_wam_loss_routing_validation_guardrails_ok(loss_routing_contract) -> bool:
    """Check whether a loss-routing contract preserves the required validation guardrails."""
    if not isinstance(loss_routing_contract, dict):
        return False
    validation = loss_routing_contract.get("validation")
    if not isinstance(validation, dict):
        return False
    return all(
        validation.get(key) is True
        for key in GAZE_WAM_REQUIRED_LOSS_ROUTING_VALIDATION_FLAGS
    )


def gaze_wam_action_normalizer_contract(
    action_dim: int = 10,
    camera_key: str = "camera0_rgb",
    heatmap_only: bool = False,
) -> dict:
    """Describe the fixed Gaze-WAM normalizer provenance for review artifacts."""
    action_dim = normalize_gaze_wam_positive_int_field("action_dim", action_dim)
    camera_key = str(camera_key)
    if heatmap_only:
        return {
            "source": "identity_action_placeholder_for_heatmap_only",
            "action_normalizer_source": "identity_placeholder_not_fitted_from_actions",
            "image_normalizer_source": "identity_image_normalizer",
            "normalizer_keys": [camera_key, "action"],
            "camera_key": camera_key,
            "action_key": "action",
            "action_dim": action_dim,
            "action_representation": "unused_zero_placeholder_for_open_only_heatmap_training",
            "robot_zarr_action_storage": "not_required_for_open_only_heatmap_training",
            "excludes_open_source_dummy_actions": True,
            "open_source_get_normalizer_allowed": False,
            "open_source_actions_are_zero_placeholders": True,
        }
    return {
        "source": "robot_dataset_relative_actions_only",
        "action_normalizer_source": "GazeWamRobotDataset.get_all_actions",
        "image_normalizer_source": "identity_image_normalizer",
        "normalizer_keys": [camera_key, "action"],
        "camera_key": camera_key,
        "action_key": "action",
        "action_dim": action_dim,
        "action_representation": "relative_tcp_from_latest_observed_absolute_base",
        "robot_zarr_action_storage": "absolute_tcp_trajectory",
        "excludes_open_source_dummy_actions": True,
        "open_source_get_normalizer_allowed": False,
        "open_source_actions_are_zero_placeholders": True,
    }


def gaze_wam_data_stream_contract(
    robot_dataset_path: str,
    open_dataset_path: str,
    robot_dataset_class: str,
    open_dataset_class: str,
    robot_batch_size: int,
    open_batch_size: int,
) -> dict:
    """Describe the fixed two-source online mixing contract for review artifacts."""
    robot_batch_size = normalize_gaze_wam_nonnegative_int_field(
        "robot_batch_size",
        robot_batch_size,
    )
    open_batch_size = normalize_gaze_wam_nonnegative_int_field(
        "open_batch_size",
        open_batch_size,
    )
    robot_dataset_path = str(robot_dataset_path)
    open_dataset_path = str(open_dataset_path)
    robot_dataset_class = str(robot_dataset_class)
    open_dataset_class = str(open_dataset_class)
    expected_robot_class = "diffusion_policy.dataset.gaze_wam_dataset.GazeWamRobotDataset"
    expected_open_class = "diffusion_policy.dataset.gaze_wam_dataset.GazeWamOpenDataset"
    total_batch_size = robot_batch_size + open_batch_size
    return {
        "source": "two_zarr_two_dataset_online_mixed_batch",
        "separate_zarr_sources": robot_dataset_path != open_dataset_path,
        "offline_merged_zarr": False,
        "robot": {
            "dataset_path": robot_dataset_path,
            "dataset_class": robot_dataset_class,
            "expected_dataset_class": expected_robot_class,
            "dataset_class_matches_expected": robot_dataset_class == expected_robot_class,
            "dataloader": "robot_dataloader",
            "batch_size_per_process": robot_batch_size,
            "enabled": robot_batch_size > 0,
            "drives_epoch": robot_batch_size > 0,
            "has_action": True,
            "normalizer_source": "robot_dataset_relative_actions_only",
        },
        "open": {
            "dataset_path": open_dataset_path,
            "dataset_class": open_dataset_class,
            "expected_dataset_class": expected_open_class,
            "dataset_class_matches_expected": open_dataset_class == expected_open_class,
            "dataloader": "open_dataloader",
            "batch_size_per_process": open_batch_size,
            "enabled": open_batch_size > 0,
            "drives_epoch": robot_batch_size <= 0 and open_batch_size > 0,
            "has_action": False,
            "action_values_used_for_loss": False,
        },
        "mixing": {
            "builder": (
                "diffusion_policy.dataset.gaze_wam_mixing."
                "build_gaze_wam_mixed_batch"
            ),
            "mode": "online_per_step_concat_after_fetch",
            "ratio_source": "robot_dataloader.batch_size/open_dataloader.batch_size",
            "shuffle_after_concat": True,
            "primary_epoch_driver": (
                "robot_dataloader"
                if robot_batch_size > 0
                else "open_dataloader"
                if open_batch_size > 0
                else "none"
            ),
            "open_iterator_policy": "restart_on_exhaustion",
            "robot_ratio_per_process": (
                robot_batch_size / total_batch_size if total_batch_size > 0 else 0.0
            ),
            "open_ratio_per_process": (
                open_batch_size / total_batch_size if total_batch_size > 0 else 0.0
            ),
        },
    }


def _parse_integer_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        signless = text[1:] if text[0] in ("+", "-") else text
        if signless.isdigit():
            return int(text)
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if value == parsed:
        return parsed
    return None


def _parse_int_field(name: str, value, default: int = 0):
    parsed = _parse_integer_value(value)
    if parsed is None:
        return default, f"{name} must be an integer, got {value!r}."
    return parsed, None


def _parse_optional_int_field(name: str, value):
    if value is None:
        return None, None
    parsed = _parse_integer_value(value)
    if parsed is None:
        return value, f"{name} must be an integer or null, got {value!r}."
    return parsed, None


def _parse_float_field(name: str, value, default: float = 0.0):
    if isinstance(value, bool):
        return default, f"{name} must be a number, got {value!r}."
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default, f"{name} must be a number, got {value!r}."
    if not math.isfinite(parsed):
        return default, f"{name} must be finite, got {value!r}."
    return parsed, None


def _parse_bool_field(name: str, value, default: bool = False):
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y", "on"):
            return True, None
        if normalized in ("false", "0", "no", "n", "off"):
            return False, None
    return default, f"{name} must be a boolean, got {value!r}."


def parse_gaze_wam_bool_field(name: str, value, default: bool = False):
    """Parse bool-like config values without relying on Python string truthiness."""
    return _parse_bool_field(name, value, default=default)


def normalize_gaze_wam_bool_field(name: str, value, default: bool = False) -> bool:
    """Return a native bool or raise a config-specific error."""
    parsed, error = parse_gaze_wam_bool_field(name, value, default=default)
    if error is not None and value is None:
        return bool(default)
    if error is not None:
        raise ValueError(error)
    return bool(parsed)


def normalize_gaze_wam_nonnegative_float_field(
    name: str,
    value,
    default: float = 0.0,
) -> float:
    """Return a finite non-negative float without treating booleans as numbers."""
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a finite non-negative float, got {value!r}."
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite non-negative float, got {value!r}."
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"{name} must be a finite non-negative float, got {value!r}."
        )
    return parsed


def normalize_gaze_wam_positive_float_field(
    name: str,
    value,
    default: float = 1.0,
) -> float:
    """Return a finite positive float without treating booleans as numbers."""
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive float, got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}.")
    return parsed


def normalize_gaze_wam_unit_interval_float_field(
    name: str,
    value,
    default: float = 0.0,
    include_one: bool = False,
) -> float:
    """Return a finite float in [0, 1) by default without accepting booleans."""
    interval = "[0, 1]" if include_one else "[0, 1)"
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be in {interval}, got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be in {interval}, got {value!r}.") from exc
    upper_ok = parsed <= 1.0 if include_one else parsed < 1.0
    if not math.isfinite(parsed) or parsed < 0.0 or not upper_ok:
        raise ValueError(f"{name} must be in {interval}, got {value!r}.")
    return parsed


def normalize_gaze_wam_positive_int_field(name: str, value, default: int = 1) -> int:
    """Return a positive integer without silently truncating bools/fractions."""
    if value is None:
        value = default
    parsed = _parse_integer_value(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(parsed)


def normalize_gaze_wam_nonnegative_int_field(name: str, value, default: int = 0) -> int:
    """Return a non-negative integer without silently truncating bools/fractions."""
    if value is None:
        value = default
    parsed = _parse_integer_value(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(parsed)


def normalize_gaze_wam_positive_int_sequence(
    name: str,
    value,
    length: int = None,
) -> list:
    """Return a positive-int list for geometry fields without truncating bad entries."""
    original_value = value
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a sequence of positive integers, got {original_value!r}."
            ) from exc
    if value is None or isinstance(value, (bool, int, float)):
        raise ValueError(
            f"{name} must be a sequence of positive integers, got {original_value!r}."
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a sequence of positive integers, got {original_value!r}."
        ) from exc
    if length is not None and len(items) != int(length):
        raise ValueError(
            f"{name} must contain {int(length)} positive integers, got {original_value!r}."
        )
    return [
        normalize_gaze_wam_positive_int_field(f"{name}[{idx}]", item)
        for idx, item in enumerate(items)
    ]


def _normalize_gaze_wam_early_bool_config(cfg):
    """Normalize bool fields used before or outside the shared training config gate."""
    fields = (
        (cfg.training, "use_ema", True, "training.use_ema"),
        (cfg.training, "resume", False, "training.resume"),
        (cfg.training, "debug", False, "training.debug"),
        (cfg.training, "require_amp", True, "training.require_amp"),
        (cfg.training, "freeze_encoder", False, "training.freeze_encoder"),
        (
            cfg.training,
            "save_val_heatmap_preview",
            False,
            "training.save_val_heatmap_preview",
        ),
        (cfg.checkpoint, "save_last_ckpt", True, "checkpoint.save_last_ckpt"),
        (cfg.checkpoint, "save_last_snapshot", False, "checkpoint.save_last_snapshot"),
        (cfg.policy, "use_block_attention_mask", True, "policy.use_block_attention_mask"),
        (cfg.policy, "use_frame_embedding", False, "policy.use_frame_embedding"),
        (cfg.policy.obs_encoder, "pretrained", False, "policy.obs_encoder.pretrained"),
        (cfg.policy.obs_encoder, "frozen", False, "policy.obs_encoder.frozen"),
        (
            cfg.policy.obs_encoder,
            "use_group_norm",
            False,
            "policy.obs_encoder.use_group_norm",
        ),
        (
            cfg.policy.obs_encoder,
            "share_rgb_model",
            False,
            "policy.obs_encoder.share_rgb_model",
        ),
    )
    for container, key, default, name in fields:
        if key not in container:
            continue
        container[key] = normalize_gaze_wam_bool_field(
            name,
            container.get(key, default),
            default=default,
        )
    return cfg


def validate_gaze_wam_task_routing_config(cfg):
    """Parse source-routing task fields that control mixed-batch supervision."""
    task = cfg.task
    errors = []

    raw_robot_gaze_dropout_prob = task.get("robot_gaze_dropout_prob", 0.0)
    try:
        robot_gaze_dropout_prob = normalize_gaze_wam_unit_interval_float_field(
            "task.robot_gaze_dropout_prob",
            raw_robot_gaze_dropout_prob,
            include_one=True,
        )
    except ValueError as exc:
        errors.append(str(exc))
        robot_gaze_dropout_prob, _ = _parse_float_field(
            "task.robot_gaze_dropout_prob",
            raw_robot_gaze_dropout_prob,
        )

    try:
        robot_heatmap_on_gaze_dropout = normalize_gaze_wam_bool_field(
            "task.robot_heatmap_on_gaze_dropout",
            task.get("robot_heatmap_on_gaze_dropout", True),
            default=True,
        )
    except ValueError as exc:
        errors.append(str(exc))
        robot_heatmap_on_gaze_dropout = True

    return {
        "robot_gaze_dropout_prob": robot_gaze_dropout_prob,
        "robot_heatmap_on_gaze_dropout": robot_heatmap_on_gaze_dropout,
        "errors": errors,
        "valid": len(errors) == 0,
    }


def _normalize_gaze_wam_task_routing_config(cfg, task_routing_config):
    """Apply parsed mixed-batch routing values so downstream code sees native types."""
    if not task_routing_config["valid"]:
        raise ValueError("Cannot normalize an invalid Gaze-WAM task routing config.")
    cfg.task.robot_gaze_dropout_prob = float(
        task_routing_config["robot_gaze_dropout_prob"]
    )
    cfg.task.robot_heatmap_on_gaze_dropout = bool(
        task_routing_config["robot_heatmap_on_gaze_dropout"]
    )
    return cfg


def _positive_int_error(name: str, value, allow_none: bool = False):
    if value is None and allow_none:
        return None
    parsed = _parse_integer_value(value)
    if parsed is None:
        return f"{name} must be a positive integer, got {value!r}."
    if parsed <= 0:
        return f"{name} must be a positive integer, got {parsed}."
    return None


def _nonnegative_int_error(name: str, value, allow_none: bool = False):
    if value is None and allow_none:
        return None
    parsed = _parse_integer_value(value)
    if parsed is None:
        return f"{name} must be a non-negative integer, got {value!r}."
    if parsed < 0:
        return f"{name} must be a non-negative integer, got {parsed}."
    return None


def _parse_dataloader_runtime_config(name: str, dataloader_cfg):
    num_workers, num_workers_error = _parse_int_field(
        f"{name}.num_workers",
        dataloader_cfg.get("num_workers", 0),
    )
    pin_memory, pin_memory_error = _parse_bool_field(
        f"{name}.pin_memory",
        dataloader_cfg.get("pin_memory", False),
    )
    persistent_workers, persistent_workers_error = _parse_bool_field(
        f"{name}.persistent_workers",
        dataloader_cfg.get("persistent_workers", False),
    )
    drop_last, drop_last_error = _parse_bool_field(
        f"{name}.drop_last",
        dataloader_cfg.get("drop_last", False),
    )
    errors = [
        error
        for error in (
            num_workers_error,
            pin_memory_error,
            persistent_workers_error,
            drop_last_error,
        )
        if error is not None
    ]
    worker_error = _nonnegative_int_error(f"{name}.num_workers", num_workers)
    if worker_error is not None:
        errors.append(worker_error)
    if persistent_workers and num_workers <= 0:
        errors.append(
            f"{name}.persistent_workers=true requires {name}.num_workers > 0."
        )
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": drop_last,
        "errors": errors,
    }


def validate_gaze_wam_training_config(cfg):
    """Summarize and validate training-loop parameters before Accelerator/DataLoader setup."""
    training = cfg.training
    errors = []
    robot_batch_size, error = _parse_int_field(
        "robot_dataloader.batch_size",
        cfg.robot_dataloader.get("batch_size", 0),
    )
    if error is not None:
        errors.append(error)
    open_batch_size, error = _parse_int_field(
        "open_dataloader.batch_size",
        cfg.open_dataloader.get("batch_size", 0),
    )
    if error is not None:
        errors.append(error)
    val_robot_batch_size, error = _parse_int_field(
        "val_robot_dataloader.batch_size",
        cfg.val_robot_dataloader.get("batch_size", 0),
    )
    if error is not None:
        errors.append(error)
    val_open_batch_size, error = _parse_int_field(
        "val_open_dataloader.batch_size",
        cfg.val_open_dataloader.get("batch_size", 0),
    )
    if error is not None:
        errors.append(error)
    gradient_accumulate_every, error = _parse_int_field(
        "training.gradient_accumulate_every",
        training.get("gradient_accumulate_every", 0),
    )
    if error is not None:
        errors.append(error)
    num_epochs, error = _parse_int_field("training.num_epochs", training.get("num_epochs", 0))
    if error is not None:
        errors.append(error)
    checkpoint_every, error = _parse_int_field(
        "training.checkpoint_every",
        training.get("checkpoint_every", 0),
    )
    if error is not None:
        errors.append(error)
    val_every, error = _parse_optional_int_field(
        "training.val_every",
        training.get("val_every", None),
    )
    if error is not None:
        errors.append(error)
    sample_every, error = _parse_optional_int_field(
        "training.sample_every",
        training.get("sample_every", None),
    )
    if error is not None:
        errors.append(error)
    gdr_every, error = _parse_optional_int_field(
        "training.gdr_every",
        training.get("gdr_every", None),
    )
    if error is not None:
        errors.append(error)
    max_train_steps, error = _parse_optional_int_field(
        "training.max_train_steps",
        training.get("max_train_steps", None),
    )
    if error is not None:
        errors.append(error)
    max_val_steps, error = _parse_optional_int_field(
        "training.max_val_steps",
        training.get("max_val_steps", None),
    )
    if error is not None:
        errors.append(error)
    lr_warmup_steps, error = _parse_int_field(
        "training.lr_warmup_steps",
        training.get("lr_warmup_steps", 0),
    )
    if error is not None:
        errors.append(error)
    tqdm_interval_sec, error = _parse_float_field(
        "training.tqdm_interval_sec",
        training.get("tqdm_interval_sec", 0.0),
    )
    if error is not None:
        errors.append(error)
    dataloaders = {
        "robot_dataloader": _parse_dataloader_runtime_config(
            "robot_dataloader",
            cfg.robot_dataloader,
        ),
        "open_dataloader": _parse_dataloader_runtime_config(
            "open_dataloader",
            cfg.open_dataloader,
        ),
        "val_robot_dataloader": _parse_dataloader_runtime_config(
            "val_robot_dataloader",
            cfg.val_robot_dataloader,
        ),
        "val_open_dataloader": _parse_dataloader_runtime_config(
            "val_open_dataloader",
            cfg.val_open_dataloader,
        ),
    }
    for dataloader_config in dataloaders.values():
        errors.extend(dataloader_config["errors"])
    total_batch_size = robot_batch_size + open_batch_size

    for name, value in (
        ("training.gradient_accumulate_every", gradient_accumulate_every),
        ("training.num_epochs", num_epochs),
        ("training.checkpoint_every", checkpoint_every),
    ):
        error = _positive_int_error(name, value)
        if error is not None:
            errors.append(error)
    for name, value in (
        ("robot_dataloader.batch_size", robot_batch_size),
        ("open_dataloader.batch_size", open_batch_size),
        ("val_robot_dataloader.batch_size", val_robot_batch_size),
        ("val_open_dataloader.batch_size", val_open_batch_size),
        ("training.lr_warmup_steps", lr_warmup_steps),
    ):
        error = _nonnegative_int_error(name, value)
        if error is not None:
            errors.append(error)
    if robot_batch_size <= 0 and val_robot_batch_size > 0:
        errors.append(
            "val_robot_dataloader.batch_size must be 0 when "
            "robot_dataloader.batch_size is 0."
        )
    for name, value in (
        ("training.val_every", val_every),
        ("training.sample_every", sample_every),
        ("training.gdr_every", gdr_every),
    ):
        error = _nonnegative_int_error(name, value, allow_none=True)
        if error is not None:
            errors.append(error)
    for name, value in (
        ("training.max_train_steps", max_train_steps),
        ("training.max_val_steps", max_val_steps),
    ):
        error = _positive_int_error(name, value, allow_none=True)
        if error is not None:
            errors.append(error)
    if total_batch_size <= 0:
        errors.append(
            "At least one training sample source must have positive batch size; "
            "robot_dataloader.batch_size + open_dataloader.batch_size must be > 0."
        )
    if tqdm_interval_sec < 0.0:
        errors.append(
            f"training.tqdm_interval_sec must be non-negative, got {tqdm_interval_sec}."
        )

    return {
        "robot_batch_size": robot_batch_size,
        "open_batch_size": open_batch_size,
        "train_batch_size_per_process": total_batch_size,
        "val_robot_batch_size": val_robot_batch_size,
        "val_open_batch_size": val_open_batch_size,
        "gradient_accumulate_every": gradient_accumulate_every,
        "num_epochs": num_epochs,
        "checkpoint_every": checkpoint_every,
        "val_every": val_every,
        "sample_every": sample_every,
        "gdr_every": gdr_every,
        "max_train_steps": max_train_steps,
        "max_val_steps": max_val_steps,
        "lr_warmup_steps": lr_warmup_steps,
        "tqdm_interval_sec": tqdm_interval_sec,
        "dataloaders": dataloaders,
        "errors": errors,
        "valid": len(errors) == 0,
    }


def _normalize_gaze_wam_training_config(cfg, training_config):
    """Apply parsed training-loop values after validation so downstream code sees native types."""
    if not training_config["valid"]:
        raise ValueError("Cannot normalize an invalid Gaze-WAM training config.")

    def optional_int(key: str):
        value = training_config[key]
        return None if value is None else int(value)

    cfg.robot_dataloader.batch_size = int(training_config["robot_batch_size"])
    cfg.open_dataloader.batch_size = int(training_config["open_batch_size"])
    cfg.val_robot_dataloader.batch_size = int(training_config["val_robot_batch_size"])
    cfg.val_open_dataloader.batch_size = int(training_config["val_open_batch_size"])

    cfg.training.gradient_accumulate_every = int(
        training_config["gradient_accumulate_every"]
    )
    cfg.training.num_epochs = int(training_config["num_epochs"])
    cfg.training.checkpoint_every = int(training_config["checkpoint_every"])
    cfg.training.val_every = optional_int("val_every")
    cfg.training.sample_every = optional_int("sample_every")
    cfg.training.gdr_every = optional_int("gdr_every")
    cfg.training.max_train_steps = optional_int("max_train_steps")
    cfg.training.max_val_steps = optional_int("max_val_steps")
    cfg.training.lr_warmup_steps = int(training_config["lr_warmup_steps"])
    cfg.training.tqdm_interval_sec = float(training_config["tqdm_interval_sec"])
    for key, dataloader_config in training_config["dataloaders"].items():
        dataloader_cfg = cfg[key]
        dataloader_cfg.num_workers = int(dataloader_config["num_workers"])
        dataloader_cfg.pin_memory = bool(dataloader_config["pin_memory"])
        dataloader_cfg.persistent_workers = bool(dataloader_config["persistent_workers"])
        if "drop_last" in dataloader_cfg or bool(dataloader_config["drop_last"]):
            dataloader_cfg.drop_last = bool(dataloader_config["drop_last"])
    return cfg
