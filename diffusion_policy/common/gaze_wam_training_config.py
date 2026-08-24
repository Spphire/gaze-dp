import ast
import math

from omegaconf import open_dict


GAZE_WAM_BATCH_SIZE_SOURCES = ("auto", "ratio", "dataloader")
GAZE_WAM_TAIL_POLICIES = ("keep", "drop", "pad")
GAZE_WAM_TRAINING_STAGES = (
    "mixed_train",
    "open_only",
    "open_pretrain",
    "robot_finetune",
    "robot_only",
)
GAZE_WAM_TRANSFER_SCOPES = ("obs_encoder", "obs_and_gaze")


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
    batching_config=None,
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
    batching_config = dict(batching_config or {})
    resolved_batch_size_source = str(
        batching_config.get("resolved_batch_size_source", "dataloader")
    )
    if resolved_batch_size_source == "ratio":
        ratio_source = (
            "data_mixing.total_batch_size_per_process+"
            "data_mixing.robot_ratio+data_mixing.open_ratio"
        )
    else:
        ratio_source = "robot_dataloader.batch_size/open_dataloader.batch_size"
    robot_tail_policy = normalize_gaze_wam_tail_policy(
        "robot_tail_policy",
        batching_config.get("robot_tail_policy", "keep"),
    )
    open_tail_policy = normalize_gaze_wam_tail_policy(
        "open_tail_policy",
        batching_config.get("open_tail_policy", "keep"),
    )
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
            "tail_policy": robot_tail_policy,
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
            "tail_policy": open_tail_policy,
            "has_action": False,
            "action_values_used_for_loss": False,
        },
        "mixing": {
            "builder": (
                "diffusion_policy.dataset.gaze_wam_mixing."
                "build_gaze_wam_mixed_batch"
            ),
            "mode": "online_per_step_concat_after_fetch",
            "ratio_source": ratio_source,
            "resolved_batch_size_source": resolved_batch_size_source,
            "requested_total_batch_size_per_process": batching_config.get(
                "requested_total_batch_size_per_process"
            ),
            "requested_robot_ratio": batching_config.get("requested_robot_ratio"),
            "requested_open_ratio": batching_config.get("requested_open_ratio"),
            "total_batch_size_per_process": total_batch_size,
            "shuffle_after_concat": True,
            "primary_epoch_driver": (
                "robot_dataloader"
                if robot_batch_size > 0
                else "open_dataloader"
                if open_batch_size > 0
                else "none"
            ),
            "open_iterator_policy": "restart_on_exhaustion",
            "robot_tail_policy": robot_tail_policy,
            "open_tail_policy": open_tail_policy,
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


def _normalize_choice_field(name: str, value, choices, default: str) -> str:
    if value is None:
        value = default
    parsed = str(value).strip().lower()
    if parsed not in choices:
        options = ", ".join(str(item) for item in choices)
        raise ValueError(f"{name} must be one of: {options}; got {value!r}.")
    return parsed


def normalize_gaze_wam_batch_size_source(name: str, value, default: str = "auto") -> str:
    return _normalize_choice_field(
        name,
        value,
        GAZE_WAM_BATCH_SIZE_SOURCES,
        default,
    )


def normalize_gaze_wam_tail_policy(name: str, value, default: str = "keep") -> str:
    return _normalize_choice_field(name, value, GAZE_WAM_TAIL_POLICIES, default)


def normalize_gaze_wam_training_stage(
    name: str,
    value,
    default: str = "mixed_train",
) -> str:
    return _normalize_choice_field(name, value, GAZE_WAM_TRAINING_STAGES, default)


def normalize_gaze_wam_transfer_scope(
    name: str,
    value,
    default: str = "obs_encoder",
) -> str:
    return _normalize_choice_field(name, value, GAZE_WAM_TRANSFER_SCOPES, default)


def resolve_gaze_wam_batching_config(cfg) -> dict:
    """Resolve source quotas from total batch size and requested source ratios.

    ``auto`` preserves old Hydra overrides that directly change the two dataloader
    batch sizes. New configs should use ``ratio`` so the total and ratios are the
    single source of truth.
    """
    errors = []
    mixing = cfg.get("data_mixing", None)
    if mixing is None:
        mixing = {}

    legacy_robot_batch, error = _parse_int_field(
        "robot_dataloader.batch_size",
        cfg.robot_dataloader.get("batch_size", 0),
    )
    if error is not None:
        errors.append(error)
    legacy_open_batch, error = _parse_int_field(
        "open_dataloader.batch_size",
        cfg.open_dataloader.get("batch_size", 0),
    )
    if error is not None:
        errors.append(error)
    try:
        batch_size_source = normalize_gaze_wam_batch_size_source(
            "data_mixing.batch_size_source",
            mixing.get("batch_size_source", "auto"),
        )
    except ValueError as exc:
        errors.append(str(exc))
        batch_size_source = "auto"

    ratio_fields_present = any(
        mixing.get(key, None) is not None
        for key in (
            "total_batch_size_per_process",
            "robot_ratio",
            "open_ratio",
        )
    )
    requested_total = None
    requested_robot_ratio = None
    requested_open_ratio = None
    ratio_robot_batch = None
    ratio_open_batch = None
    if ratio_fields_present or batch_size_source == "ratio":
        try:
            requested_total = normalize_gaze_wam_positive_int_field(
                "data_mixing.total_batch_size_per_process",
                mixing.get("total_batch_size_per_process", None),
            )
        except ValueError as exc:
            errors.append(str(exc))
        try:
            requested_robot_ratio = normalize_gaze_wam_unit_interval_float_field(
                "data_mixing.robot_ratio",
                mixing.get("robot_ratio", None),
                include_one=True,
            )
        except ValueError as exc:
            errors.append(str(exc))
        raw_open_ratio = mixing.get("open_ratio", None)
        if requested_robot_ratio is not None and raw_open_ratio is None:
            requested_open_ratio = 1.0 - requested_robot_ratio
        else:
            try:
                requested_open_ratio = normalize_gaze_wam_unit_interval_float_field(
                    "data_mixing.open_ratio",
                    raw_open_ratio,
                    include_one=True,
                )
            except ValueError as exc:
                errors.append(str(exc))
        if (
            requested_robot_ratio is not None
            and requested_open_ratio is not None
            and not math.isclose(
                requested_robot_ratio + requested_open_ratio,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append(
                "data_mixing.robot_ratio + data_mixing.open_ratio must equal 1.0; "
                f"got {requested_robot_ratio + requested_open_ratio:.12g}."
            )
        if (
            requested_total is not None
            and requested_robot_ratio is not None
            and requested_open_ratio is not None
            and math.isclose(
                requested_robot_ratio + requested_open_ratio,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            ratio_robot_batch = int(
                math.floor(requested_total * requested_robot_ratio + 0.5)
            )
            ratio_open_batch = int(requested_total - ratio_robot_batch)
            if requested_robot_ratio > 0.0 and ratio_robot_batch <= 0:
                errors.append(
                    "data_mixing total batch size is too small to allocate a positive "
                    "robot quota for the requested robot_ratio."
                )
            if requested_open_ratio > 0.0 and ratio_open_batch <= 0:
                errors.append(
                    "data_mixing total batch size is too small to allocate a positive "
                    "open-source quota for the requested open_ratio."
                )

    compatibility_fallback = False
    if batch_size_source == "dataloader" or not ratio_fields_present:
        resolved_source = "dataloader"
        robot_batch_size = legacy_robot_batch
        open_batch_size = legacy_open_batch
    elif batch_size_source == "auto" and (
        ratio_robot_batch != legacy_robot_batch
        or ratio_open_batch != legacy_open_batch
    ):
        resolved_source = "dataloader"
        compatibility_fallback = True
        robot_batch_size = legacy_robot_batch
        open_batch_size = legacy_open_batch
    else:
        resolved_source = "ratio"
        robot_batch_size = ratio_robot_batch if ratio_robot_batch is not None else 0
        open_batch_size = ratio_open_batch if ratio_open_batch is not None else 0

    total_batch_size = int(robot_batch_size) + int(open_batch_size)
    robot_ratio = (
        float(robot_batch_size) / float(total_batch_size)
        if total_batch_size > 0
        else 0.0
    )
    open_ratio = (
        float(open_batch_size) / float(total_batch_size)
        if total_batch_size > 0
        else 0.0
    )
    try:
        robot_tail_policy = normalize_gaze_wam_tail_policy(
            "data_mixing.robot_tail_policy",
            mixing.get("robot_tail_policy", None),
            default="keep" if not ratio_fields_present else "pad",
        )
    except ValueError as exc:
        errors.append(str(exc))
        robot_tail_policy = "keep"
    try:
        open_tail_policy = normalize_gaze_wam_tail_policy(
            "data_mixing.open_tail_policy",
            mixing.get("open_tail_policy", None),
            default="keep" if not ratio_fields_present else "pad",
        )
    except ValueError as exc:
        errors.append(str(exc))
        open_tail_policy = "keep"
    try:
        validation_tail_policy = normalize_gaze_wam_tail_policy(
            "data_mixing.validation_tail_policy",
            mixing.get("validation_tail_policy", "keep"),
            default="keep",
        )
    except ValueError as exc:
        errors.append(str(exc))
        validation_tail_policy = "keep"

    return {
        "batch_size_source": batch_size_source,
        "resolved_batch_size_source": resolved_source,
        "compatibility_fallback_to_dataloader": compatibility_fallback,
        "ratio_fields_present": ratio_fields_present,
        "requested_total_batch_size_per_process": requested_total,
        "requested_robot_ratio": requested_robot_ratio,
        "requested_open_ratio": requested_open_ratio,
        "configured_robot_dataloader_batch_size": legacy_robot_batch,
        "configured_open_dataloader_batch_size": legacy_open_batch,
        "robot_batch_size": int(robot_batch_size),
        "open_batch_size": int(open_batch_size),
        "train_batch_size_per_process": int(total_batch_size),
        "robot_ratio": robot_ratio,
        "open_ratio": open_ratio,
        "robot_tail_policy": robot_tail_policy,
        "open_tail_policy": open_tail_policy,
        "validation_tail_policy": validation_tail_policy,
        "errors": errors,
        "valid": len(errors) == 0,
    }


def gaze_wam_planned_optimizer_steps(
    *,
    steps_per_epoch: int,
    num_epochs: int,
    gradient_accumulate_every: int,
    max_train_steps=None,
) -> int:
    steps_per_epoch = normalize_gaze_wam_positive_int_field(
        "steps_per_epoch",
        steps_per_epoch,
    )
    num_epochs = normalize_gaze_wam_positive_int_field("num_epochs", num_epochs)
    gradient_accumulate_every = normalize_gaze_wam_positive_int_field(
        "gradient_accumulate_every",
        gradient_accumulate_every,
    )
    # Accumulation is flushed at every epoch boundary, so an incomplete final
    # accumulation window costs one optimizer step per epoch.
    planned = int(
        math.ceil(steps_per_epoch / float(gradient_accumulate_every))
        * num_epochs
    )
    if max_train_steps is not None:
        max_train_steps = normalize_gaze_wam_positive_int_field(
            "max_train_steps",
            max_train_steps,
        )
        planned = min(planned, max_train_steps)
    return max(planned, 1)


def gaze_wam_prepared_dataloader_batches(
    raw_batches: int,
    *,
    num_processes: int = 1,
    split_batches: bool = False,
    even_batches: bool = True,
    drop_last: bool = False,
    process_index: int = 0,
) -> int:
    """Mirror Accelerate's map-style batch sharding length calculation.

    The scheduler is created before ``accelerator.prepare`` in the workspace,
    so its step budget must use the same per-rank dataloader length that
    Accelerate will produce.  In particular, ``drop_last`` uses floor division
    while ``even_batches`` pads to the next process group.
    """
    raw_batches = normalize_gaze_wam_nonnegative_int_field(
        "raw_batches", raw_batches
    )
    num_processes = normalize_gaze_wam_positive_int_field(
        "num_processes", num_processes
    )
    process_index = normalize_gaze_wam_nonnegative_int_field(
        "process_index", process_index
    )
    if process_index >= num_processes:
        raise ValueError(
            f"process_index must be less than num_processes; got "
            f"{process_index} >= {num_processes}."
        )
    if raw_batches == 0:
        return 0
    if bool(split_batches) or num_processes == 1:
        return raw_batches

    quotient, remainder = divmod(raw_batches, num_processes)
    if remainder == 0 or bool(drop_last):
        return quotient
    if bool(even_batches):
        return quotient + 1
    return quotient + (1 if process_index < remainder else 0)


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
    batching_config = resolve_gaze_wam_batching_config(cfg)
    errors.extend(batching_config["errors"])
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
    if batching_config["valid"]:
        robot_batch_size = int(batching_config["robot_batch_size"])
        open_batch_size = int(batching_config["open_batch_size"])
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
    checkpoint_every_steps, error = _parse_optional_int_field(
        "training.checkpoint_every_steps",
        training.get("checkpoint_every_steps", None),
    )
    if error is not None:
        errors.append(error)
    latest_checkpoint_every_steps, error = _parse_optional_int_field(
        "training.latest_checkpoint_every_steps",
        training.get("latest_checkpoint_every_steps", None),
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
    total_batch_size = int(robot_batch_size) + int(open_batch_size)

    for name, value in (
        ("training.gradient_accumulate_every", gradient_accumulate_every),
        ("training.num_epochs", num_epochs),
        ("training.checkpoint_every", checkpoint_every),
    ):
        error = _positive_int_error(name, value)
        if error is not None:
            errors.append(error)
    error = _positive_int_error(
        "training.checkpoint_every_steps",
        checkpoint_every_steps,
        allow_none=True,
    )
    if error is not None:
        errors.append(error)
    error = _positive_int_error(
        "training.latest_checkpoint_every_steps",
        latest_checkpoint_every_steps,
        allow_none=True,
    )
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

    try:
        stage = normalize_gaze_wam_training_stage(
            "training.stage",
            training.get("stage", "mixed_train"),
        )
    except ValueError as exc:
        errors.append(str(exc))
        stage = "mixed_train"
    transfer = training.get("transfer", None) or {}
    transfer_load_path = str(transfer.get("load_path", "") or "").strip()
    transfer_export_path = str(transfer.get("export_path", "") or "").strip()
    try:
        transfer_load_scope = normalize_gaze_wam_transfer_scope(
            "training.transfer.load_scope",
            transfer.get("load_scope", "obs_encoder"),
        )
    except ValueError as exc:
        errors.append(str(exc))
        transfer_load_scope = "obs_encoder"
    try:
        transfer_export_scope = normalize_gaze_wam_transfer_scope(
            "training.transfer.export_scope",
            transfer.get("export_scope", "obs_encoder"),
        )
    except ValueError as exc:
        errors.append(str(exc))
        transfer_export_scope = "obs_encoder"
    try:
        transfer_export_overwrite = normalize_gaze_wam_bool_field(
            "training.transfer.export_overwrite",
            transfer.get("export_overwrite", False),
            default=False,
        )
    except ValueError as exc:
        errors.append(str(exc))
        transfer_export_overwrite = False
    if stage in ("open_only", "open_pretrain") and robot_batch_size > 0:
        errors.append(
            f"training.stage={stage} requires robot_dataloader.batch_size=0 "
            "so the epoch is open-source only."
        )
    if stage in ("mixed_train", "robot_finetune", "robot_only") and robot_batch_size <= 0:
        errors.append(
            f"training.stage={stage} requires a positive robot batch quota."
        )
    if stage == "mixed_train" and open_batch_size <= 0:
        errors.append(
            "training.stage=mixed_train requires a positive open-source quota; "
            "use training.stage=robot_only for a robot-only run."
        )
    if stage == "robot_only" and open_batch_size > 0:
        errors.append(
            "training.stage=robot_only requires open_dataloader.batch_size=0."
        )
    if stage == "robot_finetune" and open_batch_size > 0:
        errors.append(
            "training.stage=robot_finetune requires open_dataloader.batch_size=0; "
            "use mixed_train for a source mixture."
        )
    if stage in ("mixed_train", "robot_only", "robot_finetune") and val_robot_batch_size <= 0:
        errors.append(
            f"training.stage={stage} requires a positive val_robot_dataloader.batch_size "
            "so robot validation remains the selection gate."
        )
    if stage in ("mixed_train", "robot_only", "robot_finetune"):
        checkpoint_cfg = cfg.get("checkpoint", {}) or {}
        checkpoint_topk = checkpoint_cfg.get("topk", {}) or {}
        monitor_key = str(checkpoint_topk.get("monitor_key", ""))
        if monitor_key != "val_robot_loss":
            errors.append(
                f"training.stage={stage} requires checkpoint.topk.monitor_key="
                f"val_robot_loss, got {monitor_key!r}."
            )
    # Open pretraining may be bounded by either max_train_steps or num_epochs.
    # num_epochs is already validated as a positive integer above.
    if stage in ("open_only", "open_pretrain") and open_batch_size <= 0:
        errors.append(
            f"training.stage={stage} requires a positive open-source batch quota."
        )
    if stage == "open_pretrain" and val_open_batch_size <= 0:
        errors.append(
            "training.stage=open_pretrain requires a positive "
            "val_open_dataloader.batch_size for the optional ablation gate."
        )
    if stage == "open_pretrain" and transfer_export_path == "":
        errors.append(
            "training.stage=open_pretrain requires training.transfer.export_path "
            "so the stage has an explicit hand-off artifact."
        )
    if stage == "robot_finetune" and transfer_load_path == "":
        errors.append(
            "training.stage=robot_finetune requires training.transfer.load_path."
        )
    if bool(training.get("resume", False)) and transfer_load_path:
        errors.append(
            "training.resume and training.transfer.load_path cannot be enabled "
            "together; choose checkpoint resume or an explicit transfer hand-off."
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
        "checkpoint_every_steps": checkpoint_every_steps,
        "latest_checkpoint_every_steps": latest_checkpoint_every_steps,
        "val_every": val_every,
        "sample_every": sample_every,
        "gdr_every": gdr_every,
        "max_train_steps": max_train_steps,
        "max_val_steps": max_val_steps,
        "lr_warmup_steps": lr_warmup_steps,
        "tqdm_interval_sec": tqdm_interval_sec,
        "batching": batching_config,
        "stage": stage,
        "transfer": {
            "load_path": transfer_load_path,
            "load_scope": transfer_load_scope,
            "export_path": transfer_export_path,
            "export_scope": transfer_export_scope,
            "export_overwrite": transfer_export_overwrite,
            "export_path_configured": bool(transfer_export_path),
            "export_path_optional_warning": False,
        },
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
    with open_dict(cfg.training):
        cfg.training.checkpoint_every_steps = optional_int("checkpoint_every_steps")
        cfg.training.latest_checkpoint_every_steps = optional_int(
            "latest_checkpoint_every_steps"
        )
    cfg.training.val_every = optional_int("val_every")
    cfg.training.sample_every = optional_int("sample_every")
    cfg.training.gdr_every = optional_int("gdr_every")
    cfg.training.max_train_steps = optional_int("max_train_steps")
    cfg.training.max_val_steps = optional_int("max_val_steps")
    cfg.training.lr_warmup_steps = int(training_config["lr_warmup_steps"])
    cfg.training.tqdm_interval_sec = float(training_config["tqdm_interval_sec"])
    cfg.training.stage = str(training_config["stage"])
    if "transfer" in cfg.training:
        with open_dict(cfg.training.transfer):
            cfg.training.transfer.load_path = str(
                training_config["transfer"]["load_path"]
            )
            cfg.training.transfer.load_scope = str(
                training_config["transfer"]["load_scope"]
            )
            cfg.training.transfer.export_path = str(
                training_config["transfer"]["export_path"]
            )
            cfg.training.transfer.export_scope = str(
                training_config["transfer"]["export_scope"]
            )
            cfg.training.transfer.export_overwrite = bool(
                training_config["transfer"]["export_overwrite"]
            )
    if "data_mixing" in cfg:
        batching = training_config["batching"]
        with open_dict(cfg.data_mixing):
            cfg.data_mixing.resolved_batch_size_source = str(
                batching["resolved_batch_size_source"]
            )
            cfg.data_mixing.resolved_robot_batch_size = int(
                batching["robot_batch_size"]
            )
            cfg.data_mixing.resolved_open_batch_size = int(
                batching["open_batch_size"]
            )
            cfg.data_mixing.resolved_total_batch_size_per_process = int(
                batching["train_batch_size_per_process"]
            )
            cfg.data_mixing.resolved_robot_ratio = float(batching["robot_ratio"])
            cfg.data_mixing.resolved_open_ratio = float(batching["open_ratio"])
            cfg.data_mixing.resolved_robot_tail_policy = str(
                batching["robot_tail_policy"]
            )
            cfg.data_mixing.resolved_open_tail_policy = str(
                batching["open_tail_policy"]
            )
            cfg.data_mixing.resolved_validation_tail_policy = str(
                batching["validation_tail_policy"]
            )
    for key, dataloader_config in training_config["dataloaders"].items():
        dataloader_cfg = cfg[key]
        dataloader_cfg.num_workers = int(dataloader_config["num_workers"])
        dataloader_cfg.pin_memory = bool(dataloader_config["pin_memory"])
        dataloader_cfg.persistent_workers = bool(dataloader_config["persistent_workers"])
        if "drop_last" in dataloader_cfg or bool(dataloader_config["drop_last"]):
            dataloader_cfg.drop_last = bool(dataloader_config["drop_last"])
    return cfg
