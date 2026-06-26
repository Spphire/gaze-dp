from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_nonnegative_int_field,
)


def _parse_nonnegative_int(name: str, value) -> int:
    return normalize_gaze_wam_nonnegative_int_field(name, value)


def _safe_dataloader_batch_count(name: str, dataloader):
    if dataloader is None:
        return 0
    try:
        count = len(dataloader)
    except TypeError:
        return None
    try:
        count = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} dataloader length must be an integer, got {count!r}.") from exc
    if count < 0:
        raise ValueError(f"{name} dataloader length must be non-negative, got {count}.")
    return count


def _check_training_dataloader_lengths(
    robot_dataloader,
    robot_val_dataloader,
    open_dataloader=None,
    open_val_dataloader=None,
    robot_batch_size: int = 1,
    open_batch_size: int = 0,
):
    lengths = {
        "robot_train_batches": _safe_dataloader_batch_count(
            "robot train",
            robot_dataloader,
        ),
        "robot_val_batches": _safe_dataloader_batch_count(
            "robot val",
            robot_val_dataloader,
        ),
        "open_train_batches": _safe_dataloader_batch_count(
            "open train",
            open_dataloader,
        ),
        "open_val_batches": _safe_dataloader_batch_count(
            "open val",
            open_val_dataloader,
        ),
    }
    robot_batch_size = _parse_nonnegative_int("robot_batch_size", robot_batch_size)
    open_batch_size = _parse_nonnegative_int("open_batch_size", open_batch_size)
    errors = []
    robot_train_batches = lengths["robot_train_batches"]
    if robot_batch_size > 0 and robot_dataloader is None:
        errors.append("Robot train dataloader is enabled but was not constructed.")
    if robot_batch_size > 0 and robot_train_batches is not None and robot_train_batches <= 0:
        errors.append(
            "Robot train dataloader produced zero batches; check batch_size, drop_last, "
            "and dataset length before policy training."
        )
    if open_batch_size > 0:
        if open_dataloader is None:
            errors.append(
                "Open-source train dataloader is enabled but was not constructed."
            )
        open_train_batches = lengths["open_train_batches"]
        if open_train_batches is not None and open_train_batches <= 0:
            errors.append(
                "Open-source train dataloader is enabled but produced zero batches; "
                "check open_dataloader.batch_size, drop_last, and dataset length."
            )
    if errors:
        detail = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError("; ".join(errors) + f" Dataloader lengths: {detail}.")
    return lengths


def _check_training_dataset_lengths(
    robot_dataset,
    robot_val_dataset,
    open_dataset=None,
    open_val_dataset=None,
    robot_batch_size: int = 1,
    open_batch_size: int = 0,
):
    lengths = {
        "robot_train_samples": int(len(robot_dataset)) if robot_dataset is not None else 0,
        "robot_val_samples": int(len(robot_val_dataset)) if robot_val_dataset is not None else 0,
        "open_train_samples": int(len(open_dataset)) if open_dataset is not None else 0,
        "open_val_samples": int(len(open_val_dataset)) if open_val_dataset is not None else 0,
    }
    robot_batch_size = _parse_nonnegative_int("robot_batch_size", robot_batch_size)
    open_batch_size = _parse_nonnegative_int("open_batch_size", open_batch_size)
    errors = []
    if robot_batch_size > 0 and lengths["robot_train_samples"] <= 0:
        errors.append(
            "Robot train dataset produced zero samples; check episode length, val_ratio, "
            "action_horizon, n_latency_steps, downsampling, and action_padding."
        )
    if open_batch_size > 0 and lengths["open_train_samples"] <= 0:
        errors.append(
            "Open-source train dataset is enabled but produced zero samples; set "
            "open_dataloader.batch_size=0 or fix the open dataset sampling contract."
        )
    if errors:
        detail = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError("; ".join(errors) + f" Lengths: {detail}.")
    return lengths
