from typing import Dict, Optional

import torch

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_heatmap_supervision_mode,
    normalize_gaze_wam_unit_interval_float_field,
)


def _batch_size(name: str, value: torch.Tensor) -> int:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim == 0:
        raise ValueError(f"{name} must have a batch dimension, got scalar tensor.")
    return int(value.shape[0])


def _require_batch_size(name: str, value: torch.Tensor, expected: int) -> None:
    actual = _batch_size(name, value)
    if actual != int(expected):
        raise ValueError(
            f"{name} batch size must be {int(expected)}, got {actual} with shape "
            f"{tuple(value.shape)}."
        )


def _require_batch_vector(name: str, value: torch.Tensor, expected: int) -> None:
    _require_batch_size(name, value, expected)
    if value.ndim != 1:
        raise ValueError(
            f"{name} must have shape [{int(expected)}], got {tuple(value.shape)}."
        )


def _require_bool_mask(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be a BoolTensor, got dtype {value.dtype}.")
    return value


def _optional_presence_mask(
    source_name: str,
    batch: Dict[str, torch.Tensor],
    optional_key: str,
    batch_size: int,
    device: torch.device,
    default_value: bool,
) -> torch.Tensor:
    mask_key = f"has_{optional_key}"
    mask = batch.get(mask_key)
    if mask is None:
        return torch.full((batch_size,), bool(default_value), device=device, dtype=torch.bool)
    if not torch.is_tensor(mask):
        raise TypeError(
            f"{source_name}[{mask_key!r}] must be a torch.Tensor, got "
            f"{type(mask).__name__}."
        )
    mask = mask.to(device=device)
    _require_batch_vector(f"{source_name}[{mask_key!r}]", mask, batch_size)
    return _require_bool_mask(f"{source_name}[{mask_key!r}]", mask)


def _optional_bool_field(
    source_name: str,
    batch: Dict[str, torch.Tensor],
    key: str,
    batch_size: int,
    device: torch.device,
    default_value: bool,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.full((batch_size,), bool(default_value), device=device, dtype=torch.bool)
    if not torch.is_tensor(value):
        raise TypeError(
            f"{source_name}[{key!r}] must be a torch.Tensor, got {type(value).__name__}."
        )
    value = value.to(device=device)
    _require_batch_vector(f"{source_name}[{key!r}]", value, batch_size)
    return _require_bool_mask(f"{source_name}[{key!r}]", value)


def _source_gaze_label_mask(
    source_name: str,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = batch.get(
        "has_gaze_label",
        torch.ones(batch_size, device=device, dtype=torch.bool),
    )
    if not torch.is_tensor(value):
        raise TypeError(
            f"{source_name}['has_gaze_label'] must be a torch.Tensor, got "
            f"{type(value).__name__}."
        )
    value = value.to(device=device)
    _require_batch_vector(f"{source_name}['has_gaze_label']", value, batch_size)
    return _require_bool_mask(f"{source_name}['has_gaze_label']", value)


def _source_gaze_condition_mask(
    source_name: str,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    has_gaze_label: torch.Tensor,
) -> torch.Tensor:
    value = batch.get("has_gaze_condition", has_gaze_label)
    if not torch.is_tensor(value):
        raise TypeError(
            f"{source_name}['has_gaze_condition'] must be a torch.Tensor, got "
            f"{type(value).__name__}."
        )
    value = value.to(device=device)
    _require_batch_vector(f"{source_name}['has_gaze_condition']", value, batch_size)
    value = _require_bool_mask(f"{source_name}['has_gaze_condition']", value)
    if torch.any(has_gaze_label & ~value):
        raise ValueError(
            f"{source_name}['has_gaze_label'] cannot be True where "
            f"{source_name}['has_gaze_condition'] is False."
        )
    return value


def _require_tail_shape(name: str, value: torch.Tensor, reference_name: str, reference: torch.Tensor) -> None:
    if tuple(value.shape[1:]) != tuple(reference.shape[1:]):
        raise ValueError(
            f"{name} tail shape must match {reference_name}; got {tuple(value.shape)} "
            f"vs {tuple(reference.shape)}."
        )


def _concat_obs(robot_obs: Dict[str, torch.Tensor], open_obs: Dict[str, torch.Tensor]):
    if set(robot_obs.keys()) != set(open_obs.keys()):
        raise ValueError(
            "robot and open obs keys must match before mixing, got "
            f"{sorted(robot_obs.keys())} vs {sorted(open_obs.keys())}."
        )
    for key in sorted(robot_obs.keys()):
        _require_tail_shape(
            f"open_batch['obs'][{key!r}]",
            open_obs[key],
            f"robot_batch['obs'][{key!r}]",
            robot_obs[key],
        )
    return {
        key: torch.cat([robot_obs[key], open_obs[key].to(robot_obs[key].device)], dim=0)
        for key in sorted(robot_obs.keys())
    }


def _shuffle_batch(batch, generator: Optional[torch.Generator] = None):
    batch_size = batch["gaze_xy"].shape[0]
    if batch["gaze_xy"].device.type == "cpu":
        perm = torch.randperm(
            batch_size,
            device=batch["gaze_xy"].device,
            generator=generator,
        )
    else:
        perm = torch.randperm(batch_size, device=batch["gaze_xy"].device)

    def apply_perm(value):
        if isinstance(value, dict):
            return {key: apply_perm(item) for key, item in value.items()}
        return value[perm]

    return {key: apply_perm(value) for key, value in batch.items()}


def _zero_rows_where_mask_false(name: str, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _require_batch_vector(f"{name} availability mask", mask, _batch_size(name, value))
    mask = mask.to(device=value.device, dtype=value.dtype)
    view_shape = (int(mask.shape[0]),) + (1,) * (value.ndim - 1)
    return value * mask.reshape(view_shape)


def build_gaze_wam_mixed_batch(
    robot_batch: Optional[Dict[str, torch.Tensor]],
    open_batch: Optional[Dict[str, torch.Tensor]],
    robot_gaze_dropout_prob: float = 0.2,
    robot_heatmap_on_gaze_dropout: bool = True,
    robot_heatmap_supervision: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    shuffle: bool = True,
) -> Dict[str, torch.Tensor]:
    """Merge robot and open-source gaze batches into the Gaze-WAM tensor contract."""
    robot_gaze_dropout_prob = normalize_gaze_wam_unit_interval_float_field(
        "robot_gaze_dropout_prob",
        robot_gaze_dropout_prob,
        default=0.2,
        include_one=True,
    )
    robot_heatmap_on_gaze_dropout = normalize_gaze_wam_bool_field(
        "robot_heatmap_on_gaze_dropout",
        robot_heatmap_on_gaze_dropout,
        default=True,
    )
    if robot_heatmap_supervision is None:
        robot_heatmap_supervision = (
            "dropout_only" if robot_heatmap_on_gaze_dropout else "none"
        )
    robot_heatmap_supervision = normalize_gaze_wam_heatmap_supervision_mode(
        "robot_heatmap_supervision",
        robot_heatmap_supervision,
        default="dropout_only",
    )
    if robot_batch is None and open_batch is None:
        raise ValueError("At least one of robot_batch or open_batch must be provided.")
    if robot_batch is None:
        if "obs" not in open_batch:
            raise KeyError("open_batch must contain an 'obs' dict.")
        if "action" not in open_batch:
            raise KeyError(
                "open-only Gaze-WAM batches require open_batch['action'] zero placeholders."
            )
        if "gaze_xy" not in open_batch:
            raise KeyError("open_batch must contain gaze_xy labels.")
        if "heatmap" not in open_batch:
            raise KeyError(
                "open_batch must contain heatmap placeholders; uncached heatmap "
                "targets are generated online from gaze_xy."
            )

        open_size = _batch_size("open_batch['gaze_xy']", open_batch["gaze_xy"])
        device = open_batch["gaze_xy"].device
        open_action = open_batch["action"]
        _require_batch_size("open_batch['action']", open_action, open_size)
        _require_batch_size("open_batch['heatmap']", open_batch["heatmap"], open_size)
        for obs_key, obs_value in open_batch["obs"].items():
            _require_batch_size(f"open_batch['obs'][{obs_key!r}]", obs_value, open_size)

        open_has_gaze_label = _source_gaze_label_mask(
            "open_batch",
            open_batch,
            open_size,
            device,
        )
        open_has_gaze_condition = _source_gaze_condition_mask(
            "open_batch", open_batch, open_size, device, open_has_gaze_label
        )
        open_gaze_xy = _zero_rows_where_mask_false(
            "open_batch['gaze_xy']",
            open_batch["gaze_xy"],
            open_has_gaze_condition,
        )
        open_has_heatmap_target = _optional_bool_field(
            "open_batch", open_batch, "has_heatmap_target", open_size, device, True
        )
        open_heatmap_is_cached = _optional_bool_field(
            "open_batch", open_batch, "heatmap_is_cached", open_size, device, False
        )
        batch = {
            "obs": open_batch["obs"],
            "action": torch.zeros_like(open_action),
            "heatmap": _zero_rows_where_mask_false(
                "open_batch['heatmap']", open_batch["heatmap"], open_has_heatmap_target
            ),
            "gaze_xy": open_gaze_xy,
            "is_open": torch.ones(open_size, device=device, dtype=torch.bool),
            "has_action": torch.zeros(open_size, device=device, dtype=torch.bool),
            "has_heatmap": open_has_heatmap_target,
            "has_heatmap_target": open_has_heatmap_target,
            "heatmap_is_cached": open_heatmap_is_cached,
            "has_gaze_condition": open_has_gaze_condition,
            "has_gaze_label": open_has_gaze_label,
            "use_gaze_condition": torch.zeros(open_size, device=device, dtype=torch.bool),
            "is_gaze_condition_dropped": torch.ones(
                open_size,
                device=device,
                dtype=torch.bool,
            ),
        }
        for optional_key in ("action_abs", "action_base_abs", "heatmap_image"):
            open_value = open_batch.get(optional_key)
            if open_value is None:
                continue
            _require_batch_size(f"open_batch[{optional_key!r}]", open_value, open_size)
            if optional_key in ("action_abs", "action_base_abs"):
                batch[optional_key] = torch.zeros_like(open_value)
                batch[f"has_{optional_key}"] = torch.zeros(
                    open_size,
                    device=device,
                    dtype=torch.bool,
                )
                continue
            open_optional_mask = _optional_presence_mask(
                "open_batch",
                open_batch,
                optional_key,
                open_size,
                device,
                default_value=True,
            )
            batch[optional_key] = _zero_rows_where_mask_false(
                f"open_batch[{optional_key!r}]",
                open_value,
                open_optional_mask,
            )
            batch[f"has_{optional_key}"] = open_optional_mask
        if shuffle:
            batch = _shuffle_batch(batch, generator=generator)
        return batch

    if "obs" not in robot_batch:
        raise KeyError("robot_batch must contain an 'obs' dict.")
    if "action" not in robot_batch:
        raise KeyError("robot_batch must contain relative action targets.")
    if "gaze_xy" not in robot_batch:
        raise KeyError("robot_batch must contain gaze_xy labels.")
    if "heatmap" not in robot_batch:
        raise KeyError(
            "robot_batch must contain heatmap placeholders; uncached heatmap "
            "targets are generated online from gaze_xy when routed."
        )

    robot_action = robot_batch["action"]
    robot_size = _batch_size("robot_batch['gaze_xy']", robot_batch["gaze_xy"])
    device = robot_batch["gaze_xy"].device
    _require_batch_size("robot_batch['action']", robot_action, robot_size)
    _require_batch_size("robot_batch['heatmap']", robot_batch["heatmap"], robot_size)
    for obs_key, obs_value in robot_batch["obs"].items():
        _require_batch_size(f"robot_batch['obs'][{obs_key!r}]", obs_value, robot_size)

    robot_dropout = torch.rand(
        robot_size,
        device=device,
        generator=generator,
    ) < robot_gaze_dropout_prob
    robot_has_gaze_label = _source_gaze_label_mask(
        "robot_batch",
        robot_batch,
        robot_size,
        device,
    )
    robot_has_gaze_condition = _source_gaze_condition_mask(
        "robot_batch", robot_batch, robot_size, device, robot_has_gaze_label
    )
    robot_gaze_xy = _zero_rows_where_mask_false(
        "robot_batch['gaze_xy']",
        robot_batch["gaze_xy"],
        robot_has_gaze_condition,
    )
    robot_uses_gaze_condition = (~robot_dropout) & robot_has_gaze_condition
    robot_gaze_condition_dropped = ~robot_uses_gaze_condition
    robot_has_heatmap_target = _optional_bool_field(
        "robot_batch", robot_batch, "has_heatmap_target", robot_size, device, True
    )
    robot_heatmap_is_cached = _optional_bool_field(
        "robot_batch", robot_batch, "heatmap_is_cached", robot_size, device, False
    )
    if robot_heatmap_supervision == "all_valid":
        robot_has_heatmap = robot_has_heatmap_target & robot_has_gaze_label
    elif robot_heatmap_supervision == "dropout_only":
        robot_has_heatmap = (
            robot_gaze_condition_dropped
            & robot_has_heatmap_target
            & robot_has_gaze_label
        )
    else:
        robot_has_heatmap = torch.zeros_like(robot_gaze_condition_dropped)
    if open_batch is None:
        batch = {
            "obs": robot_batch["obs"],
            "action": robot_action,
            "heatmap": _zero_rows_where_mask_false(
                "robot_batch['heatmap']",
                robot_batch["heatmap"],
                robot_has_heatmap,
            ),
            "gaze_xy": robot_gaze_xy,
            "is_open": torch.zeros(robot_size, device=device, dtype=torch.bool),
            "has_action": torch.ones(robot_size, device=device, dtype=torch.bool),
            "has_heatmap": robot_has_heatmap,
            "has_heatmap_target": robot_has_heatmap_target,
            "heatmap_is_cached": robot_heatmap_is_cached,
            "has_gaze_condition": robot_has_gaze_condition,
            "has_gaze_label": robot_has_gaze_label,
            "use_gaze_condition": robot_uses_gaze_condition,
            "is_gaze_condition_dropped": robot_gaze_condition_dropped,
        }
        for optional_key in ("action_abs", "action_base_abs", "heatmap_image"):
            robot_value = robot_batch.get(optional_key)
            if robot_value is None:
                continue
            _require_batch_size(f"robot_batch[{optional_key!r}]", robot_value, robot_size)
            robot_optional_mask = _optional_presence_mask(
                "robot_batch",
                robot_batch,
                optional_key,
                robot_size,
                device,
                default_value=True,
            )
            batch[optional_key] = _zero_rows_where_mask_false(
                f"robot_batch[{optional_key!r}]",
                robot_value,
                robot_optional_mask,
            )
            batch[f"has_{optional_key}"] = robot_optional_mask
        if shuffle:
            batch = _shuffle_batch(batch, generator=generator)
        return batch

    if "obs" not in open_batch:
        raise KeyError("open_batch must contain an 'obs' dict.")
    if "gaze_xy" not in open_batch:
        raise KeyError("open_batch must contain gaze_xy labels.")
    if "heatmap" not in open_batch:
        raise KeyError(
            "open_batch must contain heatmap placeholders; uncached heatmap "
            "targets are generated online from gaze_xy."
        )

    open_size = _batch_size("open_batch['gaze_xy']", open_batch["gaze_xy"])
    _require_tail_shape(
        "open_batch['gaze_xy']",
        open_batch["gaze_xy"],
        "robot_batch['gaze_xy']",
        robot_batch["gaze_xy"],
    )
    _require_batch_size("open_batch['heatmap']", open_batch["heatmap"], open_size)
    _require_tail_shape(
        "open_batch['heatmap']",
        open_batch["heatmap"],
        "robot_batch['heatmap']",
        robot_batch["heatmap"],
    )
    for obs_key, obs_value in open_batch["obs"].items():
        _require_batch_size(f"open_batch['obs'][{obs_key!r}]", obs_value, open_size)
    open_has_gaze_label = _source_gaze_label_mask(
        "open_batch",
        open_batch,
        open_size,
        device,
    )
    open_has_gaze_condition = _source_gaze_condition_mask(
        "open_batch", open_batch, open_size, device, open_has_gaze_label
    )
    open_gaze_xy = _zero_rows_where_mask_false(
        "open_batch['gaze_xy']",
        open_batch["gaze_xy"].to(robot_batch["gaze_xy"].device),
        open_has_gaze_condition,
    )
    open_has_heatmap_target = _optional_bool_field(
        "open_batch", open_batch, "has_heatmap_target", open_size, device, True
    )
    open_heatmap_is_cached = _optional_bool_field(
        "open_batch", open_batch, "heatmap_is_cached", open_size, device, False
    )
    open_action = open_batch.get("action")
    if open_action is not None:
        _require_batch_size("open_batch['action']", open_action, open_size)
        _require_tail_shape(
            "open_batch['action']",
            open_action,
            "robot_batch['action']",
            robot_action,
        )
    open_action = torch.zeros(
        (open_size,) + robot_action.shape[1:],
        device=robot_action.device,
        dtype=robot_action.dtype,
    )

    obs = _concat_obs(robot_batch["obs"], open_batch["obs"])
    gaze_xy = torch.cat(
        [
            robot_gaze_xy,
            open_gaze_xy,
        ],
        dim=0,
    )
    action = torch.cat([robot_action, open_action], dim=0)
    heatmap = torch.cat(
        [
            robot_batch["heatmap"],
            open_batch["heatmap"].to(
                device=robot_batch["heatmap"].device,
                dtype=robot_batch["heatmap"].dtype,
            ),
        ],
        dim=0,
    )

    open_dropout = torch.ones(open_size, device=device, dtype=torch.bool)

    is_open = torch.cat(
        [
            torch.zeros(robot_size, device=device, dtype=torch.bool),
            torch.ones(open_size, device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    has_action = ~is_open
    has_heatmap = torch.cat([robot_has_heatmap, open_has_heatmap_target], dim=0)
    has_heatmap_target = torch.cat(
        [robot_has_heatmap_target, open_has_heatmap_target], dim=0
    )
    heatmap_is_cached = torch.cat(
        [robot_heatmap_is_cached, open_heatmap_is_cached], dim=0
    )
    heatmap = _zero_rows_where_mask_false("mixed_batch['heatmap']", heatmap, has_heatmap)
    has_gaze_label = torch.cat([robot_has_gaze_label, open_has_gaze_label], dim=0)
    has_gaze_condition = torch.cat(
        [robot_has_gaze_condition, open_has_gaze_condition], dim=0
    )
    use_gaze_condition = torch.cat(
        [
            robot_uses_gaze_condition,
            torch.zeros(open_size, device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    is_gaze_condition_dropped = ~use_gaze_condition

    batch = {
        "obs": obs,
        "action": action,
        "heatmap": heatmap,
        "gaze_xy": gaze_xy,
        "is_open": is_open,
        "has_action": has_action,
        "has_heatmap": has_heatmap,
        "has_heatmap_target": has_heatmap_target,
        "heatmap_is_cached": heatmap_is_cached,
        "has_gaze_condition": has_gaze_condition,
        "has_gaze_label": has_gaze_label,
        "use_gaze_condition": use_gaze_condition,
        "is_gaze_condition_dropped": is_gaze_condition_dropped,
    }

    for optional_key in ("action_abs", "action_base_abs", "heatmap_image"):
        robot_value = robot_batch.get(optional_key)
        open_value = open_batch.get(optional_key)
        is_action_metadata = optional_key in ("action_abs", "action_base_abs")
        if robot_value is None and open_value is None:
            continue
        if robot_value is None:
            _require_batch_size(f"open_batch[{optional_key!r}]", open_value, open_size)
            robot_value = torch.zeros(
                (robot_size,) + open_value.shape[1:],
                device=open_value.device,
                dtype=open_value.dtype,
            )
            robot_optional_mask = torch.zeros(robot_size, device=device, dtype=torch.bool)
        else:
            _require_batch_size(f"robot_batch[{optional_key!r}]", robot_value, robot_size)
            robot_optional_mask = _optional_presence_mask(
                "robot_batch",
                robot_batch,
                optional_key,
                robot_size,
                device,
                default_value=True,
            )
            robot_value = _zero_rows_where_mask_false(
                f"robot_batch[{optional_key!r}]",
                robot_value,
                robot_optional_mask,
            )
        if open_value is None:
            open_value = torch.zeros(
                (open_size,) + robot_value.shape[1:],
                device=robot_value.device,
                dtype=robot_value.dtype,
            )
            open_optional_mask = torch.zeros(open_size, device=device, dtype=torch.bool)
        else:
            _require_batch_size(f"open_batch[{optional_key!r}]", open_value, open_size)
            _require_tail_shape(
                f"open_batch[{optional_key!r}]",
                open_value,
                f"robot_batch[{optional_key!r}]",
                robot_value,
            )
            if is_action_metadata:
                open_value = torch.zeros(
                    (open_size,) + robot_value.shape[1:],
                    device=robot_value.device,
                    dtype=robot_value.dtype,
                )
                open_optional_mask = torch.zeros(open_size, device=device, dtype=torch.bool)
            else:
                open_optional_mask = _optional_presence_mask(
                    "open_batch",
                    open_batch,
                    optional_key,
                    open_size,
                    device,
                    default_value=True,
                )
                open_value = _zero_rows_where_mask_false(
                    f"open_batch[{optional_key!r}]",
                    open_value,
                    open_optional_mask,
                )
        batch[optional_key] = torch.cat(
            [robot_value, open_value.to(device=robot_value.device, dtype=robot_value.dtype)],
            dim=0,
        )
        batch[f"has_{optional_key}"] = torch.cat(
            [robot_optional_mask, open_optional_mask],
            dim=0,
        )

    if shuffle:
        batch = _shuffle_batch(batch, generator=generator)
    return batch
