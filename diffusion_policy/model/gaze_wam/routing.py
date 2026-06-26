from typing import Dict

import torch

from diffusion_policy.model.gaze_wam.loss import distributed_mask_count


def _require_bool_vector(name: str, value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be a BoolTensor, got dtype {value.dtype}.")
    if value.shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape [B] with B={batch_size}, got {tuple(value.shape)}."
        )
    return value


def bool_count(mask: torch.Tensor, use_distributed_counts: bool = False) -> int:
    if not torch.is_tensor(mask):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}.")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a BoolTensor, got dtype {mask.dtype}.")
    if use_distributed_counts:
        return int(distributed_mask_count(mask).detach().cpu().item())
    return int(mask.sum().detach().cpu().item())


def loss_routing_summary(
    mixed: Dict[str, torch.Tensor],
    action_loss_mask: torch.Tensor,
    heatmap_loss_mask: torch.Tensor,
    use_distributed_counts: bool = False,
) -> Dict[str, object]:
    is_open = mixed["is_open"]
    if not torch.is_tensor(is_open) or is_open.ndim != 1:
        shape = None if not torch.is_tensor(is_open) else tuple(is_open.shape)
        raise ValueError(f"mixed['is_open'] must be a BoolTensor vector, got {shape}.")
    batch_size = int(is_open.shape[0])
    is_open = _require_bool_vector("mixed['is_open']", is_open, batch_size)
    is_robot = ~is_open
    has_action = _require_bool_vector(
        "mixed['has_action']",
        mixed["has_action"],
        batch_size,
    ).to(device=is_open.device)
    has_heatmap = _require_bool_vector(
        "mixed['has_heatmap']",
        mixed["has_heatmap"],
        batch_size,
    ).to(device=is_open.device)
    has_gaze_label = _require_bool_vector(
        "mixed['has_gaze_label']",
        mixed["has_gaze_label"],
        batch_size,
    ).to(device=is_open.device)
    use_gaze_condition = _require_bool_vector(
        "mixed['use_gaze_condition']",
        mixed["use_gaze_condition"],
        batch_size,
    ).to(device=is_open.device)
    is_gaze_condition_dropped = _require_bool_vector(
        "mixed['is_gaze_condition_dropped']",
        mixed["is_gaze_condition_dropped"],
        batch_size,
    ).to(device=is_open.device)
    action_loss_mask = _require_bool_vector(
        "action_loss_mask",
        action_loss_mask,
        batch_size,
    ).to(device=is_open.device)
    heatmap_loss_mask = _require_bool_vector(
        "heatmap_loss_mask",
        heatmap_loss_mask,
        batch_size,
    ).to(device=is_open.device)
    robot_real_gaze = is_robot & use_gaze_condition
    robot_masked_gaze = is_robot & (~use_gaze_condition)
    open_rows = is_open
    def count(mask: torch.Tensor) -> int:
        return bool_count(mask, use_distributed_counts=use_distributed_counts)

    return {
        "robot_rows": count(is_robot),
        "open_rows": count(open_rows),
        "has_action_rows": count(has_action),
        "has_heatmap_rows": count(has_heatmap),
        "has_gaze_label_rows": count(has_gaze_label),
        "use_gaze_condition_rows": count(use_gaze_condition),
        "dropped_gaze_condition_rows": count(is_gaze_condition_dropped),
        "robot_real_gaze_rows": count(robot_real_gaze),
        "robot_masked_gaze_rows": count(robot_masked_gaze),
        "open_action_loss_count": count(open_rows & action_loss_mask),
        "open_heatmap_loss_count": count(open_rows & heatmap_loss_mask),
        "robot_action_loss_count": count(is_robot & action_loss_mask),
        "robot_heatmap_loss_count": count(is_robot & heatmap_loss_mask),
        "robot_real_gaze_action_loss_count": count(robot_real_gaze & action_loss_mask),
        "robot_real_gaze_heatmap_loss_count": count(robot_real_gaze & heatmap_loss_mask),
        "robot_masked_gaze_action_loss_count": count(robot_masked_gaze & action_loss_mask),
        "robot_masked_gaze_heatmap_loss_count": count(robot_masked_gaze & heatmap_loss_mask),
    }
