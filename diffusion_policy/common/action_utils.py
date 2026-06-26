import numpy as np

from diffusion_policy.common.pose_util import mat_to_pose10d, pose10d_to_mat


def _split_action_dims(action_dim: int):
    if action_dim == 9:
        return np.arange(9), None
    if action_dim == 10:
        return np.arange(9), 9
    raise NotImplementedError(f"Unsupported action dimension: {action_dim}")


def _as_finite_float_action_array(name: str, value: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric action array.") from exc
    if array.ndim < 1:
        raise ValueError(f"{name} must have a trailing action dimension, got scalar.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert pose9d/10d arrays to matrices while supporting arbitrary leading dims."""
    pose = np.asarray(pose)
    leading_shape = pose.shape[:-1]
    flat_pose = pose.reshape(-1, pose.shape[-1])
    flat_mat = pose10d_to_mat(flat_pose)
    return flat_mat.reshape(leading_shape + (4, 4))


def _mat_to_pose(mat: np.ndarray) -> np.ndarray:
    """Convert matrices to pose9d arrays while supporting arbitrary leading dims."""
    mat = np.asarray(mat)
    leading_shape = mat.shape[:-2]
    flat_mat = mat.reshape(-1, 4, 4)
    flat_pose = mat_to_pose10d(flat_mat)
    return flat_pose.reshape(leading_shape + (9,))


def _expand_base_to_actions(base_mat: np.ndarray, action_mat: np.ndarray) -> np.ndarray:
    """Broadcast base matrices over action horizon dimensions."""
    while base_mat.ndim < action_mat.ndim:
        base_mat = np.expand_dims(base_mat, axis=-3)
    return base_mat


def absolute_actions_to_relative_actions(
    actions: np.ndarray,
    base_absolute_action: np.ndarray,
    action_representation: str = "relative",
    gripper_mode: str = "absolute",
) -> np.ndarray:
    """Convert absolute 9D/10D TCP actions to the latest-observed-TCP relative frame."""
    if action_representation != "relative":
        raise NotImplementedError("Only action_representation='relative' is supported.")
    if gripper_mode != "absolute":
        raise NotImplementedError("Only gripper_mode='absolute' is supported.")

    actions = _as_finite_float_action_array("actions", actions).copy()
    base_absolute_action = _as_finite_float_action_array(
        "base_absolute_action",
        base_absolute_action,
    )
    tcp_dim, gripper_dim = _split_action_dims(actions.shape[-1])
    if base_absolute_action.shape[-1] not in (len(tcp_dim), actions.shape[-1]):
        raise ValueError(
            "base_absolute_action must contain TCP dims or full action dims, "
            f"got {base_absolute_action.shape[-1]} for action dim {actions.shape[-1]}."
        )

    action_tcp = actions[..., tcp_dim]
    base_tcp = base_absolute_action[..., tcp_dim] if base_absolute_action.shape[-1] == actions.shape[-1] else base_absolute_action
    base_mat = _pose_to_mat(base_tcp)
    action_mat = _pose_to_mat(action_tcp)
    base_inv = _expand_base_to_actions(np.linalg.inv(base_mat), action_mat)
    rel_mat = base_inv @ action_mat
    actions[..., tcp_dim] = _mat_to_pose(rel_mat)
    if gripper_dim is not None:
        actions[..., gripper_dim] = np.asarray(actions)[..., gripper_dim]
    return actions


def relative_actions_to_absolute_actions(
    actions: np.ndarray,
    base_absolute_action: np.ndarray,
    action_representation: str = "relative",
    gripper_mode: str = "absolute",
) -> np.ndarray:
    """Convert relative 9D/10D TCP actions back to absolute TCP commands."""
    if action_representation != "relative":
        raise NotImplementedError("Only action_representation='relative' is supported.")
    if gripper_mode != "absolute":
        raise NotImplementedError("Only gripper_mode='absolute' is supported.")

    actions = _as_finite_float_action_array("actions", actions).copy()
    base_absolute_action = _as_finite_float_action_array(
        "base_absolute_action",
        base_absolute_action,
    )
    tcp_dim, gripper_dim = _split_action_dims(actions.shape[-1])
    if base_absolute_action.shape[-1] not in (len(tcp_dim), actions.shape[-1]):
        raise ValueError(
            "base_absolute_action must contain TCP dims or full action dims, "
            f"got {base_absolute_action.shape[-1]} for action dim {actions.shape[-1]}."
        )

    rel_tcp = actions[..., tcp_dim]
    base_tcp = base_absolute_action[..., tcp_dim] if base_absolute_action.shape[-1] == actions.shape[-1] else base_absolute_action
    base_mat = _pose_to_mat(base_tcp)
    rel_mat = _pose_to_mat(rel_tcp)
    base_mat = _expand_base_to_actions(base_mat, rel_mat)
    abs_mat = base_mat @ rel_mat
    actions[..., tcp_dim] = _mat_to_pose(abs_mat)
    if gripper_dim is not None:
        actions[..., gripper_dim] = np.asarray(actions)[..., gripper_dim]
    return actions
