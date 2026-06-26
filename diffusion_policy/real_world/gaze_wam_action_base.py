from typing import Optional

import numpy as np


def _require_finite_array(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_scalar(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}.")


def action_base_abs_to_10d(
    action_base_abs,
    gripper_width: Optional[float] = None,
) -> np.ndarray:
    """Normalize direct deployment action-base inputs to the 10D Gaze-WAM convention."""
    base = np.asarray(action_base_abs, dtype=np.float32)
    _require_finite_array("action_base_abs", base)
    _require_finite_scalar("gripper_width", gripper_width)
    if base.shape[-1] == 10:
        out = base.copy()
    elif base.shape[-1] == 9:
        if gripper_width is None:
            raise ValueError(
                "9D action_base_abs requires gripper_width so deployment can build the "
                "10D Gaze-WAM action base."
            )
        out = np.concatenate(
            [base, np.zeros(base.shape[:-1] + (1,), dtype=np.float32)],
            axis=-1,
        )
    else:
        raise ValueError(
            "action_base_abs must be 9D pose-only or 10D pose+gripper, "
            f"got {base.shape}."
        )
    if gripper_width is not None:
        out[..., 9] = float(gripper_width)
    _require_finite_array("action_base_abs", out)
    return out.astype(np.float32)
