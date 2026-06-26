from typing import Optional, Sequence

import numpy as np


def as_optional_gaze_wam_key(value: Optional[str]) -> Optional[str]:
    """Normalize optional zarr key config values.

    Hydra/CLI paths can represent a missing key as None, an empty string, or a
    stringified null-like value. Treat those consistently so dense-heatmap-only
    datasets do not accidentally look for a literal "None" key.
    """
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ("", "none", "null"):
        return None
    return value


def check_gaze_bounds(
    gaze: np.ndarray,
    policy: str = "error",
    row_idx: Optional[int] = None,
    label: str = "gaze point",
) -> Optional[np.ndarray]:
    """Validate a normalized 2D gaze point.

    Args:
        gaze: Candidate normalized ``[x, y]`` gaze point.
        policy: ``error`` raises, ``drop`` returns ``None``, and ``clip`` clips finite
            out-of-frame points into ``[0, 1]``.
        row_idx: Optional row index for error messages.
        label: Human-readable label for error messages.
    """
    if policy not in ("error", "drop", "clip"):
        raise ValueError("gaze_bounds_policy must be one of: error, drop, clip.")

    gaze = np.asarray(gaze, dtype=np.float32).reshape(-1)
    finite_shape = gaze.shape == (2,) and np.all(np.isfinite(gaze))
    if not finite_shape:
        message = f"Invalid {label} {gaze!r}"
    elif np.any(gaze < 0.0) or np.any(gaze > 1.0):
        message = f"Out-of-frame {label} {gaze.tolist()}"
    else:
        return gaze.astype(np.float32)

    if row_idx is not None:
        message = f"{message} at row {row_idx}."
    else:
        message = f"{message}."

    if policy == "clip":
        if not finite_shape:
            raise ValueError(message)
        return np.clip(gaze, 0.0, 1.0).astype(np.float32)
    if policy == "drop":
        return None
    raise ValueError(message)


def gaussian_heatmaps_from_points(
    gaze_xy: np.ndarray,
    image_size: Sequence[int] = (256, 256),
    sigma_px: float = 20.0,
    window_size: int = 1,
    episode_ends: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render trailing-window Gaussian heatmaps from normalized gaze points.

    This helper is intended for zarr conversion/preparation scripts. The policy dataset should
    consume the dense heatmap stored in zarr instead of regenerating labels from ``gaze_xy``.
    """
    gaze_xy = np.asarray(gaze_xy, dtype=np.float32)
    if gaze_xy.ndim != 2 or gaze_xy.shape[-1] != 2:
        raise ValueError(f"gaze_xy must be [N,2], got {gaze_xy.shape}.")
    if not np.all(np.isfinite(gaze_xy)):
        raise ValueError("gaze_xy must contain only finite values.")
    if np.any(gaze_xy < 0.0) or np.any(gaze_xy > 1.0):
        raise ValueError("gaze_xy must be normalized to [0, 1].")

    if len(image_size) != 2:
        raise ValueError(f"image_size must be a pair of (height, width), got {image_size}.")
    image_h, image_w = int(image_size[0]), int(image_size[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"image_size dimensions must be positive, got {image_size}.")
    sigma_px = float(sigma_px)
    if not np.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError(f"sigma_px must be a finite positive value, got {sigma_px}.")
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")

    n_steps = int(gaze_xy.shape[0])
    if episode_ends is None:
        episode_ends = np.asarray([n_steps], dtype=np.int64)
    else:
        episode_ends = np.asarray(episode_ends, dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0 or int(episode_ends[-1]) != n_steps:
        raise ValueError(
            "episode_ends must be 1D and end at the number of gaze rows "
            f"({n_steps}), got {episode_ends!r}."
        )

    y = np.arange(image_h, dtype=np.float32) + 0.5
    x = np.arange(image_w, dtype=np.float32) + 0.5
    yy, xx = np.meshgrid(y, x, indexing="ij")
    out = np.zeros((n_steps, image_h, image_w), dtype=np.float32)

    episode_start = 0
    for episode_end in episode_ends.astype(np.int64).tolist():
        if episode_end <= episode_start:
            raise ValueError("episode_ends must be strictly increasing.")
        for current_idx in range(episode_start, episode_end):
            window_start = max(episode_start, current_idx - window_size + 1)
            heatmap = np.zeros((image_h, image_w), dtype=np.float32)
            for point in gaze_xy[window_start : current_idx + 1]:
                px = float(point[0]) * image_w
                py = float(point[1]) * image_h
                dist = ((xx - px) ** 2 + (yy - py) ** 2) / (sigma_px ** 2)
                heatmap += np.exp(-0.5 * dist).astype(np.float32)
            max_value = float(heatmap.max(initial=0.0))
            if max_value > 0.0:
                heatmap = heatmap / max_value
            out[current_idx] = heatmap.astype(np.float32, copy=False)
        episode_start = int(episode_end)
    return out
