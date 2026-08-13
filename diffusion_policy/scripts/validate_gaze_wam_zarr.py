from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Sequence

SUPPORTED_IMAGE_RESIZE_MODES = ("stretch",)


def _ensure_validator_runtime():
    global GazeWamOpenDataset
    global GazeWamRobotDataset
    global as_optional_gaze_wam_key
    global check_gaze_bounds
    global np
    global relative_actions_to_absolute_actions
    global zarr
    try:
        return zarr
    except NameError:
        import numpy as _np
        import zarr as _zarr
        from diffusion_policy.common.action_utils import (
            relative_actions_to_absolute_actions as _relative_actions_to_absolute_actions,
        )
        from diffusion_policy.common.gaze_utils import (
            as_optional_gaze_wam_key as _as_optional_gaze_wam_key,
            check_gaze_bounds as _check_gaze_bounds,
        )
        from diffusion_policy.dataset.gaze_wam_dataset import (
            GazeWamOpenDataset as _GazeWamOpenDataset,
            GazeWamRobotDataset as _GazeWamRobotDataset,
        )

        GazeWamOpenDataset = _GazeWamOpenDataset
        GazeWamRobotDataset = _GazeWamRobotDataset
        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        check_gaze_bounds = _check_gaze_bounds
        np = _np
        relative_actions_to_absolute_actions = _relative_actions_to_absolute_actions
        zarr = _zarr
        return zarr


def _open_root(path: str):
    _ensure_validator_runtime()
    if str(path).endswith(".zip"):
        store = zarr.ZipStore(path, mode="r")
        return zarr.group(store=store), store
    return zarr.open(path, mode="r"), None


def _resolve_groups(root):
    _ensure_validator_runtime()
    if "data" in root:
        data = root["data"]
        meta = root.get("meta", None)
        if meta is None or "episode_ends" not in meta:
            raise ValueError("Expected meta/episode_ends in diffusion-policy style zarr.")
        return data, np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if "episode_ends" not in root:
        raise ValueError("Expected either data/meta/episode_ends or flat episode_ends.")
    return root, np.asarray(root["episode_ends"][:], dtype=np.int64)


def _shape(array) -> List[int]:
    return [int(v) for v in array.shape]


def _jsonable_attr(value):
    _ensure_validator_runtime()
    if isinstance(value, np.ndarray):
        return _jsonable_attr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable_attr(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_attr(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metadata_attrs(root) -> Dict[str, object]:
    attrs = root["meta"].attrs if "meta" in root else root.attrs
    return {str(key): _jsonable_attr(value) for key, value in attrs.items()}


def _metadata_image_size_pair(value, errors: List[str]) -> Optional[tuple]:
    if value is None:
        return None
    if isinstance(value, str):
        errors.append(f"Zarr metadata image_size must be a pair of integers, got {value!r}.")
        return None
    try:
        values = list(value)
    except TypeError:
        errors.append(f"Zarr metadata image_size must be a pair of integers, got {value!r}.")
        return None
    if len(values) != 2:
        errors.append(f"Zarr metadata image_size must contain exactly two integers, got {value!r}.")
        return None
    try:
        pair = (int(values[0]), int(values[1]))
    except (TypeError, ValueError):
        errors.append(f"Zarr metadata image_size must be a pair of integers, got {value!r}.")
        return None
    if pair[0] <= 0 or pair[1] <= 0:
        errors.append(f"Zarr metadata image_size dimensions must be positive, got {value!r}.")
        return None
    return pair


def _check(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_positive_int_arg(name: str, value, errors: List[str]) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a positive integer, got {value!r}.")
        return None
    if value <= 0:
        errors.append(f"{name} must be a positive integer, got {value}.")
    return value


def _validate_nonnegative_int_arg(name: str, value, errors: List[str]) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a non-negative integer, got {value!r}.")
        return None
    if value < 0:
        errors.append(f"{name} must be a non-negative integer, got {value}.")
    return value


def _validate_positive_int_pair_arg(name: str, value, errors: List[str]) -> Optional[tuple]:
    try:
        if len(value) != 2:
            errors.append(f"{name} must be a pair of positive integers, got {value}.")
            return None
        return (
            _validate_positive_int_arg(f"{name}[0]", value[0], errors),
            _validate_positive_int_arg(f"{name}[1]", value[1], errors),
        )
    except TypeError:
        errors.append(f"{name} must be a pair of positive integers, got {value!r}.")
        return None


def _validate_episode_ends(episode_ends: np.ndarray, n_steps: int, errors: List[str]) -> None:
    _ensure_validator_runtime()
    _check(episode_ends.ndim == 1, f"episode_ends must be 1D, got {episode_ends.shape}.", errors)
    _check(len(episode_ends) > 0, "episode_ends must contain at least one episode.", errors)
    if len(episode_ends) > 0:
        _check(int(episode_ends[-1]) == int(n_steps), f"episode_ends[-1]={episode_ends[-1]} must equal n_steps={n_steps}.", errors)
    if len(episode_ends) > 1:
        _check(np.all(np.diff(episode_ends) > 0), "episode_ends must be strictly increasing.", errors)
    _check(np.all(episode_ends > 0), "episode_ends values must be positive.", errors)


def _episode_length_summary(
    episode_ends: np.ndarray,
    n_obs_steps: int,
    action_horizon: int,
    n_latency_steps: int,
    warnings: List[str],
) -> Dict[str, object]:
    _ensure_validator_runtime()
    if episode_ends.ndim != 1 or len(episode_ends) == 0:
        return {
            "lengths": [],
            "min": None,
            "max": None,
            "mean": None,
            "required_obs_steps": int(n_obs_steps),
            "required_action_steps": int(action_horizon + n_latency_steps),
            "num_short_for_obs": 0,
            "num_short_for_action": 0,
            "num_unpadded_action_starts": 0,
        }
    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    lengths = (episode_ends.astype(np.int64) - starts).astype(np.int64)
    required_obs = int(n_obs_steps)
    required_action = int(action_horizon + n_latency_steps)
    num_short_obs = int(np.sum(lengths < required_obs))
    num_short_action = int(np.sum(lengths < required_action))
    unpadded_action_starts = np.maximum(lengths - required_action + 1, 0)
    if num_short_obs > 0:
        warnings.append(
            f"{num_short_obs} episode(s) are shorter than n_obs_steps={required_obs}; "
            "dataset padding will repeat boundary observations."
        )
    if num_short_action > 0:
        warnings.append(
            f"{num_short_action} episode(s) are shorter than action_horizon+n_latency_steps="
            f"{required_action}; dataset padding will repeat future actions."
        )
    if int(unpadded_action_starts.sum()) == 0:
        warnings.append(
            "No unpadded action chunks are available under the current action_horizon/"
            "n_latency_steps; training can run with padding but real-data coverage is weak."
        )
    return {
        "lengths": [int(v) for v in lengths.tolist()],
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "required_obs_steps": required_obs,
        "required_action_steps": required_action,
        "num_short_for_obs": num_short_obs,
        "num_short_for_action": num_short_action,
        "num_unpadded_action_starts": int(unpadded_action_starts.sum()),
    }


def _numeric_summary(
    array,
    key: str,
    errors: List[str],
    chunk_size: int = 4096,
    max_chunks: Optional[int] = None,
) -> Dict[str, object]:
    _ensure_validator_runtime()
    if not np.issubdtype(array.dtype, np.number):
        summary = {
            "key": key,
            "shape": _shape(array),
            "dtype": str(array.dtype),
            "finite": False,
            "min": None,
            "max": None,
            "mean": None,
            "num_values": int(np.prod(array.shape, dtype=np.int64)),
            "num_nonfinite": None,
            "sampled": False,
        }
        errors.append(f"{key} must be numeric, got dtype {array.dtype}.")
        return summary
    summary = {
        "key": key,
        "shape": _shape(array),
        "dtype": str(array.dtype),
        "finite": True,
        "min": None,
        "max": None,
        "mean": None,
        "num_values": 0,
        "num_nonfinite": 0,
        "sampled": False,
    }
    if max_chunks is not None and int(max_chunks) <= 0:
        raise ValueError("max_chunks must be positive when provided.")
    first_axis = int(array.shape[0]) if len(array.shape) > 0 else 1
    total_chunks = int(np.ceil(first_axis / int(chunk_size))) if first_axis > 0 else 0
    chunk_indices = range(total_chunks)
    if max_chunks is not None and total_chunks > int(max_chunks):
        chunk_indices = sorted(
            set(
                np.linspace(
                    0,
                    total_chunks - 1,
                    num=int(max_chunks),
                    dtype=np.int64,
                ).tolist()
            )
        )
        summary["sampled"] = True
        summary["sampled_chunks"] = len(chunk_indices)
        summary["total_chunks"] = total_chunks
    total = 0
    finite_total = 0
    nonfinite_total = 0
    running_sum = 0.0
    min_value = None
    max_value = None
    for chunk_idx in chunk_indices:
        start = int(chunk_idx) * int(chunk_size)
        stop = min(start + int(chunk_size), first_axis)
        values = np.asarray(array[start:stop])
        values = values.astype(np.float64, copy=False)
        total += int(values.size)
        finite = np.isfinite(values)
        finite_values = values[finite]
        finite_total += int(finite_values.size)
        nonfinite_total += int(values.size - finite_values.size)
        if finite_values.size == 0:
            continue
        running_sum += float(finite_values.sum())
        this_min = float(finite_values.min())
        this_max = float(finite_values.max())
        min_value = this_min if min_value is None else min(min_value, this_min)
        max_value = this_max if max_value is None else max(max_value, this_max)
    summary["num_values"] = (
        int(np.prod(array.shape, dtype=np.int64)) if summary["sampled"] else int(total)
    )
    if summary["sampled"]:
        summary["sampled_num_values"] = int(total)
    summary["num_nonfinite"] = int(nonfinite_total)
    summary["finite"] = nonfinite_total == 0
    if finite_total > 0:
        summary["min"] = float(min_value)
        summary["max"] = float(max_value)
        summary["mean"] = float(running_sum / finite_total)
    if nonfinite_total > 0:
        errors.append(f"{key} must contain only finite values; found {nonfinite_total} non-finite value(s).")
    return summary


def _image_health_summary(image, camera_key: str, errors: List[str], warnings: List[str]) -> Optional[Dict[str, object]]:
    if image is None:
        return None
    summary = _numeric_summary(image, camera_key, errors, chunk_size=16, max_chunks=64)
    if image.ndim == 4:
        if image.shape[-1] in (1, 3, 4):
            layout = "NHWC"
            channel_count = int(image.shape[-1])
            height = int(image.shape[1])
            width = int(image.shape[2])
        elif image.shape[1] in (1, 3, 4):
            layout = "NCHW"
            channel_count = int(image.shape[1])
            height = int(image.shape[2])
            width = int(image.shape[3])
        else:
            layout = "unknown"
            channel_count = None
            height = None
            width = None
        summary.update(
            {
                "layout": layout,
                "channel_count": channel_count,
                "height": height,
                "width": width,
            }
        )
    min_value = summary.get("min")
    max_value = summary.get("max")
    if min_value is not None and max_value is not None:
        if min_value < 0.0:
            warnings.append(f"{camera_key} has negative pixel values; expected uint8 [0,255] or float [0,1].")
        if max_value <= 1.5:
            summary["range_kind"] = "float_0_1_like"
        elif max_value <= 255.0:
            summary["range_kind"] = "uint8_0_255_like"
        else:
            summary["range_kind"] = "out_of_expected_image_range"
            warnings.append(
                f"{camera_key} max={max_value:.6g} exceeds 255; check image decoding or normalization."
            )
    return summary


def _validate_image(data, camera_key: str, errors: List[str]):
    if camera_key not in data:
        errors.append(f"Missing image key '{camera_key}'.")
        return 0, None
    image = data[camera_key]
    _check(image.ndim == 4, f"{camera_key} must be [N,H,W,C] or [N,C,H,W], got {image.shape}.", errors)
    if image.ndim == 4:
        channel_ok = image.shape[-1] in (1, 3, 4) or image.shape[1] in (1, 3, 4)
        _check(channel_ok, f"{camera_key} must have 1/3/4 channels, got {image.shape}.", errors)
        return int(image.shape[0]), image
    return 0, image


def _validate_array_length(data, key: str, n_steps: int, errors: List[str]) -> Optional[object]:
    if key not in data:
        errors.append(f"Missing key '{key}'.")
        return None
    array = data[key]
    _check(int(array.shape[0]) == int(n_steps), f"{key}.shape[0]={array.shape[0]} must equal n_steps={n_steps}.", errors)
    return array


def _iter_first_axis_chunks(array, chunk_size: int = 4096):
    _ensure_validator_runtime()
    length = int(array.shape[0]) if array.shape else 1
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        yield start, stop, np.asarray(array[start:stop])


def _validate_finite_array(array, key: str, errors: List[str], chunk_size: int = 4096) -> None:
    for start, stop, values in _iter_first_axis_chunks(array, chunk_size=chunk_size):
        if not np.issubdtype(values.dtype, np.number):
            errors.append(f"{key} must be numeric, got dtype {array.dtype}.")
            return
        finite = np.isfinite(values)
        if not bool(finite.all()):
            bad = np.argwhere(~finite)
            first = bad[0].tolist() if bad.size > 0 else []
            if first:
                first[0] += start
            errors.append(f"{key} must contain only finite values; first bad index {first}.")
            return


def _validate_nonnegative_array(array, key: str, errors: List[str], chunk_size: int = 4096) -> None:
    for start, stop, values in _iter_first_axis_chunks(array, chunk_size=chunk_size):
        if not np.issubdtype(values.dtype, np.number):
            errors.append(f"{key} must be numeric, got dtype {array.dtype}.")
            return
        if values.size == 0:
            continue
        if float(np.min(values)) < 0.0:
            bad = np.argwhere(values < 0)
            first = bad[0].tolist() if bad.size > 0 else []
            if first:
                first[0] += start
            errors.append(f"{key} must be non-negative; first negative index {first}.")
            return


def _heatmap_spatial_shape(heatmap) -> Optional[tuple]:
    if heatmap.ndim == 3:
        return int(heatmap.shape[1]), int(heatmap.shape[2])
    if heatmap.ndim == 4 and heatmap.shape[-1] == 1:
        return int(heatmap.shape[1]), int(heatmap.shape[2])
    if heatmap.ndim == 4 and heatmap.shape[1] == 1:
        return int(heatmap.shape[2]), int(heatmap.shape[3])
    return None


def _validate_heatmap_array(
    heatmap,
    heatmap_key: str,
    n_steps: int,
    image_size: Sequence[int],
    metadata_attrs: Dict[str, object],
    errors: List[str],
) -> Dict[str, object]:
    _check(
        int(heatmap.shape[0]) == int(n_steps),
        f"{heatmap_key}.shape[0]={heatmap.shape[0]} must equal n_steps={n_steps}.",
        errors,
    )
    heatmap_summary = _numeric_summary(heatmap, heatmap_key, errors)
    heatmap_storage = str(metadata_attrs.get("heatmap_storage", "dense")).strip().lower()
    token_grid = metadata_attrs.get("heatmap_token_grid_for_sigma", [16, 16])
    try:
        num_tokens = int(token_grid[0]) * int(token_grid[1])
    except (TypeError, ValueError, IndexError):
        num_tokens = 256
    is_token_heatmap = (
        heatmap_storage == "token"
        or (heatmap.ndim == 3 and int(heatmap.shape[1]) == num_tokens and int(heatmap.shape[2]) == 1)
    )
    heatmap_summary["storage"] = "token" if is_token_heatmap else "dense"
    if is_token_heatmap:
        _check(
            heatmap.ndim == 3 and int(heatmap.shape[1]) == num_tokens and int(heatmap.shape[2]) == 1,
            f"{heatmap_key} token storage must be [N,{num_tokens},1], got {heatmap.shape}.",
            errors,
        )
    else:
        _check(
            heatmap.ndim in (3, 4),
            f"{heatmap_key} must be [N,H,W], [N,H,W,1], [N,1,H,W], or token [N,T,1], got {heatmap.shape}.",
            errors,
        )
        spatial_shape = _heatmap_spatial_shape(heatmap)
        _check(
            spatial_shape is not None,
            f"{heatmap_key} must be single-channel when 4D, got {heatmap.shape}.",
            errors,
        )
        if spatial_shape is not None:
            _check(
                tuple(spatial_shape) == tuple(image_size),
                f"{heatmap_key} spatial shape {spatial_shape} must match image_size={tuple(image_size)}.",
                errors,
            )
    _validate_finite_array(heatmap, heatmap_key, errors)
    _validate_nonnegative_array(heatmap, heatmap_key, errors)
    return heatmap_summary


def _validate_optional_presence_mask(data, key: str, n_steps: int, errors: List[str]) -> Optional[Dict[str, object]]:
    _ensure_validator_runtime()
    if key not in data:
        return None
    mask = data[key]
    _check(mask.ndim in (1, 2), f"{key} must be [N] or [N,1], got {mask.shape}.", errors)
    _check(
        int(mask.shape[0]) == int(n_steps),
        f"{key}.shape[0]={mask.shape[0]} must equal n_steps={n_steps}.",
        errors,
    )
    if mask.ndim == 2:
        _check(mask.shape[-1] == 1, f"{key} must be [N] or [N,1], got {mask.shape}.", errors)
    values = np.asarray(mask[:]).reshape(mask.shape[0], -1)
    summary = {
        "key": key,
        "shape": _shape(mask),
        "dtype": str(mask.dtype),
        "true_count": None,
        "false_count": None,
    }
    if values.shape[1] != 1:
        return summary
    if np.issubdtype(values.dtype, np.bool_):
        bool_values = values[:, 0].astype(bool)
    elif np.issubdtype(values.dtype, np.number):
        finite = np.isfinite(values[:, 0])
        if not bool(finite.all()):
            errors.append(f"{key} must contain only finite 0/1 values.")
            return summary
        valid = (values[:, 0] == 0) | (values[:, 0] == 1)
        if not bool(valid.all()):
            errors.append(f"{key} must contain only boolean or 0/1 values.")
            return summary
        bool_values = values[:, 0].astype(bool)
    else:
        errors.append(f"{key} must be boolean or numeric 0/1, got dtype {mask.dtype}.")
        return summary
    summary["true_count"] = int(bool_values.sum())
    summary["false_count"] = int((~bool_values).sum())
    return summary


def _validate_gaze_label_presence_consistency(
    presence_summaries: Dict[str, Dict[str, object]],
    has_gaze: bool,
    has_heatmap: bool,
    gaze_key: Optional[str],
    heatmap_key: Optional[str],
    errors: List[str],
) -> None:
    summary = presence_summaries.get("has_gaze_label")
    if not isinstance(summary, dict):
        return
    true_count = summary.get("true_count")
    false_count = summary.get("false_count")
    if isinstance(true_count, int) and true_count > 0 and not has_gaze:
        errors.append(
            "has_gaze_label marks "
            f"{true_count} row(s) with point gaze labels, but point gaze key "
            f"{gaze_key!r} is missing."
        )
    if isinstance(false_count, int) and false_count > 0 and not has_heatmap:
        errors.append(
            "has_gaze_label marks "
            f"{false_count} dense-heatmap-only row(s), but dense heatmap key "
            f"{heatmap_key!r} is missing."
        )


def _validate_gaze_array(gaze, gaze_key: str, errors: List[str]) -> None:
    _ensure_validator_runtime()
    _check(gaze.ndim == 2 and gaze.shape[-1] == 2, f"{gaze_key} must be [N,2], got {gaze.shape}.", errors)
    if gaze.ndim != 2 or gaze.shape[-1] != 2:
        return
    values = np.asarray(gaze[:], dtype=np.float32)
    for idx, point in enumerate(values):
        try:
            check_gaze_bounds(point, policy="error", row_idx=idx)
        except ValueError as exc:
            errors.append(f"{gaze_key}: {exc}")
            break


def _as_optional_key(value: Optional[str]) -> Optional[str]:
    _ensure_validator_runtime()
    return as_optional_gaze_wam_key(value)


def _validate_timestamp_array(
    data,
    key: Optional[str],
    n_steps: int,
    errors: List[str],
    require: bool = False,
    strictly_increasing: bool = False,
    episode_ends: Optional[np.ndarray] = None,
):
    _ensure_validator_runtime()
    key = _as_optional_key(key)
    if key is None:
        return None
    if key not in data:
        if require:
            errors.append(f"Missing timestamp key '{key}'.")
        return None
    timestamp = data[key]
    _check(
        int(timestamp.shape[0]) == int(n_steps),
        f"{key}.shape[0]={timestamp.shape[0]} must equal n_steps={n_steps}.",
        errors,
    )
    values = np.asarray(timestamp[:], dtype=np.float64).reshape(timestamp.shape[0], -1)
    if values.shape[1] != 1:
        errors.append(f"{key} must be [N] or [N,1], got {timestamp.shape}.")
        return None
    values = values[:, 0]
    if not np.all(np.isfinite(values)):
        errors.append(f"{key} must contain only finite timestamps.")
        return values
    starts = np.concatenate([[0], episode_ends[:-1]]) if episode_ends is not None else [0]
    ends = episode_ends if episode_ends is not None else [len(values)]
    for episode_index, (start, end) in enumerate(zip(starts, ends)):
        diffs = np.diff(values[int(start):int(end)])
        if diffs.size == 0:
            continue
        if strictly_increasing:
            monotonic = np.all(diffs > 0)
            relation = "strictly increasing"
        else:
            monotonic = np.all(diffs >= 0)
            relation = "nondecreasing"
        if not monotonic:
            errors.append(f"{key} must be {relation} within episode {episode_index}.")
            break
    return values


def _summarize_timestamp_alignment(
    base_key: str,
    base: np.ndarray,
    other_key: str,
    other: Optional[np.ndarray],
    max_delta: Optional[float],
    errors: List[str],
) -> Optional[Dict[str, float]]:
    _ensure_validator_runtime()
    if other is None:
        return None
    delta = np.asarray(other, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    abs_delta = np.abs(delta)
    summary = {
        "max_abs_delta": float(abs_delta.max(initial=0.0)),
        "mean_abs_delta": float(abs_delta.mean()) if abs_delta.size > 0 else 0.0,
    }
    if max_delta is not None and summary["max_abs_delta"] > float(max_delta):
        errors.append(
            f"Timestamp alignment {other_key} vs {base_key} max_abs_delta="
            f"{summary['max_abs_delta']:.6g} exceeds {float(max_delta):.6g}."
        )
    return summary


def _summarize_timestamp_intervals(
    key: str,
    values: np.ndarray,
    max_step: Optional[float],
    errors: List[str],
    episode_ends: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    _ensure_validator_runtime()
    values = np.asarray(values, dtype=np.float64)
    starts = np.concatenate([[0], episode_ends[:-1]]) if episode_ends is not None else [0]
    ends = episode_ends if episode_ends is not None else [len(values)]
    episode_diffs = [
        np.diff(values[int(start):int(end)])
        for start, end in zip(starts, ends)
        if int(end) - int(start) > 1
    ]
    diffs = np.concatenate(episode_diffs) if episode_diffs else np.asarray([], dtype=np.float64)
    if diffs.size == 0:
        return {
            "count": 0,
            "min_step": None,
            "max_step": None,
            "mean_step": None,
            "std_step": None,
        }
    summary = {
        "count": int(diffs.size),
        "min_step": float(diffs.min()),
        "max_step": float(diffs.max()),
        "mean_step": float(diffs.mean()),
        "std_step": float(diffs.std()),
    }
    if max_step is not None and summary["max_step"] > float(max_step):
        errors.append(
            f"Timestamp interval {key} max_step={summary['max_step']:.6g} "
            f"exceeds {float(max_step):.6g}."
        )
    return summary


def _validate_timestamps(
    data,
    n_steps: int,
    errors: List[str],
    warnings: List[str],
    timestamp_key: Optional[str],
    image_timestamp_key: Optional[str],
    robot_state_timestamp_key: Optional[str],
    action_timestamp_key: Optional[str],
    gaze_timestamp_key: Optional[str],
    require_timestamps: bool,
    timestamp_max_delta: Optional[float],
    timestamp_max_step: Optional[float],
    episode_ends: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    requested = [
        timestamp_key,
        image_timestamp_key,
        robot_state_timestamp_key,
        action_timestamp_key,
        gaze_timestamp_key,
    ]
    requested = [_as_optional_key(key) for key in requested]
    any_requested = any(key is not None for key in requested)
    if require_timestamps and not any_requested:
        timestamp_key = "timestamp"
        requested[0] = timestamp_key
    elif timestamp_key is None and "timestamp" in data:
        timestamp_key = "timestamp"
        requested[0] = timestamp_key

    if not any_requested and timestamp_key is None:
        return {"checked": False, "keys": [], "alignment": {}}

    base = _validate_timestamp_array(
        data,
        timestamp_key,
        n_steps,
        errors,
        require=require_timestamps,
        strictly_increasing=False,
        episode_ends=episode_ends,
    )
    timestamp_values = {}
    if base is not None:
        timestamp_values[timestamp_key] = base

    for key in (image_timestamp_key, robot_state_timestamp_key, action_timestamp_key, gaze_timestamp_key):
        key = _as_optional_key(key)
        if key is None or key == timestamp_key:
            continue
        values = _validate_timestamp_array(
            data,
            key,
            n_steps,
            errors,
            require=require_timestamps,
            strictly_increasing=False,
            episode_ends=episode_ends,
        )
        if values is not None:
            timestamp_values[key] = values

    if require_timestamps and not timestamp_values:
        errors.append("Timestamp validation was required but no timestamp arrays were found.")
    elif any_requested and not timestamp_values:
        warnings.append("Timestamp validation requested but no timestamp arrays were found.")

    alignment = {}
    intervals = {}
    for key, values in timestamp_values.items():
        intervals[key] = _summarize_timestamp_intervals(
            key=key,
            values=values,
            max_step=timestamp_max_step,
            errors=errors,
            episode_ends=episode_ends,
        )
    if base is not None:
        for key, values in timestamp_values.items():
            if key == timestamp_key:
                continue
            summary = _summarize_timestamp_alignment(
                base_key=timestamp_key,
                base=base,
                other_key=key,
                other=values,
                max_delta=timestamp_max_delta,
                errors=errors,
            )
            if summary is not None:
                alignment[key] = summary

    return {
        "checked": bool(timestamp_values),
        "keys": sorted(timestamp_values.keys()),
        "base_key": timestamp_key if base is not None else None,
        "intervals": intervals,
        "alignment": alignment,
    }


def _check_robot_action_roundtrip(
    sample: Dict[str, object],
    atol: float,
    errors: List[str],
) -> Optional[float]:
    _ensure_validator_runtime()
    required = ("action", "action_abs", "action_base_abs")
    missing = [key for key in required if key not in sample]
    if missing:
        errors.append(
            "Robot action roundtrip check requires sample keys "
            f"{required}, missing {missing}."
        )
        return None

    action = np.asarray(sample["action"], dtype=np.float64)
    action_abs = np.asarray(sample["action_abs"], dtype=np.float64)
    action_base_abs = np.asarray(sample["action_base_abs"], dtype=np.float64)
    reconstructed = relative_actions_to_absolute_actions(action, action_base_abs)
    max_error = float(np.max(np.abs(reconstructed - action_abs)))
    if not np.isfinite(max_error) or max_error > float(atol):
        errors.append(
            "Robot relative-action roundtrip failed: "
            f"max_error={max_error:.6g} exceeds atol={float(atol):.6g}."
        )
    return max_error


def validate_gaze_wam_zarr(
    dataset_path: str,
    dataset_type: str,
    camera_key: str = "camera0_rgb",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = None,
    action_abs_key: str = "action_abs_tcp",
    tcp_pose_key: str = "tcp_pose_abs",
    gripper_key: str = "gripper_width",
    n_obs_steps: int = 2,
    action_horizon: int = 16,
    n_latency_steps: int = 0,
    image_size: Sequence[int] = (256, 256),
    image_resize_mode: str = "stretch",
    heatmap_token_grid: Sequence[int] = (16, 16),
    heatmap_dim: Optional[int] = None,
    action_dim: int = 10,
    sample_index: int = 0,
    check_dataset_sample: bool = True,
    check_action_roundtrip: bool = True,
    action_roundtrip_atol: float = 1e-4,
    timestamp_key: Optional[str] = None,
    image_timestamp_key: Optional[str] = None,
    robot_state_timestamp_key: Optional[str] = None,
    action_timestamp_key: Optional[str] = None,
    gaze_timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
) -> Dict[str, object]:
    """Validate robot/open zarr files against the Gaze-WAM dataset contract."""
    _ensure_validator_runtime()
    if dataset_type not in ("robot", "open"):
        raise ValueError("dataset_type must be 'robot' or 'open'.")
    if image_resize_mode not in SUPPORTED_IMAGE_RESIZE_MODES:
        raise ValueError(
            "Gaze-WAM validation currently supports only direct stretch resize. "
            f"Got image_resize_mode={image_resize_mode!r}; crop/letterbox modes must remap gaze "
            "coordinates and dense heatmaps before validation."
        )

    errors: List[str] = []
    warnings: List[str] = []
    n_obs_steps_checked = _validate_positive_int_arg("n_obs_steps", n_obs_steps, errors)
    action_horizon_checked = _validate_positive_int_arg("action_horizon", action_horizon, errors)
    n_latency_steps_checked = _validate_nonnegative_int_arg("n_latency_steps", n_latency_steps, errors)
    if n_obs_steps_checked is not None:
        n_obs_steps = n_obs_steps_checked
    if action_horizon_checked is not None:
        action_horizon = action_horizon_checked
    if n_latency_steps_checked is not None:
        n_latency_steps = n_latency_steps_checked
    else:
        n_latency_steps = 0
    image_size_checked = _validate_positive_int_pair_arg("image_size", image_size, errors)
    heatmap_token_grid_checked = _validate_positive_int_pair_arg(
        "heatmap_token_grid",
        heatmap_token_grid,
        errors,
    )
    image_size_valid = image_size_checked is not None and all(
        value is not None and value > 0 for value in image_size_checked
    )
    heatmap_token_grid_valid = heatmap_token_grid_checked is not None and all(
        value is not None and value > 0 for value in heatmap_token_grid_checked
    )
    if image_size_valid:
        image_size = image_size_checked
    if heatmap_token_grid_valid:
        heatmap_token_grid = heatmap_token_grid_checked
    gaze_key = _as_optional_key(gaze_key)
    heatmap_key = _as_optional_key(heatmap_key)
    if heatmap_dim is None:
        geometry_valid = (
            image_size_valid
            and heatmap_token_grid_valid
        )
        if geometry_valid:
            heatmap_dim = int(image_size[0] // heatmap_token_grid[0]) * int(
                image_size[1] // heatmap_token_grid[1]
            )
        else:
            heatmap_dim = 1
    heatmap_dim_checked = _validate_positive_int_arg("heatmap_dim", heatmap_dim, errors)
    action_dim_checked = _validate_positive_int_arg("action_dim", action_dim, errors)
    if heatmap_dim_checked is not None:
        heatmap_dim = heatmap_dim_checked
    sample_index_checked = _validate_nonnegative_int_arg("sample_index", sample_index, errors)
    if action_dim_checked is not None:
        action_dim = action_dim_checked
    if sample_index_checked is not None:
        sample_index = sample_index_checked
    else:
        sample_index = 0
    if dataset_type == "open" and action_dim_checked is not None and action_dim <= 0:
        errors.append(f"Open zarr dummy action_dim must be a positive integer, got {action_dim}.")
    root, store = _open_root(dataset_path)
    try:
        metadata_attrs = _metadata_attrs(root)
        metadata_dataset_type = metadata_attrs.get("dataset_type")
        if (
            metadata_dataset_type is not None
            and str(metadata_dataset_type) != str(dataset_type)
        ):
            errors.append(
                "Zarr metadata dataset_type="
                f"{metadata_dataset_type!r} does not match validation "
                f"dataset_type={dataset_type!r}."
            )
        metadata_image_resize_mode = metadata_attrs.get("image_resize_mode")
        if (
            metadata_image_resize_mode is not None
            and str(metadata_image_resize_mode) != str(image_resize_mode)
        ):
            errors.append(
                "Zarr metadata image_resize_mode="
                f"{metadata_image_resize_mode!r} does not match validation "
                f"image_resize_mode={image_resize_mode!r}."
            )
        metadata_image_size = _metadata_image_size_pair(metadata_attrs.get("image_size"), errors)
        if metadata_image_size is not None and tuple(metadata_image_size) != tuple(image_size):
            errors.append(
                "Zarr metadata image_size="
                f"{list(metadata_image_size)!r} does not match validation "
                f"image_size={list(image_size)!r}."
            )
        data, episode_ends = _resolve_groups(root)
        n_steps, image = _validate_image(data, camera_key, errors)
        _validate_episode_ends(episode_ends, n_steps, errors)
        episode_summary = _episode_length_summary(
            episode_ends=episode_ends,
            n_obs_steps=n_obs_steps,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            warnings=warnings,
        )
        image_summary = _image_health_summary(image, camera_key, errors, warnings)
        presence_summaries = {}
        for presence_key in (
            "has_action_abs",
            "has_action_base_abs",
            "has_heatmap_image",
            "has_gaze_label",
        ):
            presence_summary = _validate_optional_presence_mask(
                data,
                presence_key,
                n_steps,
                errors,
            )
            if presence_summary is not None:
                presence_summaries[presence_key] = presence_summary
        timestamp_summary = _validate_timestamps(
            data=data,
            n_steps=n_steps,
            errors=errors,
            warnings=warnings,
            timestamp_key=timestamp_key,
            image_timestamp_key=image_timestamp_key,
            robot_state_timestamp_key=robot_state_timestamp_key,
            action_timestamp_key=action_timestamp_key,
            gaze_timestamp_key=gaze_timestamp_key,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
            episode_ends=episode_ends,
        )

        if dataset_type == "robot":
            action_abs = _validate_array_length(data, action_abs_key, n_steps, errors)
            tcp_pose = _validate_array_length(data, tcp_pose_key, n_steps, errors)
            gripper = _validate_array_length(data, gripper_key, n_steps, errors)
            has_gaze = gaze_key is not None and gaze_key in data
            has_heatmap = heatmap_key is not None and heatmap_key in data
            _check(
                has_gaze,
                f"Robot zarr must contain normalized point gaze key {gaze_key!r}.",
                errors,
            )
            _validate_gaze_label_presence_consistency(
                presence_summaries=presence_summaries,
                has_gaze=has_gaze,
                has_heatmap=has_heatmap,
                gaze_key=gaze_key,
                heatmap_key=heatmap_key,
                errors=errors,
            )
            action_abs_summary = None
            tcp_pose_summary = None
            gripper_summary = None
            heatmap_summary = None
            if action_abs is not None:
                _check(action_abs.ndim == 2, f"{action_abs_key} must be [N,D], got {action_abs.shape}.", errors)
                _check(action_abs.shape[-1] in (9, 10), f"{action_abs_key} dim must be 9 or 10, got {action_abs.shape[-1]}.", errors)
                action_abs_summary = _numeric_summary(action_abs, action_abs_key, errors)
                _validate_finite_array(action_abs, action_abs_key, errors)
            if tcp_pose is not None:
                _check(tcp_pose.ndim == 2, f"{tcp_pose_key} must be [N,9] or [N,10], got {tcp_pose.shape}.", errors)
                _check(tcp_pose.shape[-1] in (9, 10), f"{tcp_pose_key} dim must be 9 or 10, got {tcp_pose.shape[-1]}.", errors)
                tcp_pose_summary = _numeric_summary(tcp_pose, tcp_pose_key, errors)
                _validate_finite_array(tcp_pose, tcp_pose_key, errors)
            if gripper is not None:
                _check(gripper.ndim in (1, 2), f"{gripper_key} must be [N] or [N,1], got {gripper.shape}.", errors)
                if gripper.ndim == 2:
                    _check(gripper.shape[-1] == 1, f"{gripper_key} must be [N] or [N,1], got {gripper.shape}.", errors)
                gripper_summary = _numeric_summary(gripper, gripper_key, errors)
                _validate_finite_array(gripper, gripper_key, errors)
            if has_gaze:
                gaze = data[gaze_key]
                _check(gaze.shape[0] == n_steps, f"{gaze_key}.shape[0]={gaze.shape[0]} must equal n_steps={n_steps}.", errors)
                _validate_gaze_array(gaze, gaze_key, errors)
            if has_heatmap:
                heatmap_summary = _validate_heatmap_array(
                    data[heatmap_key],
                    heatmap_key,
                    n_steps,
                    image_size,
                    metadata_attrs,
                    errors,
                )

        if dataset_type == "open":
            has_gaze = gaze_key is not None and gaze_key in data
            has_heatmap = heatmap_key is not None and heatmap_key in data
            _check(has_gaze, f"Open zarr must contain normalized point gaze key {gaze_key!r}.", errors)
            _validate_gaze_label_presence_consistency(
                presence_summaries=presence_summaries,
                has_gaze=has_gaze,
                has_heatmap=has_heatmap,
                gaze_key=gaze_key,
                heatmap_key=heatmap_key,
                errors=errors,
            )
            if has_gaze:
                gaze = data[gaze_key]
                _check(gaze.shape[0] == n_steps, f"{gaze_key}.shape[0]={gaze.shape[0]} must equal n_steps={n_steps}.", errors)
                _validate_gaze_array(gaze, gaze_key, errors)
            if has_heatmap:
                heatmap_summary = _validate_heatmap_array(
                    data[heatmap_key],
                    heatmap_key,
                    n_steps,
                    image_size,
                    metadata_attrs,
                    errors,
                )

        sample_summary = None
        if check_dataset_sample and not errors:
            try:
                if dataset_type == "robot":
                    dataset = GazeWamRobotDataset(
                        dataset_path=dataset_path,
                        camera_key=camera_key,
                        gaze_key=gaze_key,
                        heatmap_key=heatmap_key,
                        action_abs_key=action_abs_key,
                        tcp_pose_key=tcp_pose_key,
                        gripper_key=gripper_key,
                        n_obs_steps=n_obs_steps,
                        action_horizon=action_horizon,
                        n_latency_steps=n_latency_steps,
                        image_size=image_size,
                        image_resize_mode=image_resize_mode,
                        heatmap_token_grid=heatmap_token_grid,
                        heatmap_dim=heatmap_dim,
                        action_padding=True,
                    )
                else:
                    dataset = GazeWamOpenDataset(
                        dataset_path=dataset_path,
                        camera_key=camera_key,
                        gaze_key=gaze_key,
                        heatmap_key=heatmap_key,
                        n_obs_steps=n_obs_steps,
                        action_horizon=action_horizon,
                        n_latency_steps=n_latency_steps,
                        action_dim=action_dim,
                        image_size=image_size,
                        image_resize_mode=image_resize_mode,
                        heatmap_token_grid=heatmap_token_grid,
                        heatmap_dim=heatmap_dim,
                        action_padding=True,
                    )
                _check(len(dataset) > 0, "Dataset adapter produced zero samples.", errors)
                if len(dataset) > 0:
                    sample = dataset[min(sample_index, len(dataset) - 1)]
                    sample_summary = {
                        "dataset_len": int(len(dataset)),
                        "obs_shape": _shape(sample["obs"][camera_key]),
                        "action_shape": _shape(sample["action"]),
                        "heatmap_shape": _shape(sample["heatmap"]),
                        "has_gaze_label": bool(sample["has_gaze_label"].item()),
                        "use_gaze_condition": bool(sample["use_gaze_condition"].item()),
                        "is_gaze_condition_dropped": bool(
                            sample["is_gaze_condition_dropped"].item()
                        ),
                    }
                    for optional_key in ("action_abs", "action_base_abs", "heatmap_image"):
                        if optional_key in sample:
                            sample_summary[f"{optional_key}_shape"] = _shape(sample[optional_key])
                    for mask_key in ("has_action_abs", "has_action_base_abs", "has_heatmap_image"):
                        if mask_key in sample:
                            sample_summary[mask_key] = bool(sample[mask_key].item())
                    if "action_base_abs" in sample:
                        sample_summary["action_base_abs_shape"] = _shape(sample["action_base_abs"])
                    if dataset_type == "robot" and check_action_roundtrip:
                        max_error = _check_robot_action_roundtrip(
                            sample=sample,
                            atol=action_roundtrip_atol,
                            errors=errors,
                        )
                        if max_error is not None:
                            sample_summary["action_roundtrip_max_error"] = max_error
                            sample_summary["action_roundtrip_atol"] = float(action_roundtrip_atol)
            except Exception as exc:
                errors.append(f"Dataset adapter sample check failed: {type(exc).__name__}: {exc}")

        summary = {
            "dataset_path": str(dataset_path),
            "dataset_type": dataset_type,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "n_steps": int(n_steps),
            "num_episodes": int(len(episode_ends)) if "episode_ends" in locals() else 0,
            "episode_ends": [int(v) for v in episode_ends.tolist()] if "episode_ends" in locals() else [],
            "episode_lengths": episode_summary if "episode_summary" in locals() else None,
            "keys": sorted(list(data.keys())) if "data" in locals() else [],
            "metadata_attrs": metadata_attrs if "metadata_attrs" in locals() else {},
            "image_resize_mode": image_resize_mode,
            "image_shape": _shape(image) if image is not None else None,
            "image": image_summary if "image_summary" in locals() else None,
            "timestamps": timestamp_summary if "timestamp_summary" in locals() else {"checked": False, "keys": [], "alignment": {}, "intervals": {}},
            "presence_masks": presence_summaries if "presence_summaries" in locals() else {},
            "sample": sample_summary,
        }
        if dataset_type == "robot":
            summary["robot_numeric"] = {
                "action_abs": action_abs_summary if "action_abs_summary" in locals() else None,
                "tcp_pose": tcp_pose_summary if "tcp_pose_summary" in locals() else None,
                "gripper": gripper_summary if "gripper_summary" in locals() else None,
                "heatmap": heatmap_summary if "heatmap_summary" in locals() else None,
            }
        if dataset_type == "open":
            summary["heatmap"] = heatmap_summary if "heatmap_summary" in locals() else None
        return summary
    finally:
        if store is not None:
            store.close()


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Validate a Gaze-WAM robot/open zarr schema.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-type", choices=("robot", "open"), required=True)
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument(
        "--heatmap-key",
        default="none",
        help=(
            "Optional dense/token heatmap key. Use 'none' for the current xy-only "
            "DSNT/JS training contract."
        ),
    )
    parser.add_argument("--action-abs-key", default="action_abs_tcp")
    parser.add_argument("--tcp-pose-key", default="tcp_pose_abs")
    parser.add_argument("--gripper-key", default="gripper_width")
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--n-latency-steps", type=int, default=0)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument(
        "--image-resize-mode",
        choices=SUPPORTED_IMAGE_RESIZE_MODES,
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--heatmap-token-grid", type=int, nargs=2, default=(16, 16), metavar=("H", "W"))
    parser.add_argument(
        "--heatmap-dim",
        type=int,
        default=None,
        help=(
            "Heatmap token channel dimension for dataset sample validation. "
            "Defaults to the patch area implied by --image-size and --heatmap-token-grid."
        ),
    )
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--skip-sample-check", action="store_true")
    parser.add_argument("--skip-action-roundtrip-check", action="store_true")
    parser.add_argument("--action-roundtrip-atol", type=float, default=1e-4)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--image-timestamp-key", default=None)
    parser.add_argument("--robot-state-timestamp-key", default=None)
    parser.add_argument("--action-timestamp-key", default=None)
    parser.add_argument("--gaze-timestamp-key", default=None)
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = validate_gaze_wam_zarr(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        camera_key=args.camera_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        action_abs_key=args.action_abs_key,
        tcp_pose_key=args.tcp_pose_key,
        gripper_key=args.gripper_key,
        n_obs_steps=args.n_obs_steps,
        action_horizon=args.action_horizon,
        n_latency_steps=args.n_latency_steps,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        heatmap_token_grid=args.heatmap_token_grid,
        heatmap_dim=args.heatmap_dim,
        action_dim=args.action_dim,
        sample_index=args.sample_index,
        check_dataset_sample=not args.skip_sample_check,
        check_action_roundtrip=not args.skip_action_roundtrip_check,
        action_roundtrip_atol=args.action_roundtrip_atol,
        timestamp_key=args.timestamp_key,
        image_timestamp_key=args.image_timestamp_key,
        robot_state_timestamp_key=args.robot_state_timestamp_key,
        action_timestamp_key=args.action_timestamp_key,
        gaze_timestamp_key=args.gaze_timestamp_key,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["valid"] or (args.fail_on_warning and summary["warnings"]):
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
