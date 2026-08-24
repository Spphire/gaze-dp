import copy
import hashlib
import json
import pathlib
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from diffusion_policy.common.action_utils import absolute_actions_to_relative_actions
from diffusion_policy.common.gaze_wam_image import image_sequence_to_chw_float
from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key, check_gaze_bounds
from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_nonnegative_int_field,
    normalize_gaze_wam_positive_int_field,
    normalize_gaze_wam_positive_int_sequence,
    normalize_gaze_wam_unit_interval_float_field,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_image_identity_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.gaze_wam.heatmap_codec import HeatmapTokenCodec

SUPPORTED_IMAGE_RESIZE_MODES = ("stretch", "letterbox")
ACTION_TARGET_START_OFFSET_STEPS = 1


def _validate_positive_int(name: str, value: int) -> int:
    return normalize_gaze_wam_positive_int_field(name, value)


def _validate_nonnegative_int(name: str, value: int) -> int:
    return normalize_gaze_wam_nonnegative_int_field(name, value)


def _validate_positive_int_pair(name: str, value: Sequence[int]) -> Tuple[int, int]:
    parsed = normalize_gaze_wam_positive_int_sequence(name, value, length=2)
    return tuple(parsed)


def _validate_val_ratio(value: float) -> float:
    return normalize_gaze_wam_unit_interval_float_field("val_ratio", value)


def _validate_positive_float(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}.")
    return parsed


def _validate_temporal_heatmap_mode(value: str) -> str:
    value = str(value)
    if value not in ("off", "bidirectional", "causal"):
        raise ValueError(
            "temporal_heatmap_mode must be one of: off, bidirectional, causal; "
            f"got {value!r}."
        )
    return value


def _require_finite_array(label: str, value):
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        first = bad[0].tolist() if bad.size > 0 else []
        raise ValueError(f"{label} must contain only finite values; first bad index {first}.")
    return value


def _require_nonnegative_array(label: str, value):
    array = np.asarray(value)
    if array.size > 0 and float(np.min(array)) < 0.0:
        bad = np.argwhere(array < 0.0)
        first = bad[0].tolist() if bad.size > 0 else []
        raise ValueError(f"{label} must be non-negative; first negative index {first}.")
    return value


def _presence_mask_values_to_bool(label: str, value) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.bool_)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"{label} must contain boolean or numeric 0/1 values, got dtype {array.dtype}."
        )
    _require_finite_array(label, array)
    if not np.all((array == 0) | (array == 1)):
        bad = np.argwhere((array != 0) & (array != 1))
        first = bad[0].tolist() if bad.size > 0 else []
        raise ValueError(f"{label} must contain only boolean or numeric 0/1 values; first bad index {first}.")
    return array.astype(np.bool_)


def _presence_mask_values_to_vector(label: str, value) -> np.ndarray:
    mask = _presence_mask_values_to_bool(label, value)
    if mask.ndim == 0:
        return mask.reshape(1)
    if mask.ndim == 1:
        return mask
    if mask.ndim == 2 and mask.shape[-1] == 1:
        return mask[:, 0]
    raise ValueError(f"{label} must be [N] or [N,1] when sampled, got {mask.shape}.")


def _open_zarr_root(dataset_path: str):
    if dataset_path.endswith(".zip"):
        store = zarr.ZipStore(dataset_path, mode="r")
        return zarr.group(store=store), store
    root = zarr.open(dataset_path, mode="r")
    return root, None


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _cache_dataset_name(dataset_path: str) -> str:
    name = pathlib.Path(str(dataset_path).rstrip("/\\")).name
    if name.endswith(".zip"):
        name = name[:-4]
    if name.endswith(".zarr"):
        name = name[:-5]
    return name


class _HeatmapLatentCache:
    """Read a completed rank-sharded offline Cosmos latent cache."""

    def __init__(
        self,
        cache_root: str,
        dataset_path: str,
        source_rows: int,
        episode_ends: np.ndarray,
        token_shape: Tuple[int, int],
        latent_scale: float,
        latent_offset: float,
    ) -> None:
        root = pathlib.Path(str(cache_root)).expanduser()
        dataset_dir = root / _cache_dataset_name(dataset_path)
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Heatmap latent cache manifest not found for {_cache_dataset_name(dataset_path)!r}: "
                f"{manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Heatmap latent cache is not complete: {manifest_path}")
        if int(manifest.get("source_rows", -1)) != int(source_rows):
            raise ValueError(
                f"Heatmap cache source_rows mismatch for {dataset_path}: "
                f"cache={manifest.get('source_rows')}, zarr={source_rows}."
            )
        episode_hash = _array_sha256(np.asarray(episode_ends, dtype=np.int64))
        if manifest.get("source_episode_ends_sha256") != episode_hash:
            raise ValueError(
                f"Heatmap cache episode boundaries do not match {dataset_path}."
            )
        cosmos = manifest.get("cosmos") or {}
        expected_shape = [int(token_shape[0]), int(token_shape[1])]
        if list(cosmos.get("token_shape", [])) != expected_shape:
            raise ValueError(
                f"Heatmap cache token shape mismatch: cache={cosmos.get('token_shape')}, "
                f"dataset={expected_shape}."
            )
        if not np.isclose(float(cosmos.get("latent_scale", np.nan)), float(latent_scale)):
            raise ValueError("Heatmap cache latent_scale does not match the training config.")
        if not np.isclose(float(cosmos.get("latent_offset", np.nan)), float(latent_offset)):
            raise ValueError("Heatmap cache latent_offset does not match the training config.")

        workers = manifest.get("workers") or []
        if not workers:
            raise ValueError(f"Heatmap cache has no worker manifests: {manifest_path}")
        self._ranges = []
        self._latents = []
        self._has = []
        for worker in sorted(workers, key=lambda item: int(item.get("rank", 0))):
            if worker.get("status") != "complete":
                raise ValueError(f"Heatmap cache worker is not complete: {worker}")
            start = int(worker["global_start"])
            end = int(worker["global_end"])
            rank_dir = dataset_dir / f"rank_{int(worker['rank']):05d}"
            latent_path = rank_dir / "heatmap_latent.npy"
            has_path = rank_dir / "has_heatmap.npy"
            latent = np.load(latent_path, mmap_mode="r", allow_pickle=False)
            has = np.load(has_path, mmap_mode="r", allow_pickle=False)
            if tuple(latent.shape) != (end - start, *expected_shape):
                raise ValueError(f"Invalid heatmap cache latent shape: {latent_path} {latent.shape}")
            if tuple(has.shape) != (end - start,):
                raise ValueError(f"Invalid heatmap cache availability shape: {has_path} {has.shape}")
            self._ranges.append((start, end))
            self._latents.append(latent)
            self._has.append(has)
        if self._ranges[0][0] != 0 or self._ranges[-1][1] != int(source_rows):
            raise ValueError("Heatmap cache worker ranges do not cover the complete source dataset.")
        for previous, current in zip(self._ranges, self._ranges[1:]):
            if previous[1] != current[0]:
                raise ValueError("Heatmap cache worker ranges are not contiguous.")
        self.dataset_name = _cache_dataset_name(dataset_path)
        self.token_shape = tuple(token_shape)

    def get(self, global_row: int) -> Tuple[np.ndarray, bool]:
        row = int(global_row)
        for (start, end), latent, has in zip(self._ranges, self._latents, self._has):
            if start <= row < end:
                local = row - start
                return np.asarray(latent[local], dtype=np.float32), bool(has[local])
        raise IndexError(f"Heatmap cache row {row} is outside all worker ranges.")


def _resolve_data_group(root):
    if "data" in root:
        data_group = root["data"]
        if "meta" in root and "episode_ends" in root["meta"]:
            return data_group, np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if "episode_ends" in root:
        return root, np.asarray(root["episode_ends"][:], dtype=np.int64)
    raise KeyError(
        "Expected either a diffusion_policy-style zarr with data/meta/episode_ends "
        "or a flat zarr group containing episode_ends."
    )


def _build_sample_indices(
    episode_ends: np.ndarray,
    action_horizon: int,
    n_latency_steps: int,
    action_downsample_steps: int,
    action_padding: bool,
    episode_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    indices = []
    for episode_idx, episode_end in enumerate(episode_ends):
        if episode_mask is not None and not episode_mask[episode_idx]:
            continue
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        episode_end = int(episode_end)
        for current_idx in range(episode_start, episode_end):
            sampled_horizon = action_horizon + n_latency_steps
            required_end = (
                current_idx
                + ACTION_TARGET_START_OFFSET_STEPS
                + (sampled_horizon - 1) * action_downsample_steps
                + 1
            )
            if (not action_padding) and required_end > episode_end:
                continue
            indices.append((current_idx, episode_start, episode_end, episode_idx))
    if not indices:
        return np.empty((0, 4), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _sample_history(
    array,
    current_idx: int,
    episode_start: int,
    horizon: int,
    downsample_steps: int,
) -> np.ndarray:
    target_idx = np.asarray(
        [current_idx - i * downsample_steps for i in range(horizon - 1, -1, -1)],
        dtype=np.int64,
    )
    target_idx = np.clip(target_idx, episode_start, current_idx)
    return np.asarray(array[target_idx])


def _sample_future(
    array,
    current_idx: int,
    episode_end: int,
    horizon: int,
    downsample_steps: int,
) -> np.ndarray:
    """Sample future rows starting at t+1, never the observation row at t."""
    target_idx = np.asarray(
        [
            current_idx
            + ACTION_TARGET_START_OFFSET_STEPS
            + i * downsample_steps
            for i in range(horizon)
        ],
        dtype=np.int64,
    )
    target_idx = np.minimum(target_idx, episode_end - 1)
    return np.asarray(array[target_idx])


def _splat_gaussian_heatmap(
    heatmap: np.ndarray,
    gaze_xy: np.ndarray,
    weight: float,
    sigma_px: float,
    radius_sigma: float = 3.0,
) -> None:
    height, width = heatmap.shape
    x = float(gaze_xy[0]) * float(width - 1)
    y = float(gaze_xy[1]) * float(height - 1)
    radius = max(1, int(round(float(radius_sigma) * float(sigma_px))))
    x0 = max(0, int(np.floor(x)) - radius)
    x1 = min(width, int(np.floor(x)) + radius + 1)
    y0 = max(0, int(np.floor(y)) - radius)
    y1 = min(height, int(np.floor(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * float(sigma_px) ** 2))
    heatmap[y0:y1, x0:x1] += float(weight) * patch.astype(np.float32)


def _compose_action_with_gripper(action: np.ndarray, gripper: np.ndarray, label: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == 10:
        return action
    if action.shape[-1] != 9:
        raise ValueError(f"{label} must be 9D or 10D, got {action.shape}.")
    gripper = np.asarray(gripper, dtype=np.float32).reshape(action.shape[:-1] + (-1,))
    if gripper.shape[-1] != 1:
        raise ValueError(
            f"{label} requires exactly one gripper scalar to compose 10D action, "
            f"got gripper shape {gripper.shape}."
        )
    return np.concatenate([action, gripper], axis=-1)


def _infer_image_hw(image: np.ndarray) -> Tuple[int, int]:
    if image.ndim != 4:
        raise ValueError(f"Expected image sequence [T,H,W,C] or [T,C,H,W], got {image.shape}.")
    if image.shape[-1] in (1, 3, 4):
        return int(image.shape[1]), int(image.shape[2])
    if image.shape[1] in (1, 3, 4):
        return int(image.shape[2]), int(image.shape[3])
    raise ValueError(f"Cannot infer image height/width for image shape {image.shape}.")


def _image_to_chw_float(
    image: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    return image_sequence_to_chw_float(
        image,
        image_size=image_size,
        name="camera image",
    )


def _heatmap_image_to_float(
    heatmap_image: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    heatmap_image = np.asarray(heatmap_image)
    if heatmap_image.ndim == 2:
        heatmap_image = heatmap_image[None, None]
    elif heatmap_image.ndim == 3:
        if heatmap_image.shape[-1] == 1:
            heatmap_image = np.moveaxis(heatmap_image, -1, 0)[None]
        elif heatmap_image.shape[0] == 1:
            heatmap_image = heatmap_image[None]
        else:
            heatmap_image = heatmap_image[None, None]
    elif heatmap_image.ndim == 4:
        if heatmap_image.shape[-1] == 1:
            heatmap_image = np.moveaxis(heatmap_image, -1, 1)
        elif heatmap_image.shape[1] != 1:
            raise ValueError(
                f"Expected dense heatmap with one channel, got {heatmap_image.shape}."
            )
    else:
        raise ValueError(f"Unsupported heatmap image shape {heatmap_image.shape}.")

    heatmap = heatmap_image.astype(np.float32)
    _require_finite_array("dense gaze heatmap", heatmap)
    _require_nonnegative_array("dense gaze heatmap", heatmap)
    if heatmap.max(initial=0) > 1.5:
        heatmap = heatmap / 255.0
    heatmap_tensor = torch.from_numpy(np.ascontiguousarray(heatmap[:, :1]))
    if image_size is not None and heatmap_tensor.shape[-2:] != tuple(image_size):
        heatmap_tensor = F.interpolate(
            heatmap_tensor,
            size=tuple(image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return heatmap_tensor


def _heatmap_tokens_to_float(
    heatmap_tokens: np.ndarray,
    num_tokens: int,
    heatmap_dim: int = 1,
) -> Optional[torch.Tensor]:
    heatmap_tokens = np.asarray(heatmap_tokens)
    heatmap_dim = _validate_positive_int("heatmap_dim", heatmap_dim)
    if heatmap_tokens.ndim == 1 and heatmap_tokens.shape[0] == int(num_tokens) and heatmap_dim == 1:
        heatmap_tokens = heatmap_tokens[:, None]
    if heatmap_tokens.ndim != 2 or heatmap_tokens.shape != (int(num_tokens), int(heatmap_dim)):
        return None
    tokens = heatmap_tokens.astype(np.float32)
    _require_finite_array("token gaze heatmap", tokens)
    _require_nonnegative_array("token gaze heatmap", tokens)
    if tokens.max(initial=0) > 1.5:
        tokens = tokens / 255.0
    return torch.from_numpy(np.ascontiguousarray(tokens)).to(dtype=torch.float32)


def _looks_like_dense_heatmap_row(
    heatmap_value: np.ndarray,
    image_size: Tuple[int, int],
) -> bool:
    heatmap_value = np.asarray(heatmap_value)
    if heatmap_value.ndim == 2:
        return tuple(heatmap_value.shape) == tuple(image_size)
    if heatmap_value.ndim == 3:
        return (
            heatmap_value.shape[-1] == 1
            and tuple(heatmap_value.shape[:2]) == tuple(image_size)
        ) or (
            heatmap_value.shape[0] == 1
            and tuple(heatmap_value.shape[1:]) == tuple(image_size)
        )
    return False


def _looks_like_scalar_token_heatmap_row(
    heatmap_value: np.ndarray,
    num_tokens: int,
) -> bool:
    heatmap_value = np.asarray(heatmap_value)
    return (
        heatmap_value.ndim == 1
        and heatmap_value.shape[0] == int(num_tokens)
    ) or (
        heatmap_value.ndim == 2
        and heatmap_value.shape == (int(num_tokens), 1)
    )


def _normalize_gaze_xy(
    gaze_xy: np.ndarray,
    source_image_size: Tuple[int, int],
    gaze_is_normalized: bool,
) -> np.ndarray:
    gaze_xy = np.asarray(gaze_xy, dtype=np.float32)
    if not gaze_is_normalized:
        image_h, image_w = source_image_size
        gaze_xy = gaze_xy / np.asarray([image_w, image_h], dtype=np.float32)
    return check_gaze_bounds(gaze_xy.reshape(-1)[:2], policy="error")


def _validate_image_resize_mode(image_resize_mode: str) -> str:
    if image_resize_mode not in SUPPORTED_IMAGE_RESIZE_MODES:
        raise ValueError(
            "Gaze-WAM point-gaze labels support pre-aligned stretch or letterbox inputs. "
            f"Got image_resize_mode={image_resize_mode!r}; other geometry modes must remap "
            "gaze_xy and dense heatmaps before entering this dataset."
        )
    return image_resize_mode


def _make_action_normalizer(actions: np.ndarray):
    actions = np.asarray(actions)
    if actions.size == 0 or actions.shape[0] <= 0:
        raise ValueError("Cannot fit Gaze-WAM action normalizer from zero robot samples.")
    if actions.ndim < 2:
        raise ValueError(f"Gaze-WAM actions must include a feature dimension, got {actions.shape}.")
    action_dim = actions.shape[-1]
    stat = array_to_stats(actions.reshape(-1, action_dim))
    parts = [get_range_normalizer_from_stat({k: v[..., :3] for k, v in stat.items()})]
    parts.append(get_identity_normalizer_from_stat({k: v[..., 3:9] for k, v in stat.items()}))
    if action_dim == 10:
        parts.append(get_range_normalizer_from_stat({k: v[..., 9:10] for k, v in stat.items()}))
    elif action_dim != 9:
        raise NotImplementedError(f"Unsupported Gaze-WAM action dim {action_dim}.")
    return concatenate_normalizer(parts)


class _BaseGazeWamZarrDataset(BaseDataset):
    def __init__(
        self,
        dataset_path: str,
        camera_key: str = "camera0_rgb",
        camera_keys: Optional[Sequence[str]] = None,
        camera_key_map: Optional[Mapping[str, str]] = None,
        gaze_key: Optional[str] = "gaze_xy",
        heatmap_key: Optional[str] = "gaze_heatmap",
        n_obs_steps: int = 2,
        obs_downsample_steps: int = 1,
        action_horizon: int = 48,
        n_latency_steps: int = 0,
        action_downsample_steps: int = 1,
        image_size: Sequence[int] = (256, 256),
        image_resize_mode: str = "stretch",
        heatmap_token_grid: Sequence[int] = (16, 16),
        heatmap_dim: int = 1,
        heatmap_sigma_tokens: float = 1.25,
        gaze_is_normalized: bool = True,
        action_padding: bool = True,
        temporal_heatmap_mode: str = "off",
        temporal_heatmap_window_radius: int = 30,
        temporal_heatmap_beta: float = 10.0,
        temporal_heatmap_sigma_px: float = 6.0,
        temporal_heatmap_current_weight: float = 1.0,
        heatmap_cache_root: str = "",
        heatmap_latent_scale: float = 0.25,
        heatmap_latent_offset: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.camera_key = str(camera_key)
        if camera_keys is None:
            self.camera_keys = (self.camera_key,)
        else:
            if isinstance(camera_keys, (str, bytes)):
                raise TypeError("camera_keys must be a sequence of zarr camera keys, not a string.")
            normalized_camera_keys = tuple(str(key) for key in camera_keys)
            if not normalized_camera_keys:
                raise ValueError("camera_keys must contain at least one camera key.")
            if len(set(normalized_camera_keys)) != len(normalized_camera_keys):
                raise ValueError(f"camera_keys must be unique, got {normalized_camera_keys!r}.")
            if self.camera_key not in normalized_camera_keys:
                raise ValueError(
                    f"camera_key {self.camera_key!r} must be included in camera_keys "
                    f"{normalized_camera_keys!r}."
                )
            self.camera_keys = normalized_camera_keys
        if camera_key_map is None:
            self.camera_key_map = {key: key for key in self.camera_keys}
        else:
            if not isinstance(camera_key_map, Mapping):
                raise TypeError("camera_key_map must be a mapping from output keys to zarr keys.")
            self.camera_key_map = {
                str(output_key): str(source_key)
                for output_key, source_key in camera_key_map.items()
            }
            expected_keys = set(self.camera_keys)
            actual_keys = set(self.camera_key_map)
            if actual_keys != expected_keys:
                raise ValueError(
                    "camera_key_map keys must exactly match camera_keys; "
                    f"expected {sorted(expected_keys)!r}, got {sorted(actual_keys)!r}."
                )
            empty_source_keys = [
                output_key
                for output_key, source_key in self.camera_key_map.items()
                if not source_key
            ]
            if empty_source_keys:
                raise ValueError(
                    "camera_key_map source keys must be non-empty; empty values for "
                    f"{empty_source_keys!r}."
                )
        self.gaze_key = as_optional_gaze_wam_key(gaze_key)
        self.heatmap_key = as_optional_gaze_wam_key(heatmap_key)
        self.n_obs_steps = _validate_positive_int("n_obs_steps", n_obs_steps)
        self.obs_downsample_steps = _validate_positive_int(
            "obs_downsample_steps",
            obs_downsample_steps,
        )
        self.action_horizon = _validate_positive_int("action_horizon", action_horizon)
        self.n_latency_steps = _validate_nonnegative_int(
            "n_latency_steps",
            n_latency_steps,
        )
        self.action_downsample_steps = _validate_positive_int(
            "action_downsample_steps",
            action_downsample_steps,
        )
        self.image_size = _validate_positive_int_pair("image_size", image_size)
        heatmap_token_grid = _validate_positive_int_pair("heatmap_token_grid", heatmap_token_grid)
        self.heatmap_dim = _validate_positive_int("heatmap_dim", heatmap_dim)
        self.image_resize_mode = _validate_image_resize_mode(str(image_resize_mode))
        self.gaze_is_normalized = normalize_gaze_wam_bool_field(
            "gaze_is_normalized",
            gaze_is_normalized,
            default=True,
        )
        self.action_padding = normalize_gaze_wam_bool_field(
            "action_padding",
            action_padding,
            default=True,
        )
        self.temporal_heatmap_mode = _validate_temporal_heatmap_mode(temporal_heatmap_mode)
        self.temporal_heatmap_window_radius = _validate_nonnegative_int(
            "temporal_heatmap_window_radius",
            temporal_heatmap_window_radius,
        )
        self.temporal_heatmap_beta = _validate_positive_float(
            "temporal_heatmap_beta",
            temporal_heatmap_beta,
        )
        self.temporal_heatmap_sigma_px = _validate_positive_float(
            "temporal_heatmap_sigma_px",
            temporal_heatmap_sigma_px,
        )
        self.temporal_heatmap_current_weight = _validate_positive_float(
            "temporal_heatmap_current_weight",
            temporal_heatmap_current_weight,
        )
        self.seed = normalize_gaze_wam_nonnegative_int_field("seed", seed)
        self.val_ratio = _validate_val_ratio(val_ratio)
        self.heatmap_codec = HeatmapTokenCodec(
            token_grid=heatmap_token_grid,
            image_size=self.image_size,
            sigma_tokens=heatmap_sigma_tokens,
        )
        self.heatmap_cache_root = str(heatmap_cache_root or "")
        self.heatmap_latent_scale = _validate_positive_float(
            "heatmap_latent_scale", heatmap_latent_scale
        )
        try:
            self.heatmap_latent_offset = float(heatmap_latent_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"heatmap_latent_offset must be a finite float, got {heatmap_latent_offset!r}."
            ) from exc
        if not np.isfinite(self.heatmap_latent_offset):
            raise ValueError(
                f"heatmap_latent_offset must be a finite float, got {heatmap_latent_offset!r}."
            )

        root, store = _open_zarr_root(dataset_path)
        self._zarr_store = store
        self.root = root
        self.data_group, self.episode_ends = _resolve_data_group(root)
        missing_camera_keys = sorted(
            {
                source_key
                for source_key in self.camera_key_map.values()
                if source_key not in self.data_group
            }
        )
        if missing_camera_keys:
            raise KeyError(
                "Canonical Gaze-WAM zarr missing required camera key(s): "
                f"{missing_camera_keys!r}."
            )
        if self.gaze_key is None:
            raise ValueError("Canonical Gaze-WAM zarrs must configure a non-null point gaze key.")
        if self.gaze_key not in self.data_group:
            raise KeyError(f"Canonical Gaze-WAM zarr missing required point gaze key '{self.gaze_key}'.")
        self.has_heatmap_key = self.heatmap_key is not None and self.heatmap_key in self.data_group
        self.heatmap_cache = None
        if self.heatmap_cache_root:
            self.heatmap_cache = _HeatmapLatentCache(
                cache_root=self.heatmap_cache_root,
                dataset_path=dataset_path,
                source_rows=int(self.episode_ends[-1]),
                episode_ends=self.episode_ends,
                token_shape=(self.heatmap_codec.num_tokens, self.heatmap_dim),
                latent_scale=self.heatmap_latent_scale,
                latent_offset=self.heatmap_latent_offset,
            )
        self.val_mask = get_val_mask(
            n_episodes=len(self.episode_ends),
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        self.train_mask = ~self.val_mask
        self.indices = _build_sample_indices(
            episode_ends=self.episode_ends,
            action_horizon=self.action_horizon,
            n_latency_steps=self.n_latency_steps,
            action_downsample_steps=self.action_downsample_steps,
            action_padding=self.action_padding,
            episode_mask=self.train_mask,
        )

    def __len__(self):
        return len(self.indices)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = _build_sample_indices(
            episode_ends=self.episode_ends,
            action_horizon=self.action_horizon,
            n_latency_steps=self.n_latency_steps,
            action_downsample_steps=self.action_downsample_steps,
            action_padding=self.action_padding,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask.copy()
        val_set.val_mask = self.train_mask.copy()
        return val_set

    def _sample_temporal_heatmap_image(
        self,
        current_idx: int,
        episode_start: int,
        episode_end: int,
        source_image_size: Tuple[int, int],
        current_has_gaze_label: bool,
    ) -> Optional[torch.Tensor]:
        if self.temporal_heatmap_mode == "off":
            return None
        if self.gaze_key is None:
            return None
        height, width = self.image_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        radius = int(self.temporal_heatmap_window_radius)
        if self.temporal_heatmap_mode == "causal":
            lo = max(int(episode_start), int(current_idx) - radius)
            hi = int(current_idx) + 1
        else:
            lo = max(int(episode_start), int(current_idx) - radius)
            hi = min(int(episode_end), int(current_idx) + radius + 1)
        for source_idx in range(lo, hi):
            has_label = True
            if "has_gaze_label" in self.data_group:
                mask = _presence_mask_values_to_vector(
                    "has_gaze_label",
                    self.data_group["has_gaze_label"][source_idx],
                )
                has_label = bool(mask[0])
            if not has_label:
                continue
            gaze_xy = _normalize_gaze_xy(
                self.data_group[self.gaze_key][source_idx],
                source_image_size=source_image_size,
                gaze_is_normalized=self.gaze_is_normalized,
            )
            dt = abs(int(source_idx) - int(current_idx))
            weight = float(np.exp(-float(dt) / self.temporal_heatmap_beta))
            if dt == 0:
                weight *= self.temporal_heatmap_current_weight
            _splat_gaussian_heatmap(
                heatmap,
                gaze_xy=gaze_xy,
                weight=weight,
                sigma_px=self.temporal_heatmap_sigma_px,
            )
        denom = float(heatmap.sum())
        if denom <= 1e-12:
            if not current_has_gaze_label:
                return torch.zeros((1, height, width), dtype=torch.float32)
            gaze_xy = _normalize_gaze_xy(
                self.data_group[self.gaze_key][current_idx],
                source_image_size=source_image_size,
                gaze_is_normalized=self.gaze_is_normalized,
            )
            _splat_gaussian_heatmap(
                heatmap,
                gaze_xy=gaze_xy,
                weight=1.0,
                sigma_px=self.temporal_heatmap_sigma_px,
            )
            denom = float(heatmap.sum())
        if denom > 1e-12:
            heatmap /= denom
        return torch.from_numpy(np.ascontiguousarray(heatmap[None], dtype=np.float32))

    def _sample_obs_and_gaze(self, idx: int):
        current_idx, episode_start, episode_end, _ = self.indices[idx]
        sampled_source_images = {}
        for source_key in dict.fromkeys(self.camera_key_map.values()):
            sampled_source_images[source_key] = _sample_history(
                self.data_group[source_key],
                current_idx=current_idx,
                episode_start=episode_start,
                horizon=self.n_obs_steps,
                downsample_steps=self.obs_downsample_steps,
            )
        primary_image = sampled_source_images[self.camera_key_map[self.camera_key]]
        source_image_size = _infer_image_hw(primary_image)
        images = {
            output_key: _image_to_chw_float(
                sampled_source_images[source_key],
                image_size=self.image_size,
            )
            for output_key, source_key in self.camera_key_map.items()
        }
        has_gaze_label_mask = self._sample_current_presence_mask(
            "has_gaze_label",
            current_idx=current_idx,
        )
        has_gaze_label = True
        if has_gaze_label_mask is not None:
            has_gaze_label = bool(has_gaze_label_mask.item())

        if has_gaze_label:
            gaze_xy = _normalize_gaze_xy(
                self.data_group[self.gaze_key][current_idx],
                source_image_size=source_image_size,
                gaze_is_normalized=self.gaze_is_normalized,
            )
        else:
            gaze_xy = np.zeros(2, dtype=np.float32)
        heatmap_is_cached = False
        has_heatmap_target = False
        if self.heatmap_cache is not None:
            heatmap_np, has_heatmap_target = self.heatmap_cache.get(current_idx)
            heatmap = torch.from_numpy(np.ascontiguousarray(heatmap_np)).to(dtype=torch.float32)
            heatmap_image = torch.zeros(
                (1, self.image_size[0], self.image_size[1]), dtype=torch.float32
            )
            heatmap_is_cached = True
            default_has_heatmap_image = False
        elif self.has_heatmap_key:
            heatmap_value = self.data_group[self.heatmap_key][current_idx]
            is_dense_heatmap = _looks_like_dense_heatmap_row(
                heatmap_value,
                image_size=self.image_size,
            )
            heatmap = None
            if not is_dense_heatmap:
                heatmap = _heatmap_tokens_to_float(
                    heatmap_value,
                    num_tokens=self.heatmap_codec.num_tokens,
                    heatmap_dim=self.heatmap_dim,
                )
            if heatmap is None:
                if self.heatmap_dim != 1 and _looks_like_scalar_token_heatmap_row(
                    heatmap_value,
                    num_tokens=self.heatmap_codec.num_tokens,
                ):
                    raise ValueError(
                        "Scalar token heatmap zarr rows with shape "
                        f"[{self.heatmap_codec.num_tokens}, 1] cannot supervise "
                        f"full-resolution heatmap_dim={self.heatmap_dim}. Regenerate "
                        "the zarr with dense image heatmaps, or train with heatmap_dim=1."
                    )
                dense_heatmap = _heatmap_image_to_float(
                    heatmap_value,
                    image_size=self.image_size,
                )[0, 0]
                if self.heatmap_dim == 1:
                    heatmap = self.heatmap_codec.encode_image(dense_heatmap)
                elif self.heatmap_dim == self.heatmap_codec.patch_area:
                    heatmap = self.heatmap_codec.patchify_image(dense_heatmap)
                else:
                    heatmap = self.heatmap_codec.encode_latent_image(
                        dense_heatmap,
                        latent_channels=self.heatmap_dim,
                    )
                heatmap_image = dense_heatmap.unsqueeze(0).to(dtype=torch.float32)
            else:
                heatmap_image = self.heatmap_codec.decode_tokens(
                    heatmap,
                    method="gaussian_splat",
                ).unsqueeze(0).to(dtype=torch.float32)
            default_has_heatmap_image = True
            has_heatmap_target = bool(default_has_heatmap_image)
        else:
            heatmap = torch.zeros(
                (self.heatmap_codec.num_tokens, self.heatmap_dim),
                dtype=torch.float32,
            )
            heatmap_image = torch.zeros(
                (1, self.image_size[0], self.image_size[1]),
                dtype=torch.float32,
            )
            default_has_heatmap_image = False
            # The policy can always build an online target from gaze_xy when
            # no dense/cache target exists; keep this availability separate
            # from the optional dense image presence flag.
            has_heatmap_target = True
            temporal_heatmap_image = self._sample_temporal_heatmap_image(
                current_idx=current_idx,
                episode_start=episode_start,
                episode_end=episode_end,
                source_image_size=source_image_size,
                current_has_gaze_label=has_gaze_label,
            )
            if temporal_heatmap_image is not None:
                heatmap_image = temporal_heatmap_image
                default_has_heatmap_image = True
        result = {
            "obs": {
                key: torch.from_numpy(image) for key, image in images.items()
            },
            "gaze_xy": torch.from_numpy(gaze_xy),
            "heatmap": heatmap.unsqueeze(0).to(dtype=torch.float32),
            "has_gaze_label": torch.tensor(has_gaze_label, dtype=torch.bool),
            "heatmap_is_cached": torch.tensor(heatmap_is_cached, dtype=torch.bool),
            "has_heatmap_target": torch.tensor(has_heatmap_target, dtype=torch.bool),
        }
        result["heatmap_image"] = heatmap_image
        if self.heatmap_cache is not None or not self.has_heatmap_key:
            has_heatmap_image = torch.tensor(default_has_heatmap_image, dtype=torch.bool)
        else:
            has_heatmap_image = self._sample_current_presence_mask(
                "has_heatmap_image",
                current_idx=current_idx,
            )
            if has_heatmap_image is None:
                has_heatmap_image = torch.tensor(default_has_heatmap_image, dtype=torch.bool)
            has_heatmap_target = bool(has_heatmap_image.item())
        result["has_heatmap_image"] = has_heatmap_image
        result["has_heatmap_target"] = torch.tensor(has_heatmap_target, dtype=torch.bool)
        return result

    def _sample_current_presence_mask(self, key: str, current_idx: int) -> Optional[torch.Tensor]:
        if key not in self.data_group:
            return None
        mask = _presence_mask_values_to_vector(key, self.data_group[key][current_idx])
        if mask.shape[0] != 1:
            raise ValueError(
                f"{key} must provide exactly one presence value for zarr row {int(current_idx)}, "
                f"got {mask.shape}."
            )
        return torch.tensor(bool(mask[0]), dtype=torch.bool)

    def _sample_future_presence_mask(
        self,
        key: str,
        current_idx: int,
        episode_end: int,
    ) -> Optional[torch.Tensor]:
        if key not in self.data_group:
            return None
        mask = _sample_future(
            self.data_group[key],
            current_idx=current_idx,
            episode_end=episode_end,
            horizon=self.action_horizon + self.n_latency_steps,
            downsample_steps=self.action_downsample_steps,
        )
        if self.n_latency_steps > 0:
            mask = mask[self.n_latency_steps :]
        mask = _presence_mask_values_to_vector(key, mask)
        if mask.shape[0] != self.action_horizon:
            raise ValueError(
                f"{key} sampled future mask must have {self.action_horizon} values, "
                f"got {mask.shape}."
            )
        return torch.tensor(bool(np.all(mask)), dtype=torch.bool)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        for camera_key in self.camera_keys:
            normalizer[camera_key] = get_image_identity_normalizer()
        normalizer["action"] = self._get_action_normalizer()
        return normalizer

    def _get_action_normalizer(self):
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()


class GazeWamRobotDataset(_BaseGazeWamZarrDataset):
    def __init__(
        self,
        dataset_path: str,
        action_abs_key: str = "action_abs_tcp",
        tcp_pose_key: str = "tcp_pose_abs",
        gripper_key: str = "gripper_width",
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)
        self.action_abs_key = action_abs_key
        self.tcp_pose_key = tcp_pose_key
        self.gripper_key = gripper_key

    def _compose_base_abs(self, current_idx: int) -> np.ndarray:
        tcp_pose = np.asarray(self.data_group[self.tcp_pose_key][current_idx], dtype=np.float32)
        gripper = np.asarray(self.data_group[self.gripper_key][current_idx], dtype=np.float32)
        base_abs = _compose_action_with_gripper(tcp_pose, gripper, self.tcp_pose_key)
        _require_finite_array("robot action_base_abs", base_abs)
        return base_abs

    def _compose_action_abs(self, current_idx: int, episode_end: int) -> np.ndarray:
        action_abs = _sample_future(
            self.data_group[self.action_abs_key],
            current_idx=current_idx,
            episode_end=episode_end,
            horizon=self.action_horizon + self.n_latency_steps,
            downsample_steps=self.action_downsample_steps,
        ).astype(np.float32)
        gripper = _sample_future(
            self.data_group[self.gripper_key],
            current_idx=current_idx,
            episode_end=episode_end,
            horizon=self.action_horizon + self.n_latency_steps,
            downsample_steps=self.action_downsample_steps,
        ).astype(np.float32)
        action_abs = _compose_action_with_gripper(
            action_abs,
            gripper,
            self.action_abs_key,
        )
        if self.n_latency_steps > 0:
            action_abs = action_abs[self.n_latency_steps :]
        _require_finite_array("robot action_abs", action_abs)
        return action_abs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        current_idx, _, episode_end, _ = self.indices[idx]
        result = self._sample_obs_and_gaze(idx)
        action_abs = self._compose_action_abs(
            current_idx=current_idx,
            episode_end=episode_end,
        )
        action_base_abs = self._compose_base_abs(current_idx).astype(np.float32)
        action = absolute_actions_to_relative_actions(action_abs, action_base_abs).astype(np.float32)

        result.update(
            {
                "action": torch.from_numpy(action),
                "action_abs": torch.from_numpy(action_abs),
                "action_base_abs": torch.from_numpy(action_base_abs),
                "is_open": torch.tensor(False, dtype=torch.bool),
                "has_action": torch.tensor(True, dtype=torch.bool),
                "has_heatmap": torch.tensor(False, dtype=torch.bool),
                "use_gaze_condition": result["has_gaze_label"].clone(),
                "is_gaze_condition_dropped": (~result["has_gaze_label"]).clone(),
            }
        )
        has_action_abs = self._sample_future_presence_mask(
            "has_action_abs",
            current_idx=current_idx,
            episode_end=episode_end,
        )
        if has_action_abs is not None:
            result["has_action_abs"] = has_action_abs
        has_action_base_abs = self._sample_current_presence_mask(
            "has_action_base_abs",
            current_idx=current_idx,
        )
        if has_action_base_abs is not None:
            result["has_action_base_abs"] = has_action_base_abs
        return result

    def get_all_actions(self) -> torch.Tensor:
        if len(self) <= 0:
            raise ValueError(
                "Robot dataset produced zero relative-action samples; cannot fit the "
                "Gaze-WAM action normalizer. Check episode length, val_ratio, "
                "action_horizon, n_latency_steps, downsampling, and action_padding."
            )

        action_source = np.asarray(self.data_group[self.action_abs_key][:], dtype=np.float32)
        tcp_pose_source = np.asarray(self.data_group[self.tcp_pose_key][:], dtype=np.float32)
        gripper_source = np.asarray(self.data_group[self.gripper_key][:], dtype=np.float32)
        current_indices = self.indices[:, 0]
        episode_ends = self.indices[:, 2]
        sampled_horizon = self.action_horizon + self.n_latency_steps
        future_offsets = (
            ACTION_TARGET_START_OFFSET_STEPS
            + np.arange(sampled_horizon, dtype=np.int64) * self.action_downsample_steps
        )

        actions = np.empty((len(self), self.action_horizon, 10), dtype=np.float32)
        batch_size = 4096
        for batch_start in range(0, len(self), batch_size):
            batch_end = min(batch_start + batch_size, len(self))
            batch_current = current_indices[batch_start:batch_end]
            batch_episode_ends = episode_ends[batch_start:batch_end]
            target_indices = np.minimum(
                batch_current[:, None] + future_offsets[None, :],
                batch_episode_ends[:, None] - 1,
            )

            action_abs = _compose_action_with_gripper(
                action_source[target_indices],
                gripper_source[target_indices],
                self.action_abs_key,
            )
            if self.n_latency_steps > 0:
                action_abs = action_abs[:, self.n_latency_steps :]
            action_base_abs = _compose_action_with_gripper(
                tcp_pose_source[batch_current],
                gripper_source[batch_current],
                self.tcp_pose_key,
            )
            relative_actions = absolute_actions_to_relative_actions(
                action_abs,
                action_base_abs,
            ).astype(np.float32)
            _require_finite_array("robot normalizer relative actions", relative_actions)
            actions[batch_start:batch_end] = relative_actions

        return torch.from_numpy(actions)

    def _get_action_normalizer(self):
        return _make_action_normalizer(self.get_all_actions().numpy())


class GazeWamOpenDataset(_BaseGazeWamZarrDataset):
    def __init__(
        self,
        dataset_path: str,
        action_dim: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)
        self.action_dim = _validate_positive_int("action_dim", action_dim)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self._sample_obs_and_gaze(idx)
        result.update(
            {
                "action": torch.zeros(self.action_horizon, self.action_dim, dtype=torch.float32),
                "is_open": torch.tensor(True, dtype=torch.bool),
                "has_action": torch.tensor(False, dtype=torch.bool),
                "has_heatmap": result["has_heatmap_target"].clone(),
                "use_gaze_condition": torch.tensor(False, dtype=torch.bool),
                "is_gaze_condition_dropped": torch.tensor(True, dtype=torch.bool),
            }
        )
        return result

    def get_all_actions(self) -> torch.Tensor:
        return torch.zeros(len(self), self.action_horizon, self.action_dim, dtype=torch.float32)

    def _get_action_normalizer(self):
        raise RuntimeError(
            "GazeWamOpenDataset contains zero dummy action placeholders and must not "
            "fit the action normalizer. Fit the policy normalizer from "
            "GazeWamRobotDataset relative actions instead."
        )
