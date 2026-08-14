if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import json
import math
import os
import pathlib
import pickle
import random

import cv2
import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import zarr
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf, open_dict

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.omegaconf_resolvers import register_safe_omegaconf_resolvers
from diffusion_policy.common.gaze_wam_dataloader_checks import (
    _check_training_dataloader_lengths,
    _check_training_dataset_lengths,
)
from diffusion_policy.common.gaze_wam_training_config import (
    _normalize_gaze_wam_early_bool_config,
    _normalize_gaze_wam_training_config,
    _normalize_gaze_wam_task_routing_config,
    _parse_dataloader_runtime_config,
    gaze_wam_action_normalizer_contract,
    gaze_wam_data_stream_contract,
    gaze_wam_prepared_dataloader_batches,
    gaze_wam_planned_optimizer_steps,
    gaze_wam_loss_routing_validation_guardrails_ok,
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_nonnegative_float_field,
    normalize_gaze_wam_nonnegative_int_field,
    normalize_gaze_wam_positive_float_field,
    normalize_gaze_wam_positive_int_sequence,
    normalize_gaze_wam_positive_int_field,
    normalize_gaze_wam_unit_interval_float_field,
    validate_gaze_wam_training_config,
    validate_gaze_wam_task_routing_config,
    resolve_gaze_wam_batching_config,
)
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.normalize_util import get_image_identity_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.gaze_wam_mixing import build_gaze_wam_mixed_batch
from diffusion_policy.dataset.gaze_wam_batching import build_gaze_wam_dataloader
from diffusion_policy.dataset.gaze_wam_dataset import ACTION_TARGET_START_OFFSET_STEPS
from diffusion_policy.common.gaze_wam_transfer import (
    export_gaze_wam_transfer_artifact,
    load_gaze_wam_transfer_artifact,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.gaze_wam.loss import distributed_mask_count
from diffusion_policy.model.gaze_wam.routing import loss_routing_summary
from diffusion_policy.policy.gaze_wam_policy import GazeWamPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

register_safe_omegaconf_resolvers()

SUPPORTED_PREALIGNED_IMAGE_RESIZE_MODES = ("stretch", "letterbox")


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    return float(value)


def _ensure_optimizer_initial_lr_for_resume(optimizer, base_lr, obs_encoder_lr):
    """Backfill scheduler metadata for checkpoints saved before resume support."""
    base_lr = float(base_lr)
    obs_encoder_lr = float(obs_encoder_lr)
    for group in optimizer.param_groups:
        if "initial_lr" in group:
            continue
        current_lr = float(group.get("lr", base_lr))
        if abs(current_lr - obs_encoder_lr) < abs(current_lr - base_lr):
            group["initial_lr"] = obs_encoder_lr
        else:
            group["initial_lr"] = base_lr


def _planned_prepared_epoch_batches(dataloader, accelerator) -> int:
    """Return the exact per-rank length expected after ``accelerator.prepare``."""
    raw_batches = int(len(dataloader))
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    split_batches = bool(getattr(accelerator, "split_batches", False))
    even_batches = bool(getattr(accelerator, "even_batches", True))
    process_index = int(getattr(accelerator, "process_index", 0) or 0)
    drop_last = bool(
        getattr(
            dataloader,
            "drop_last",
            getattr(getattr(dataloader, "batch_sampler", None), "drop_last", False),
        )
    )
    if (
        num_processes > 1
        and not split_batches
        and not even_batches
        and raw_batches % num_processes != 0
    ):
        raise ValueError(
            "Gaze-WAM requires equal per-rank epoch lengths for global step "
            "accounting; set Accelerate even_batches=true or use a dataloader "
            "whose batch count is divisible by num_processes."
        )
    return gaze_wam_prepared_dataloader_batches(
        raw_batches,
        num_processes=num_processes,
        split_batches=split_batches,
        even_batches=even_batches,
        drop_last=drop_last,
        process_index=process_index,
    )


def _validate_prepared_epoch_driver_length(
    planned_batches: int,
    actual_batches: int,
) -> int:
    """Fail fast when Accelerate changes the epoch clock after preparation."""
    planned_batches = int(planned_batches)
    actual_batches = int(actual_batches)
    if actual_batches != planned_batches:
        raise RuntimeError(
            "Accelerate prepared epoch length disagrees with the scheduler plan: "
            f"planned={planned_batches}, actual={actual_batches}."
        )
    return actual_batches


def _validate_gaze_wam_accumulation_flush_contract(accelerator) -> bool:
    """Require Accelerate to flush a partial accumulation window per epoch."""
    gradient_state = getattr(accelerator, "gradient_state", None)
    sync_with_dataloader = getattr(gradient_state, "sync_with_dataloader", None)
    if sync_with_dataloader is not True:
        raise RuntimeError(
            "Gaze-WAM requires Accelerate sync_with_dataloader=true so the final "
            "partial gradient-accumulation window is applied at each epoch boundary."
        )
    return True


def _gaze_wam_checkpoint_due(
    epoch: int,
    checkpoint_every: int,
    stop_after_epoch: bool,
) -> bool:
    """Always retain the terminal epoch when a global step budget stops a run."""
    return bool(stop_after_epoch) or int(epoch) % int(checkpoint_every) == 0


def _new_train_window_log_accumulator():
    return {
        "raw_loss_sum": 0.0,
        "microbatch_count": 0.0,
        "action_loss_sum": 0.0,
        "action_mask_count": 0.0,
        "heatmap_loss_sum": 0.0,
        "heatmap_xy_loss_sum": 0.0,
        "heatmap_point_nll_loss_sum": 0.0,
        "heatmap_js_loss_sum": 0.0,
        "heatmap_token_kl_loss_sum": 0.0,
        "heatmap_mask_count": 0.0,
        "heatmap_xy_mask_count": 0.0,
        "routing": {},
    }


def _accumulate_train_window_log(accumulator, raw_loss, components, routing_summary):
    accumulator["raw_loss_sum"] += _to_float(raw_loss)
    accumulator["microbatch_count"] += 1.0
    action_mask_count = _to_float(components["action_loss_mask_count"])
    heatmap_mask_count = _to_float(components["heatmap_loss_mask_count"])
    heatmap_xy_mask_count = _to_float(components["heatmap_xy_loss_mask_count"])
    accumulator["action_loss_sum"] += _to_float(components["action_loss"]) * action_mask_count
    accumulator["action_mask_count"] += action_mask_count
    accumulator["heatmap_loss_sum"] += (
        _to_float(components["heatmap_loss"]) * heatmap_mask_count
    )
    accumulator["heatmap_xy_loss_sum"] += (
        _to_float(components["heatmap_xy_loss"]) * heatmap_xy_mask_count
    )
    accumulator["heatmap_point_nll_loss_sum"] += (
        _to_float(components["heatmap_point_nll_loss"]) * heatmap_xy_mask_count
    )
    accumulator["heatmap_js_loss_sum"] += (
        _to_float(components["heatmap_js_loss"]) * heatmap_mask_count
    )
    accumulator["heatmap_token_kl_loss_sum"] += (
        _to_float(components["heatmap_token_kl_loss"]) * heatmap_mask_count
    )
    accumulator["heatmap_mask_count"] += heatmap_mask_count
    accumulator["heatmap_xy_mask_count"] += heatmap_xy_mask_count
    routing_accumulator = accumulator["routing"]
    for key, value in routing_summary.items():
        routing_accumulator[key] = routing_accumulator.get(key, 0.0) + float(value)


def _finalize_train_window_log(accumulator):
    microbatch_count = max(float(accumulator["microbatch_count"]), 1.0)
    action_mask_count = float(accumulator["action_mask_count"])
    heatmap_mask_count = float(accumulator["heatmap_mask_count"])
    heatmap_xy_mask_count = float(accumulator["heatmap_xy_mask_count"])
    result = {
        "train_loss": float(accumulator["raw_loss_sum"]) / microbatch_count,
        "train_action_loss": (
            float(accumulator["action_loss_sum"]) / action_mask_count
            if action_mask_count > 0
            else 0.0
        ),
        "train_heatmap_loss": (
            float(accumulator["heatmap_loss_sum"]) / heatmap_mask_count
            if heatmap_mask_count > 0
            else 0.0
        ),
        "train_heatmap_xy_loss": (
            float(accumulator["heatmap_xy_loss_sum"]) / heatmap_xy_mask_count
            if heatmap_xy_mask_count > 0
            else 0.0
        ),
        "train_heatmap_point_nll_loss": (
            float(accumulator["heatmap_point_nll_loss_sum"]) / heatmap_xy_mask_count
            if heatmap_xy_mask_count > 0
            else 0.0
        ),
        "train_heatmap_js_loss": (
            float(accumulator["heatmap_js_loss_sum"]) / heatmap_mask_count
            if heatmap_mask_count > 0
            else 0.0
        ),
        "train_heatmap_token_kl_loss": (
            float(accumulator["heatmap_token_kl_loss_sum"]) / heatmap_mask_count
            if heatmap_mask_count > 0
            else 0.0
        ),
        "train_action_mask_count": action_mask_count,
        "train_heatmap_mask_count": heatmap_mask_count,
        "train_heatmap_xy_mask_count": heatmap_xy_mask_count,
        "train_accumulated_microbatches": int(accumulator["microbatch_count"]),
    }
    for key, value in accumulator["routing"].items():
        result[f"train_routing_{key}"] = int(value)
    return result


def _distributed_scalar_sum(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class _NullJsonLogger:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log(self, data: dict):
        return None


class _RestartingDataLoaderIterator:
    """Stream batches and rebuild the dataloader iterator when it is exhausted."""

    def __init__(self, dataloader, name: str):
        self.dataloader = dataloader
        self.name = str(name)
        self._iterator = None
        self.restart_count = 0

    def next(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            self.restart_count += 1
            self._iterator = iter(self.dataloader)
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"{self.name} dataloader produced no batches. Check dataset length, "
                    "batch_size, and drop_last before starting policy training."
                ) from exc


def _slice_batch(batch, mask):
    def apply_mask(value):
        if isinstance(value, dict):
            return {key: apply_mask(item) for key, item in value.items()}
        return value[mask]

    return {key: apply_mask(value) for key, value in batch.items()}


def _obs_for_metric(batch):
    obs = dict(batch["obs"])
    obs["gaze_xy"] = batch["gaze_xy"]
    obs["has_gaze_label"] = batch["has_gaze_label"]
    return obs


def _make_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _make_generator_for_device(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _make_heatmap_only_identity_normalizer(camera_key: str) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer[str(camera_key)] = get_image_identity_normalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def _cfg_list(value):
    return normalize_gaze_wam_positive_int_sequence("config sequence", value)


def _cfg_positive_int_sequence(name: str, value, length: int = None):
    return normalize_gaze_wam_positive_int_sequence(name, value, length=length)


def _cfg_get_str(container, key: str, default: str = "") -> str:
    if container is None:
        return default
    try:
        value = container.get(key, default)
    except AttributeError:
        value = getattr(container, key, default)
    if value is None:
        return default
    return str(value)


def _cfg_get_path_str(container, key: str) -> str:
    return _cfg_get_str(container, key, "").strip()


def _cfg_get_bool(container, key: str, default: bool = False, name: str = None) -> bool:
    if container is None:
        value = default
    else:
        try:
            value = container.get(key, default)
        except AttributeError:
            value = getattr(container, key, default)
    return normalize_gaze_wam_bool_field(name or key, value, default=default)


def _cfg_get_nonnegative_float(container, key: str, default: float = 0.0, name: str = None) -> float:
    field_name = name or key
    if container is None:
        value = default
    else:
        try:
            value = container.get(key, default)
        except AttributeError:
            value = getattr(container, key, default)
    if value is None:
        value = default
    return normalize_gaze_wam_nonnegative_float_field(field_name, value, default)


def _cfg_get_positive_float(container, key: str, default: float = 1.0, name: str = None) -> float:
    field_name = name or key
    value = default
    if container is not None:
        try:
            value = container.get(key, default)
        except AttributeError:
            value = getattr(container, key, default)
    value = default if value is None else value
    return normalize_gaze_wam_positive_float_field(field_name, value, default=default)


def _cfg_get_finite_float(container, key: str, default: float = 0.0, name: str = None) -> float:
    field_name = name or key
    value = default
    if container is not None:
        try:
            value = container.get(key, default)
        except AttributeError:
            value = getattr(container, key, default)
    value = default if value is None else value
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite float, got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float, got {value!r}.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite float, got {value!r}.")
    return parsed


def _cfg_get_optional_bool(container, key: str):
    if container is None:
        return None
    try:
        value = container.get(key, None)
    except AttributeError:
        value = getattr(container, key, None)
    if value is None:
        return None
    return normalize_gaze_wam_bool_field(key, value, default=False)


def _configured_path_exists(value: str) -> bool:
    return bool(value) and pathlib.Path(value).exists()


def _configured_path_is_file(value: str) -> bool:
    return bool(value) and pathlib.Path(value).is_file()


def _configured_path_is_dir(value: str) -> bool:
    return bool(value) and pathlib.Path(value).is_dir()


def _latent_stats_summary(path: str) -> dict:
    result = {
        "path": str(path or ""),
        "exists": False,
        "raw_latent_abs_max": None,
        "raw_latent_abs_p99_5": None,
        "scale_for_abs_max": None,
        "scale_for_abs_max_rounded_down_0p01": None,
        "recommended_default": None,
        "recommended_default_basis": None,
        "raw_latent_min": None,
        "raw_latent_max": None,
        "raw_latent_std": None,
    }
    if not path:
        return result
    file = pathlib.Path(path)
    result["exists"] = file.is_file()
    if not file.is_file():
        return result
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    raw_latent = data.get("raw_latent") or {}
    raw_abs = data.get("raw_latent_abs") or {}
    scale = data.get("scale_recommendations") or {}
    result.update(
        {
            "raw_latent_abs_max": raw_abs.get("abs_max"),
            "raw_latent_abs_p99_5": raw_abs.get("abs_p99_5"),
            "scale_for_abs_max": scale.get("scale_for_abs_max"),
            "scale_for_abs_max_rounded_down_0p01": scale.get(
                "scale_for_abs_max_rounded_down_0p01"
            ),
            "recommended_default": scale.get("recommended_default"),
            "recommended_default_basis": scale.get("recommended_default_basis"),
            "raw_latent_min": raw_latent.get("min"),
            "raw_latent_max": raw_latent.get("max"),
            "raw_latent_std": raw_latent.get("std"),
        }
    )
    return result


def _scaled_latent_range_from_stats(
    stats: dict,
    scale: float,
    offset: float,
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> dict:
    raw_min = stats.get("raw_latent_min")
    raw_max = stats.get("raw_latent_max")
    result = {
        "available": raw_min is not None and raw_max is not None,
        "normalization": "(raw_latent - offset) * scale",
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
    }
    if not result["available"]:
        return result
    scaled_min = (float(raw_min) - float(offset)) * float(scale)
    scaled_max = (float(raw_max) - float(offset)) * float(scale)
    low = min(scaled_min, scaled_max)
    high = max(scaled_min, scaled_max)
    result.update(
        {
            "raw_min": float(raw_min),
            "raw_max": float(raw_max),
            "scale": float(scale),
            "offset": float(offset),
            "scaled_min": float(low),
            "scaled_max": float(high),
            "within_clip": low >= float(clip_low) - 1e-6
            and high <= float(clip_high) + 1e-6,
        }
    )
    return result


def _jsonable_attr(value):
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


def _read_zarr_metadata_attrs(dataset_path: str):
    if not dataset_path:
        return {"available": False, "attrs": {}, "error": "dataset_path is empty"}
    store = None
    try:
        if str(dataset_path).endswith(".zip"):
            store = zarr.ZipStore(dataset_path, mode="r")
            root = zarr.group(store=store)
        else:
            root = zarr.open(dataset_path, mode="r")
        attrs = root["meta"].attrs if "meta" in root else root.attrs
        return {
            "available": True,
            "attrs": {str(key): _jsonable_attr(value) for key, value in attrs.items()},
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "attrs": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if store is not None:
            store.close()


def _metadata_image_size_pair(value):
    if value is None or isinstance(value, str):
        return []
    try:
        pair = _cfg_positive_int_sequence("meta.attrs.image_size", value, length=2)
    except ValueError:
        return []
    return pair


def _dataset_sampling_summary(dataset_cfg):
    return {
        "n_obs_steps": normalize_gaze_wam_positive_int_field(
            "dataset.n_obs_steps",
            dataset_cfg.get("n_obs_steps", 0),
        ),
        "obs_downsample_steps": normalize_gaze_wam_positive_int_field(
            "dataset.obs_downsample_steps",
            dataset_cfg.get("obs_downsample_steps", 1),
        ),
        "action_horizon": normalize_gaze_wam_positive_int_field(
            "dataset.action_horizon",
            dataset_cfg.get("action_horizon", 0),
        ),
        "n_latency_steps": normalize_gaze_wam_nonnegative_int_field(
            "dataset.n_latency_steps",
            dataset_cfg.get("n_latency_steps", 0),
        ),
        "action_downsample_steps": normalize_gaze_wam_positive_int_field(
            "dataset.action_downsample_steps",
            dataset_cfg.get("action_downsample_steps", 1),
        ),
        "action_padding": _cfg_get_bool(
            dataset_cfg,
            "action_padding",
            True,
            "dataset.action_padding",
        ),
    }


def _zarr_data_source_summary(
    dataset_path: str,
    expected_dataset_type: str,
    expected_image_resize_mode: str,
    expected_image_size,
):
    expected_image_size = _cfg_positive_int_sequence(
        "expected_image_size",
        expected_image_size,
        length=2,
    )
    metadata = _read_zarr_metadata_attrs(dataset_path)
    attrs = metadata["attrs"] if metadata["available"] else {}
    dataset_type = attrs.get("dataset_type")
    image_resize_mode = attrs.get("image_resize_mode")
    image_size = _metadata_image_size_pair(attrs.get("image_size"))
    return {
        "path": str(dataset_path),
        "expected_dataset_type": str(expected_dataset_type),
        "expected_image_resize_mode": str(expected_image_resize_mode),
        "expected_image_size": expected_image_size,
        "metadata_attrs_available": bool(metadata["available"]),
        "metadata_attrs": attrs,
        "metadata_error": metadata["error"],
        "metadata_dataset_type": "" if dataset_type is None else str(dataset_type),
        "metadata_image_resize_mode": "" if image_resize_mode is None else str(image_resize_mode),
        "metadata_image_size": image_size,
        "dataset_type_matches_expected": (
            dataset_type is not None and str(dataset_type) == str(expected_dataset_type)
        ),
        "image_resize_mode_matches_expected": (
            image_resize_mode is not None
            and str(image_resize_mode) == str(expected_image_resize_mode)
        ),
        "image_size_matches_expected": bool(image_size)
        and image_size == expected_image_size,
    }


def _close_fraction(value: float, target: float, eps: float = 1e-9) -> bool:
    return abs(float(value) - float(target)) < eps


def _shape(value):
    if torch.is_tensor(value):
        return [int(item) for item in value.shape]
    return None


def _sample_optional_metadata_summary(dataset):
    if dataset is None or len(dataset) <= 0:
        return {
            "available": False,
            "length": int(len(dataset)) if dataset is not None else 0,
        }
    sample = dataset[0]
    summary = {
        "available": True,
        "length": int(len(dataset)),
        "optional_shapes": {},
        "presence_masks": {},
    }
    for key in ("action_abs", "action_base_abs", "heatmap_image"):
        if key in sample:
            summary["optional_shapes"][key] = _shape(sample[key])
    for key in ("has_action_abs", "has_action_base_abs", "has_heatmap_image"):
        if key in sample:
            summary["presence_masks"][key] = bool(sample[key].item())
    return summary


def _build_training_contract_summary(
    cfg,
    robot_dataset,
    robot_val_dataset,
    open_dataset=None,
    open_val_dataset=None,
    dataloader_lengths=None,
    policy=None,
    accelerator=None,
    transfer_load_summary=None,
    planned_optimizer_steps=None,
    epoch_driver_batches=None,
    prepared_epoch_batches=None,
):
    training_config = validate_gaze_wam_training_config(cfg)
    task_routing_config = validate_gaze_wam_task_routing_config(cfg)
    robot_gaze_dropout_prob = normalize_gaze_wam_unit_interval_float_field(
        "task.robot_gaze_dropout_prob",
        task_routing_config.get("robot_gaze_dropout_prob", 0.0),
        default=0.0,
        include_one=True,
    )
    robot_heatmap_on_gaze_dropout = normalize_gaze_wam_bool_field(
        "task.robot_heatmap_on_gaze_dropout",
        task_routing_config.get("robot_heatmap_on_gaze_dropout", True),
        default=True,
    )
    robot_batch_size = int(training_config["robot_batch_size"])
    open_batch_size = int(training_config["open_batch_size"])
    total_batch_size = robot_batch_size + open_batch_size
    use_robot_data = robot_batch_size > 0
    use_open_data = open_batch_size > 0
    gradient_accumulate_every = int(training_config["gradient_accumulate_every"])
    robot_ratio = robot_batch_size / total_batch_size if total_batch_size > 0 else 0.0
    open_ratio = open_batch_size / total_batch_size if total_batch_size > 0 else 0.0
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    mixed_precision = str(getattr(accelerator, "mixed_precision", "no") or "no")
    require_amp = bool(cfg.training.get("require_amp", True))
    distributed_type = getattr(accelerator, "distributed_type", "NO")
    distributed_type = getattr(distributed_type, "name", str(distributed_type))
    effective_train_batch_size = total_batch_size * num_processes * gradient_accumulate_every
    effective_robot_batch_size = robot_batch_size * num_processes * gradient_accumulate_every
    effective_open_batch_size = open_batch_size * num_processes * gradient_accumulate_every

    image_shape = _cfg_positive_int_sequence(
        "task.image_shape",
        cfg.task.image_shape,
        length=3,
    )
    task_image_size = image_shape[-2:]
    heatmap_token_grid = _cfg_positive_int_sequence(
        "task.heatmap_token_grid",
        cfg.task.heatmap_token_grid,
        length=2,
    )
    n_obs_steps = normalize_gaze_wam_positive_int_field(
        "task.n_obs_steps",
        cfg.task.n_obs_steps,
    )
    n_latency_steps = normalize_gaze_wam_nonnegative_int_field(
        "task.n_latency_steps",
        cfg.task.get("n_latency_steps", 0),
    )
    action_horizon = normalize_gaze_wam_positive_int_field(
        "task.action_horizon",
        cfg.task.action_horizon,
    )
    action_dim = normalize_gaze_wam_positive_int_field(
        "task.action_dim",
        cfg.task.action_dim,
    )
    camera_key = _cfg_get_str(
        cfg.task.robot_dataset,
        "camera_key",
        _cfg_get_str(cfg.task, "camera_key", "camera0_rgb"),
    )
    normalizer_contract = gaze_wam_action_normalizer_contract(
        action_dim=action_dim,
        camera_key=camera_key,
        heatmap_only=not use_robot_data,
    )
    data_stream_contract = gaze_wam_data_stream_contract(
        robot_dataset_path=str(cfg.task.robot_dataset.dataset_path),
        open_dataset_path=str(cfg.task.open_dataset.dataset_path),
        robot_dataset_class=_cfg_get_str(cfg.task.robot_dataset, "_target_"),
        open_dataset_class=_cfg_get_str(cfg.task.open_dataset, "_target_"),
        robot_batch_size=robot_batch_size,
        open_batch_size=open_batch_size,
        batching_config=training_config["batching"],
    )
    heatmap_num_tokens = normalize_gaze_wam_positive_int_field(
        "task.heatmap_num_tokens",
        cfg.task.heatmap_num_tokens,
    )
    heatmap_dim = normalize_gaze_wam_positive_int_field(
        "task.heatmap_dim",
        cfg.task.heatmap_dim,
    )
    heatmap_patch_area = 0
    if len(heatmap_token_grid) == 2 and len(task_image_size) == 2:
        if (
            task_image_size[0] % heatmap_token_grid[0] == 0
            and task_image_size[1] % heatmap_token_grid[1] == 0
        ):
            heatmap_patch_area = (
                task_image_size[0]
                // heatmap_token_grid[0]
                * task_image_size[1]
                // heatmap_token_grid[1]
            )
    heatmap_dim_matches_lossless_patch = heatmap_dim == heatmap_patch_area
    heatmap_spatial_decoder = _cfg_get_str(
        cfg.policy,
        "heatmap_spatial_decoder",
        "cosmos_tokenizer",
    )
    heatmap_distribution_mode = _cfg_get_str(
        cfg.policy,
        "heatmap_distribution_mode",
        "intensity_softplus",
    )
    heatmap_dsnt_temperature = _cfg_get_positive_float(
        cfg.policy,
        "heatmap_dsnt_temperature",
        0.1,
        "policy.heatmap_dsnt_temperature",
    )
    heatmap_latent_scale = _cfg_get_positive_float(
        cfg.policy,
        "heatmap_latent_scale",
        1.0,
        "policy.heatmap_latent_scale",
    )
    heatmap_latent_offset = _cfg_get_finite_float(
        cfg.policy,
        "heatmap_latent_offset",
        0.0,
        "policy.heatmap_latent_offset",
    )
    heatmap_latent_stats_path = _cfg_get_path_str(
        cfg.policy,
        "heatmap_latent_stats_path",
    )
    heatmap_latent_stats = _latent_stats_summary(heatmap_latent_stats_path)
    heatmap_latent_scaled_range = _scaled_latent_range_from_stats(
        heatmap_latent_stats,
        heatmap_latent_scale,
        heatmap_latent_offset,
    )
    heatmap_scheduler_clip_sample = _cfg_get_optional_bool(
        cfg.policy,
        "heatmap_scheduler_clip_sample",
    )
    max_image_tokens = int(cfg.policy.max_image_tokens)
    action_loss_weight = _cfg_get_nonnegative_float(
        cfg.policy,
        "action_loss_weight",
        0.0,
        "policy.action_loss_weight",
    )
    heatmap_loss_weight = _cfg_get_nonnegative_float(
        cfg.policy,
        "heatmap_loss_weight",
        0.0,
        "policy.heatmap_loss_weight",
    )
    heatmap_token_kl_loss_weight = _cfg_get_nonnegative_float(
        cfg.policy,
        "heatmap_token_kl_loss_weight",
        0.0,
        "policy.heatmap_token_kl_loss_weight",
    )
    heatmap_xy_loss_weight = _cfg_get_nonnegative_float(
        cfg.policy,
        "heatmap_xy_loss_weight",
        1.0,
        "policy.heatmap_xy_loss_weight",
    )
    heatmap_point_nll_loss_weight = _cfg_get_nonnegative_float(
        cfg.policy,
        "heatmap_point_nll_loss_weight",
        0.0,
        "policy.heatmap_point_nll_loss_weight",
    )
    heatmap_js_loss_weight = _cfg_get_nonnegative_float(
        cfg.policy,
        "heatmap_js_loss_weight",
        1.0,
        "policy.heatmap_js_loss_weight",
    )
    heatmap_diffusion_final_loss_enabled = _cfg_get_bool(
        cfg.policy,
        "heatmap_diffusion_final_loss_enabled",
        False,
        "policy.heatmap_diffusion_final_loss_enabled",
    )
    heatmap_final_loss_timestep_weighting = _cfg_get_str(
        cfg.policy,
        "heatmap_final_loss_timestep_weighting",
        "none",
    )
    heatmap_dsnt_target_sigma_px = _cfg_get_nonnegative_float(
        cfg.policy,
        "heatmap_dsnt_target_sigma_px",
        6.0,
        "policy.heatmap_dsnt_target_sigma_px",
    )
    cfg_scale = _cfg_get_nonnegative_float(
        cfg.policy,
        "cfg_scale",
        1.0,
        "policy.cfg_scale",
    )
    task_image_resize_mode = _cfg_get_str(cfg.task, "image_resize_mode", "stretch")
    robot_image_resize_mode = _cfg_get_str(
        cfg.task.robot_dataset,
        "image_resize_mode",
        task_image_resize_mode,
    )
    open_image_resize_mode = _cfg_get_str(
        cfg.task.open_dataset,
        "image_resize_mode",
        task_image_resize_mode,
    )
    image_resize_modes = {
        "task": task_image_resize_mode,
        "robot_dataset": robot_image_resize_mode,
        "open_dataset": open_image_resize_mode,
    }
    robot_image_size = _cfg_positive_int_sequence(
        "task.robot_dataset.image_size",
        cfg.task.robot_dataset.image_size,
        length=2,
    )
    open_image_size = _cfg_positive_int_sequence(
        "task.open_dataset.image_size",
        cfg.task.open_dataset.image_size,
        length=2,
    )
    image_sizes = {
        "task": task_image_size,
        "robot_dataset": robot_image_size,
        "open_dataset": open_image_size,
    }
    robot_data_source = _zarr_data_source_summary(
        dataset_path=str(cfg.task.robot_dataset.dataset_path),
        expected_dataset_type="robot",
        expected_image_resize_mode=robot_image_resize_mode,
        expected_image_size=robot_image_size,
    )
    open_data_source = _zarr_data_source_summary(
        dataset_path=str(cfg.task.open_dataset.dataset_path),
        expected_dataset_type="open",
        expected_image_resize_mode=open_image_resize_mode,
        expected_image_size=open_image_size,
    )
    obs_encoder_checkpoint_path = _cfg_get_path_str(cfg.policy.obs_encoder, "checkpoint_path")
    obs_encoder_cache_dir = _cfg_get_path_str(cfg.policy.obs_encoder, "cache_dir")
    obs_encoder_checkpoint_exists = _configured_path_exists(obs_encoder_checkpoint_path)
    obs_encoder_cache_dir_exists = _configured_path_exists(obs_encoder_cache_dir)
    obs_encoder_checkpoint_is_file = _configured_path_is_file(obs_encoder_checkpoint_path)
    obs_encoder_cache_dir_is_dir = _configured_path_is_dir(obs_encoder_cache_dir)
    obs_encoder_local_weight_source_configured = bool(
        obs_encoder_checkpoint_path or obs_encoder_cache_dir
    )
    obs_encoder_local_weight_source_exists = (
        (not obs_encoder_checkpoint_path or obs_encoder_checkpoint_exists)
        and (not obs_encoder_cache_dir or obs_encoder_cache_dir_exists)
    )
    obs_encoder_local_weight_source_valid = (
        (not obs_encoder_checkpoint_path or obs_encoder_checkpoint_is_file)
        and (not obs_encoder_cache_dir or obs_encoder_cache_dir_is_dir)
    )
    robot_train_samples = int(len(robot_dataset)) if robot_dataset is not None else 0
    robot_val_samples = int(len(robot_val_dataset)) if robot_val_dataset is not None else 0
    open_train_samples = int(len(open_dataset)) if open_dataset is not None else 0
    open_val_samples = int(len(open_val_dataset)) if open_val_dataset is not None else 0
    dataloader_lengths = dict(dataloader_lengths or {})
    robot_train_batches = dataloader_lengths.get("robot_train_batches")
    robot_val_batches = dataloader_lengths.get("robot_val_batches")
    open_train_batches = dataloader_lengths.get("open_train_batches")
    open_val_batches = dataloader_lengths.get("open_val_batches")
    task_sampling = {
        "n_obs_steps": n_obs_steps,
        "action_horizon": action_horizon,
        "n_latency_steps": n_latency_steps,
    }
    robot_sampling = _dataset_sampling_summary(cfg.task.robot_dataset)
    open_sampling = _dataset_sampling_summary(cfg.task.open_dataset)
    sampling_compare_keys = ["n_obs_steps", "action_horizon", "n_latency_steps"]
    robot_sampling_matches_task = all(
        robot_sampling[key] == task_sampling[key] for key in sampling_compare_keys
    )
    open_sampling_matches_task = all(
        open_sampling[key] == task_sampling[key] for key in sampling_compare_keys
    )

    requested_robot_ratio = training_config["batching"].get("requested_robot_ratio")
    requested_open_ratio = training_config["batching"].get("requested_open_ratio")
    requested_total = training_config["batching"].get(
        "requested_total_batch_size_per_process"
    )
    expected_robot_batch = None
    expected_open_batch = None
    if (
        requested_total is not None
        and requested_robot_ratio is not None
        and requested_open_ratio is not None
    ):
        expected_robot_batch = int(
            math.floor(float(requested_total) * float(requested_robot_ratio) + 0.5)
        )
        expected_open_batch = int(requested_total) - expected_robot_batch
    resolved_quota_matches_request = (
        expected_robot_batch is None
        or (
            robot_batch_size == expected_robot_batch
            and open_batch_size == expected_open_batch
        )
    )
    action_loss_weight_matches_active_sources = (
        _close_fraction(action_loss_weight, 1.0)
        if use_robot_data
        else _close_fraction(action_loss_weight, 0.0)
    )
    obs_encoder_source_valid = (
        obs_encoder_local_weight_source_exists
        and obs_encoder_local_weight_source_valid
        if obs_encoder_local_weight_source_configured
        else _cfg_get_bool(
            cfg.policy.obs_encoder,
            "pretrained",
            False,
            "policy.obs_encoder.pretrained",
        )
    )

    heatmap_objective = _cfg_get_str(cfg.policy, "heatmap_objective")
    heatmap_uses_latent_mse = heatmap_objective != "dsnt_js"
    heatmap_uses_diffusion_final_loss = (
        heatmap_objective == "diffusion" and heatmap_diffusion_final_loss_enabled
    )
    heatmap_supervision = (
        "full_resolution_dsnt_plus_js_after_frozen_decoder"
        if heatmap_objective == "dsnt_js"
        else "latent_diffusion_mse_plus_decoded_final_heatmap_loss"
        if heatmap_uses_diffusion_final_loss
        else "latent_diffusion_mse_against_frozen_cosmos_target"
    )
    temporal_heatmap_mode = _cfg_get_str(cfg.task, "temporal_heatmap_mode", "off")
    temporal_heatmap_label_source = (
        "temporal_window_dense_heatmap"
        if temporal_heatmap_mode != "off"
        else "single_point_gaussian_from_gaze_xy"
    )
    checks = {
        "robot_train_samples_positive": (
            not use_robot_data or robot_train_samples > 0
        ),
        "open_train_samples_positive_when_enabled": (
            not use_open_data or open_train_samples > 0
        ),
        "robot_train_dataloader_batches_positive": (
            not use_robot_data
            or robot_train_batches is None
            or robot_train_batches > 0
        ),
        "open_train_dataloader_batches_positive_when_enabled": (
            not use_open_data
            or open_train_batches is None
            or open_train_batches > 0
        ),
        "source_ratio_matches_configured_quota": resolved_quota_matches_request,
        # Kept as a compatibility key for old reports.  It now means that the
        # resolved source quotas match the active configuration, not 75/25.
        "source_ratio_75_25_when_mixed": resolved_quota_matches_request,
        "source_ratio_open_only_when_robot_disabled": (
            use_robot_data
            or not use_open_data
            or (
                _close_fraction(robot_ratio, 0.0)
                and _close_fraction(open_ratio, 1.0)
            )
        ),
        "robot_gaze_dropout_prob_0p2": _close_fraction(
            robot_gaze_dropout_prob,
            0.2,
        ),
        "robot_heatmap_on_gaze_dropout": robot_heatmap_on_gaze_dropout,
        "image_shape_256": image_shape == [3, 256, 256],
        "image_resize_modes_supported": all(
            mode in SUPPORTED_PREALIGNED_IMAGE_RESIZE_MODES
            for mode in image_resize_modes.values()
        ),
        "robot_image_size_matches_task": robot_image_size == task_image_size,
        "open_image_size_matches_task": open_image_size == task_image_size,
        "n_obs_steps_2": n_obs_steps == 2,
        "action_horizon_48": action_horizon == 48,
        "robot_sampling_matches_task": robot_sampling_matches_task,
        "open_sampling_matches_task": open_sampling_matches_task,
        "action_dim_10": action_dim == 10,
        "normalizer_source_robot_relative_when_robot_enabled": (
            not use_robot_data
            or normalizer_contract["source"] == "robot_dataset_relative_actions_only"
        ),
        "normalizer_source_matches_active_sources": (
            (
                use_robot_data
                and normalizer_contract["source"] == "robot_dataset_relative_actions_only"
            )
            or (
                not use_robot_data
                and normalizer_contract["source"]
                == "identity_action_placeholder_for_heatmap_only"
            )
        ),
        "normalizer_keys_match_robot_camera_and_action": (
            normalizer_contract["normalizer_keys"] == [camera_key, "action"]
        ),
        "normalizer_action_dim_10": normalizer_contract["action_dim"] == 10,
        "normalizer_excludes_open_dummy_actions": (
            normalizer_contract["excludes_open_source_dummy_actions"] is True
            and normalizer_contract["open_source_get_normalizer_allowed"] is False
        ),
        "data_stream_separate_zarr_sources": (
            data_stream_contract["separate_zarr_sources"] is True
        ),
        "data_stream_robot_dataset_class": (
            data_stream_contract["robot"]["dataset_class_matches_expected"] is True
        ),
        "data_stream_open_dataset_class": (
            data_stream_contract["open"]["dataset_class_matches_expected"] is True
        ),
        "data_stream_online_mixed_batch_builder": (
            data_stream_contract["source"] == "two_zarr_two_dataset_online_mixed_batch"
            and data_stream_contract["offline_merged_zarr"] is False
            and data_stream_contract["mixing"]["builder"]
            == (
                "diffusion_policy.dataset.gaze_wam_mixing."
                "build_gaze_wam_mixed_batch"
            )
            and data_stream_contract["mixing"]["mode"]
            == "online_per_step_concat_after_fetch"
            and data_stream_contract["mixing"]["ratio_source"]
            in (
                "robot_dataloader.batch_size/open_dataloader.batch_size",
                "data_mixing.total_batch_size_per_process+"
                "data_mixing.robot_ratio+data_mixing.open_ratio",
            )
        ),
        "heatmap_token_grid_16x16": heatmap_token_grid == [16, 16],
        "heatmap_num_tokens_256": heatmap_num_tokens == 256,
        "heatmap_dim_16": heatmap_dim == 16,
        "heatmap_dim_positive_latent_channels": heatmap_dim > 0,
        "heatmap_dim_not_lossless_patch_area": (
            heatmap_patch_area <= 0 or not heatmap_dim_matches_lossless_patch
        ),
        "heatmap_spatial_decoder_cosmos_tokenizer": (
            heatmap_spatial_decoder == "cosmos_tokenizer"
        ),
        "heatmap_distribution_mode_intensity_softplus": (
            heatmap_distribution_mode == "intensity_softplus"
        ),
        "heatmap_dsnt_temperature_0p1": _close_fraction(
            heatmap_dsnt_temperature,
            0.1,
        ),
        "heatmap_latent_scale_positive": heatmap_latent_scale > 0.0,
        "heatmap_latent_scaled_observed_range_within_clip": (
            not heatmap_latent_stats_path
            or not heatmap_latent_scaled_range["available"]
            or heatmap_latent_scaled_range["within_clip"]
        ),
        "heatmap_latent_stats_file_present_when_configured": (
            not heatmap_latent_stats_path or heatmap_latent_stats["exists"]
        ),
        "heatmap_scheduler_clip_sample_enabled": heatmap_scheduler_clip_sample is True,
        "heatmap_grid_product": (
            len(heatmap_token_grid) == 2
            and heatmap_token_grid[0] * heatmap_token_grid[1] == heatmap_num_tokens
        ),
        "obs_encoder_dinov3_vit16": _cfg_get_str(
            cfg.policy.obs_encoder,
            "model_name",
        )
        == "vit_base_patch16_dinov3",
        "obs_encoder_pretrained": _cfg_get_bool(
            cfg.policy.obs_encoder,
            "pretrained",
            False,
            "policy.obs_encoder.pretrained",
        ),
        "obs_encoder_source_valid": obs_encoder_source_valid,
        "obs_encoder_local_weight_source_optional": (
            not obs_encoder_local_weight_source_configured
            or (
                obs_encoder_local_weight_source_exists
                and obs_encoder_local_weight_source_valid
            )
        ),
        "obs_encoder_local_weight_source_exists": obs_encoder_local_weight_source_exists,
        "obs_encoder_local_weight_source_valid": obs_encoder_local_weight_source_valid,
        "obs_encoder_checkpoint_path_file_when_configured": (
            not obs_encoder_checkpoint_path or obs_encoder_checkpoint_is_file
        ),
        "obs_encoder_cache_dir_directory_when_configured": (
            not obs_encoder_cache_dir or obs_encoder_cache_dir_is_dir
        ),
        "robot_metadata_dataset_type_when_enabled": (
            not use_robot_data or robot_data_source["dataset_type_matches_expected"]
        ),
        "robot_metadata_image_resize_mode_when_enabled": (
            not use_robot_data
            or robot_data_source["image_resize_mode_matches_expected"]
        ),
        "robot_metadata_image_size_when_enabled": (
            not use_robot_data or robot_data_source["image_size_matches_expected"]
        ),
        "open_metadata_dataset_type_when_enabled": (
            open_batch_size <= 0 or open_data_source["dataset_type_matches_expected"]
        ),
        "open_metadata_image_resize_mode_when_enabled": (
            open_batch_size <= 0 or open_data_source["image_resize_mode_matches_expected"]
        ),
        "open_metadata_image_size_when_enabled": (
            open_batch_size <= 0 or open_data_source["image_size_matches_expected"]
        ),
        "block_attention_mask": _cfg_get_bool(
            cfg.policy,
            "use_block_attention_mask",
            False,
            "policy.use_block_attention_mask",
        ),
        "heatmap_objective_dsnt_js": heatmap_objective == "dsnt_js",
        "action_loss_weight_matches_active_sources": (
            action_loss_weight_matches_active_sources
        ),
        "action_loss_weight_1_when_robot_enabled": (
            not use_robot_data or _close_fraction(action_loss_weight, 1.0)
        ),
        "action_loss_weight_0_when_open_only": (
            use_robot_data or _close_fraction(action_loss_weight, 0.0)
        ),
        "heatmap_loss_weight_1": _close_fraction(heatmap_loss_weight, 1.0),
        "heatmap_token_kl_loss_weight_0": _close_fraction(
            heatmap_token_kl_loss_weight,
            0.0,
        ),
        "heatmap_point_nll_loss_weight_0": _close_fraction(
            heatmap_point_nll_loss_weight,
            0.0,
        ),
        "heatmap_xy_loss_weight_1": _close_fraction(heatmap_xy_loss_weight, 1.0),
        "heatmap_js_loss_weight_1": _close_fraction(heatmap_js_loss_weight, 1.0),
        "heatmap_diffusion_final_loss_enabled_valid": (
            not heatmap_diffusion_final_loss_enabled
            or heatmap_objective == "diffusion"
        ),
    }

    train_sequence_tokens = max_image_tokens + 1 + action_horizon + heatmap_num_tokens
    inference_sequence_tokens = max_image_tokens + 1 + action_horizon
    attention_contract = {
        "source": "config",
        "use_block_attention_mask": _cfg_get_bool(
            cfg.policy,
            "use_block_attention_mask",
            False,
            "policy.use_block_attention_mask",
        ),
        "num_image_tokens": int(max_image_tokens),
        "gaze_token_count": 1,
        "action_horizon": int(action_horizon),
        "heatmap_num_tokens": int(heatmap_num_tokens),
        "condition_reads_targets": False,
        "action_reads_heatmap": False,
        "action_reads_noisy_heatmap": False,
        "action_reads_heatmap_target": False,
        "heatmap_reads_action": False,
        "action_inference_drops_heatmap": True,
        "shared_world_kv_cache": False,
        "world_cache_consumed_by_action": False,
        "action_reads_heatmap_world_cache": False,
        "train_sequence_tokens": int(train_sequence_tokens),
        "inference_sequence_tokens": int(inference_sequence_tokens),
    }
    transformer = getattr(policy, "model", None) if policy is not None else None
    if transformer is not None and hasattr(transformer, "attention_contract_summary"):
        attention_contract = {
            "source": "model",
            **transformer.attention_contract_summary(num_image_tokens=max_image_tokens),
        }
    routing_contract = {
        "source": "config",
        "dynamic_head_freezing": False,
        "action_loss_mask": "(~is_open) & has_action",
        "heatmap_loss_mask": "has_heatmap",
        "open_rows": {
            "has_action": False,
            "has_heatmap": True,
            "use_gaze_condition": False,
            "gaze_token": "learned_mask",
            "trains_action": False,
            "trains_heatmap": True,
        },
        "robot_real_gaze_rows": {
            "is_open": False,
            "has_action": True,
            "use_gaze_condition": True,
            "has_heatmap": False,
            "trains_action": True,
            "trains_heatmap": False,
        },
        "robot_masked_gaze_rows": {
            "is_open": False,
            "has_action": True,
            "use_gaze_condition": False,
            "gaze_token": "learned_mask",
            "trains_action": True,
            "trains_heatmap": "has_heatmap",
        },
    }
    if policy is not None and hasattr(policy, "loss_routing_contract_summary"):
        routing_contract = policy.loss_routing_contract_summary()
    routing_contract = {
        **routing_contract,
        "robot_gaze_dropout_prob": robot_gaze_dropout_prob,
        "robot_heatmap_on_gaze_dropout": robot_heatmap_on_gaze_dropout,
    }
    checks["routing_validation_guardrails"] = (
        gaze_wam_loss_routing_validation_guardrails_ok(routing_contract)
    )
    return {
        "name": str(cfg.name),
        "task_name": str(cfg.task.name),
        "canonical_main_config_ok": all(checks.values()),
        "checks": checks,
        "data": {
            "robot_dataset_path": str(cfg.task.robot_dataset.dataset_path),
            "open_dataset_path": str(cfg.task.open_dataset.dataset_path),
            "robot_train_samples": robot_train_samples,
            "robot_val_samples": robot_val_samples,
            "open_train_samples": open_train_samples,
            "open_val_samples": open_val_samples,
            "requires_robot_train_samples": use_robot_data,
            "requires_open_train_samples": use_open_data,
            "allows_empty_validation_sets": True,
            "action_target_start_offset_steps": ACTION_TARGET_START_OFFSET_STEPS,
            "action_chunk_semantics": (
                "state@(t+1...t+H) relative to the latest observed state@t"
            ),
        },
        "data_sources": {
            "image_resize_modes": image_resize_modes,
            "image_sizes": image_sizes,
            "robot": robot_data_source,
            "open": open_data_source,
        },
        "data_stream": data_stream_contract,
        "optional_metadata": {
            "optional_keys": ["action_abs", "action_base_abs", "heatmap_image"],
            "presence_mask_keys": [
                "has_action_abs",
                "has_action_base_abs",
                "has_heatmap_image",
            ],
            "presence_mask_semantics": {
                "has_action_abs": (
                    "True only when every sampled target timestep in action_abs is available."
                ),
                "has_action_base_abs": "True when the current-row absolute action base is available.",
                "has_heatmap_image": "True when the current-row dense heatmap image is available.",
            },
            "samples": {
                "robot_train": _sample_optional_metadata_summary(robot_dataset),
                "robot_val": _sample_optional_metadata_summary(robot_val_dataset),
                "open_train": _sample_optional_metadata_summary(open_dataset),
                "open_val": _sample_optional_metadata_summary(open_val_dataset),
            },
        },
        "batching": {
            "requested_batch_size_source": str(
                training_config["batching"].get("batch_size_source", "auto")
            ),
            "resolved_batch_size_source": str(
                training_config["batching"].get(
                    "resolved_batch_size_source", "dataloader"
                )
            ),
            "compatibility_fallback_to_dataloader": bool(
                training_config["batching"].get(
                    "compatibility_fallback_to_dataloader", False
                )
            ),
            "ratio_fields_present": bool(
                training_config["batching"].get("ratio_fields_present", False)
            ),
            "requested_total_batch_size_per_process": training_config[
                "batching"
            ].get("requested_total_batch_size_per_process"),
            "requested_robot_ratio": training_config["batching"].get(
                "requested_robot_ratio"
            ),
            "requested_open_ratio": training_config["batching"].get(
                "requested_open_ratio"
            ),
            "configured_robot_dataloader_batch_size": int(
                training_config["batching"].get(
                    "configured_robot_dataloader_batch_size", robot_batch_size
                )
            ),
            "configured_open_dataloader_batch_size": int(
                training_config["batching"].get(
                    "configured_open_dataloader_batch_size", open_batch_size
                )
            ),
            "robot_batch_size_per_process": robot_batch_size,
            "open_batch_size_per_process": open_batch_size,
            "train_batch_size_per_process": total_batch_size,
            "num_processes": num_processes,
            "robot_ratio": float(robot_ratio),
            "open_ratio": float(open_ratio),
            "gradient_accumulate_every": gradient_accumulate_every,
            "effective_robot_batch_size_per_optimizer_step": effective_robot_batch_size,
            "effective_open_batch_size_per_optimizer_step": effective_open_batch_size,
            "effective_train_batch_size_per_optimizer_step": effective_train_batch_size,
            "mixed_precision": mixed_precision,
            "require_amp": require_amp,
            "distributed_type": str(distributed_type),
        },
        "batch_streaming": {
            "robot_train_enabled": use_robot_data,
            "open_train_enabled": open_batch_size > 0,
            "open_val_configured": (
                int(training_config["val_open_batch_size"]) > 0 and open_val_samples > 0
            ),
            "open_val_enabled": (
                int(training_config["val_open_batch_size"]) > 0
                and open_val_samples > 0
                and (open_val_batches is None or open_val_batches > 0)
            ),
            "open_iterator_policy": "restart_on_exhaustion",
            "open_iterator_caches_epoch_batches": False,
            "open_iterator_preserves_dataloader_shuffle_on_restart": True,
            "robot_iterator_policy": (
                "single_pass_epoch_driver" if use_robot_data else "disabled"
            ),
            "primary_epoch_driver": data_stream_contract["mixing"]["primary_epoch_driver"],
        },
        "dataloader_batches": {
            "robot_train_batches_per_epoch": robot_train_batches,
            "robot_val_batches_per_epoch": robot_val_batches,
            "open_train_batches_per_epoch": open_train_batches,
            "open_val_batches_per_epoch": open_val_batches,
        },
        "sampling": {
            "task": task_sampling,
            "robot_dataset": robot_sampling,
            "open_dataset": open_sampling,
            "compare_keys": sampling_compare_keys,
            "robot_matches_task": robot_sampling_matches_task,
            "open_matches_task": open_sampling_matches_task,
        },
        "training_config": training_config,
        "task_routing_config": task_routing_config,
        "schedule": {
            "stage": training_config["stage"],
            "epoch_driver": (
                "robot_dataloader"
                if use_robot_data
                else "open_dataloader"
                if use_open_data
                else "none"
            ),
            "epoch_definition": (
                "one pass over the robot train dataloader; open iterator restarts"
                if use_robot_data
                else "one pass over the open train dataloader"
            ),
            "epoch_driver_batches": (
                int(epoch_driver_batches)
                if epoch_driver_batches is not None
                else None
            ),
            "prepared_epoch_batches_per_process": (
                int(prepared_epoch_batches)
                if prepared_epoch_batches is not None
                else None
            ),
            "gradient_accumulate_every": gradient_accumulate_every,
            "gradient_accumulation_flush": "accelerate_sync_with_dataloader",
            "planned_optimizer_steps": (
                int(planned_optimizer_steps)
                if planned_optimizer_steps is not None
                else None
            ),
            "max_train_steps_scope": "global_optimizer_steps",
            "max_train_steps": training_config["max_train_steps"],
            "validation_primary_source": (
                "robot"
                if use_robot_data
                else "open"
                if use_open_data
                else "none"
            ),
            "open_pretrain_is_optional_stage": True,
        },
        "transfer": {
            "load": dict(transfer_load_summary or {}),
            "configured_export_path": training_config["transfer"][
                "export_path"
            ],
            "configured_export_scope": training_config["transfer"][
                "export_scope"
            ],
            "export_overwrite": training_config["transfer"][
                "export_overwrite"
            ],
            "export": None,
        },
        "routing": routing_contract,
        "loss": {
            "action_loss_weight": action_loss_weight,
            "heatmap_loss_weight": heatmap_loss_weight,
            "heatmap_token_kl_loss_weight": heatmap_token_kl_loss_weight,
            "heatmap_xy_loss_weight": heatmap_xy_loss_weight,
            "heatmap_point_nll_loss_weight": heatmap_point_nll_loss_weight,
            "heatmap_js_loss_weight": heatmap_js_loss_weight,
            "heatmap_dsnt_temperature": heatmap_dsnt_temperature,
            "heatmap_distribution_mode": heatmap_distribution_mode,
            "heatmap_dsnt_target_sigma_px": heatmap_dsnt_target_sigma_px,
            "heatmap_objective": heatmap_objective,
            "heatmap_supervision": heatmap_supervision,
            "latent_mse_loss": heatmap_uses_latent_mse,
            "diffusion_final_heatmap_loss": heatmap_uses_diffusion_final_loss,
            "heatmap_diffusion_final_loss_enabled": (
                heatmap_diffusion_final_loss_enabled
            ),
            "heatmap_final_loss_timestep_weighting": (
                heatmap_final_loss_timestep_weighting
            ),
            "heatmap_label_source": temporal_heatmap_label_source,
            "temporal_heatmap": {
                "mode": temporal_heatmap_mode,
                "window_radius": int(cfg.task.get("temporal_heatmap_window_radius", 0)),
                "beta": float(cfg.task.get("temporal_heatmap_beta", 0.0)),
                "sigma_px": float(cfg.task.get("temporal_heatmap_sigma_px", 0.0)),
                "current_weight": float(
                    cfg.task.get("temporal_heatmap_current_weight", 1.0)
                ),
            },
            "heatmap_decoder_output_interpretation": "decoded_intensity_distribution",
            "latent_mse_loss": heatmap_uses_latent_mse,
        },
        "normalizer": normalizer_contract,
        "tokens": {
            "image_shape": image_shape,
            "n_obs_steps": n_obs_steps,
            "image_tokens_per_frame": int(cfg.policy.image_tokens_per_frame),
            "visual_token_count": max_image_tokens,
            "gaze_token_count": 1,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "heatmap_token_grid": heatmap_token_grid,
            "heatmap_num_tokens": heatmap_num_tokens,
            "heatmap_dim": int(cfg.policy.heatmap_dim),
            "heatmap_representation": "channel_latent_decoder_to_full_resolution",
            "heatmap_spatial_decoder": heatmap_spatial_decoder,
            "heatmap_latent_grid": [
                int(heatmap_token_grid[0]),
                int(heatmap_token_grid[1]),
            ],
            "heatmap_latent_channels": int(cfg.policy.heatmap_dim),
            "heatmap_latent_scale": heatmap_latent_scale,
            "heatmap_latent_offset": heatmap_latent_offset,
            "heatmap_latent_stats": heatmap_latent_stats,
            "heatmap_latent_scaled_range": heatmap_latent_scaled_range,
            "heatmap_scheduler_clip_sample": heatmap_scheduler_clip_sample,
            "heatmap_patch_area": int(heatmap_patch_area),
            "heatmap_decoded_image_size": task_image_size,
            "train_sequence_tokens": train_sequence_tokens,
            "inference_sequence_tokens": inference_sequence_tokens,
        },
        "model": {
            "obs_encoder_model_name": _cfg_get_str(cfg.policy.obs_encoder, "model_name"),
            "obs_encoder_pretrained": _cfg_get_bool(
                cfg.policy.obs_encoder,
                "pretrained",
                False,
                "policy.obs_encoder.pretrained",
            ),
            "obs_encoder_checkpoint_path": obs_encoder_checkpoint_path,
            "obs_encoder_checkpoint_path_exists": obs_encoder_checkpoint_exists,
            "obs_encoder_checkpoint_path_is_file": obs_encoder_checkpoint_is_file,
            "obs_encoder_cache_dir": obs_encoder_cache_dir,
            "obs_encoder_cache_dir_exists": obs_encoder_cache_dir_exists,
            "obs_encoder_cache_dir_is_dir": obs_encoder_cache_dir_is_dir,
            "obs_encoder_local_weight_source_configured": obs_encoder_local_weight_source_configured,
            "obs_encoder_local_weight_source_exists": obs_encoder_local_weight_source_exists,
            "obs_encoder_local_weight_source_valid": obs_encoder_local_weight_source_valid,
            "use_block_attention_mask": _cfg_get_bool(
                cfg.policy,
                "use_block_attention_mask",
                False,
                "policy.use_block_attention_mask",
            ),
            "use_frame_embedding": _cfg_get_bool(
                cfg.policy,
                "use_frame_embedding",
                False,
                "policy.use_frame_embedding",
            ),
            "num_inference_steps": int(cfg.policy.num_inference_steps),
            "cfg_scale": cfg_scale,
        },
        "attention": attention_contract,
    }


def _write_training_contract_summary(summary, output_dir: str) -> str:
    path = pathlib.Path(output_dir) / "training_contract.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _stamp_training_scale_into_cfg(cfg, training_contract):
    """Persist launcher/contract training-scale evidence into checkpoint cfg."""
    batching = training_contract.get("batching", {})
    if "data_mixing" in cfg:
        with open_dict(cfg.data_mixing):
            cfg.data_mixing.requested_batch_size_source = str(
                batching.get("requested_batch_size_source", "auto")
            )
            cfg.data_mixing.resolved_batch_size_source = str(
                batching.get("resolved_batch_size_source", "dataloader")
            )
            cfg.data_mixing.compatibility_fallback_to_dataloader = bool(
                batching.get("compatibility_fallback_to_dataloader", False)
            )
            requested_total = batching.get(
                "requested_total_batch_size_per_process", None
            )
            cfg.data_mixing.requested_total_batch_size_per_process = (
                None if requested_total is None else int(requested_total)
            )
            requested_robot_ratio = batching.get("requested_robot_ratio", None)
            cfg.data_mixing.requested_robot_ratio = (
                None
                if requested_robot_ratio is None
                else float(requested_robot_ratio)
            )
            requested_open_ratio = batching.get("requested_open_ratio", None)
            cfg.data_mixing.requested_open_ratio = (
                None if requested_open_ratio is None else float(requested_open_ratio)
            )
    with open_dict(cfg.training):
        cfg.training.robot_batch_size_per_process = int(
            batching.get("robot_batch_size_per_process", 0)
        )
        cfg.training.open_batch_size_per_process = int(
            batching.get("open_batch_size_per_process", 0)
        )
        cfg.training.train_batch_size_per_process = int(
            batching.get("train_batch_size_per_process", 0)
        )
        cfg.training.robot_ratio = float(batching.get("robot_ratio", 0.0))
        cfg.training.open_ratio = float(batching.get("open_ratio", 0.0))
        cfg.training.num_processes = int(batching.get("num_processes", 1))
        cfg.training.mixed_precision = str(batching.get("mixed_precision", "no"))
        cfg.training.distributed_type = str(batching.get("distributed_type", "NO"))
        cfg.training.effective_robot_batch_size_per_optimizer_step = int(
            batching.get("effective_robot_batch_size_per_optimizer_step", 0)
        )
        cfg.training.effective_open_batch_size_per_optimizer_step = int(
            batching.get("effective_open_batch_size_per_optimizer_step", 0)
        )
        cfg.training.effective_train_batch_size_per_optimizer_step = int(
            batching.get("effective_train_batch_size_per_optimizer_step", 0)
        )
    return cfg


def _imwrite_unicode(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for '{path}'.")
    encoded.tofile(str(path))


def _heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - heatmap.min(initial=0.0)
    denom = heatmap.max(initial=0.0)
    if denom > 1e-12:
        heatmap = heatmap / denom
    return (heatmap * 255.0).round().astype(np.uint8)


def _latest_obs_rgb(batch, camera_key: str) -> np.ndarray:
    image = batch["obs"][camera_key][0, -1].detach().float().cpu().numpy()
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def _draw_gaze(image_rgb: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    out = image_rgb.copy()
    if not valid:
        return out
    h, w = out.shape[:2]
    x = int(round(float(gaze_xy[0]) * (w - 1)))
    y = int(round(float(gaze_xy[1]) * (h - 1)))
    cv2.drawMarker(
        out,
        (x, y),
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(8, min(h, w) // 10),
        thickness=2,
    )
    cv2.circle(out, (x, y), radius=max(2, min(h, w) // 40), color=(255, 0, 0), thickness=-1)
    return out


def _overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, heat_color_rgb, 0.45, 0.0)
    return _draw_gaze(overlay, gaze_xy=gaze_xy, valid=valid)


def _heatmap_panel(heatmap: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    return _draw_gaze(heat_color_rgb, gaze_xy=gaze_xy, valid=valid)


def _title_panel(image_rgb: np.ndarray, title: str) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]
    bar_h = max(22, min(32, h // 9))
    cv2.rectangle(out, (0, 0), (w, bar_h), color=(0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        str(title),
        (8, max(16, bar_h - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, min(0.65, h / 420.0)),
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


def _heatmap_comparison_strip(
    image_rgb: np.ndarray,
    pred_heatmap: np.ndarray,
    target_heatmap: np.ndarray,
    gaze_xy,
    valid: bool,
) -> np.ndarray:
    panels = [
        _title_panel(_draw_gaze(image_rgb, gaze_xy, valid), "RGB + gaze_xy"),
        _title_panel(
            _heatmap_panel(pred_heatmap, gaze_xy, valid),
            "Pred heatmap",
        ),
        _title_panel(
            _overlay_heatmap(image_rgb, pred_heatmap, gaze_xy, valid),
            "Pred overlay",
        ),
        _title_panel(
            _heatmap_panel(target_heatmap, gaze_xy, valid),
            "Target heatmap",
        ),
        _title_panel(
            _overlay_heatmap(image_rgb, target_heatmap, gaze_xy, valid),
            "Target overlay",
        ),
    ]
    separator = np.full(
        (panels[0].shape[0], max(4, panels[0].shape[1] // 64), 3),
        255,
        dtype=panels[0].dtype,
    )
    strip_panels = []
    for idx, panel in enumerate(panels):
        if idx > 0:
            strip_panels.append(separator)
        strip_panels.append(panel)
    return np.concatenate(strip_panels, axis=1)


def _select_heatmap_preview_indices(
    preview_indices: torch.Tensor,
    max_samples: int,
    sample_seed=None,
) -> torch.Tensor:
    max_samples = max(1, int(max_samples))
    if preview_indices.numel() <= max_samples:
        return preview_indices[:max_samples]
    if sample_seed is None:
        return preview_indices[:max_samples]
    seed = int(sample_seed)
    order = torch.randperm(int(preview_indices.numel()), generator=_make_cpu_generator(seed))
    order = order[:max_samples].to(device=preview_indices.device)
    return preview_indices[order]


def _write_heatmap_preview(
    policy: GazeWamPolicy,
    batch,
    output_dir: str,
    epoch: int,
    camera_key: str,
    max_samples: int = 1,
    sample_seed=None,
):
    preview_indices = torch.nonzero(batch["has_heatmap"], as_tuple=False).reshape(-1)
    if preview_indices.numel() == 0:
        return None
    max_samples = max(1, int(max_samples))
    selected_preview_indices = _select_heatmap_preview_indices(
        preview_indices=preview_indices,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    preview_batch = _slice_batch(batch, selected_preview_indices)
    obs = dict(preview_batch["obs"])
    obs["gaze_xy"] = preview_batch["gaze_xy"]
    obs["has_gaze_label"] = preview_batch["has_gaze_label"]
    obs["use_gaze_condition"] = preview_batch["use_gaze_condition"]

    pred = policy.predict_heatmap(obs, decode=True)
    target_mask = preview_batch["has_heatmap"].to(policy.device) & preview_batch[
        "has_gaze_label"
    ].to(policy.device)
    target_image = policy._target_heatmap_image_from_batch_or_xy(
        batch=preview_batch,
        gaze_xy=preview_batch["gaze_xy"].to(device=policy.device, dtype=policy.dtype),
        valid_mask=target_mask,
    )
    target_tokens = policy._heatmap_image_to_training_tokens(
        target_image,
    )

    preview_dir = pathlib.Path(output_dir) / "media" / "val_heatmap" / f"epoch_{epoch:04d}"
    summary_path = preview_dir / "summary.json"

    sample_summaries = []
    legacy_paths = None
    pred_images = pred["heatmap_image"].detach().float().cpu().numpy()
    target_images = target_image.detach().float().cpu().numpy()
    for sample_idx in range(preview_batch["gaze_xy"].shape[0]):
        sample_batch = _slice_batch(
            preview_batch,
            torch.tensor([sample_idx], device=preview_batch["gaze_xy"].device),
        )
        image_rgb = _latest_obs_rgb(sample_batch, camera_key=camera_key)
        gaze_xy = sample_batch["gaze_xy"][0].detach().cpu().numpy()
        has_gaze = bool(sample_batch["has_gaze_label"][0].item())
        pred_image = pred_images[sample_idx]
        one_target_image = target_images[sample_idx]

        sample_dir = preview_dir / f"sample_{sample_idx:03d}"
        pred_heatmap_path = sample_dir / "pred_heatmap.png"
        target_heatmap_path = sample_dir / "target_heatmap.png"
        pred_overlay_path = sample_dir / "pred_overlay.png"
        target_overlay_path = sample_dir / "target_overlay.png"
        comparison_path = sample_dir / "comparison.png"
        rgb_path = sample_dir / "rgb.png"

        _imwrite_unicode(
            rgb_path,
            cv2.cvtColor(_draw_gaze(image_rgb, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR),
        )
        _imwrite_unicode(
            pred_heatmap_path,
            cv2.applyColorMap(_heatmap_to_uint8(pred_image), cv2.COLORMAP_JET),
        )
        _imwrite_unicode(
            target_heatmap_path,
            cv2.applyColorMap(_heatmap_to_uint8(one_target_image), cv2.COLORMAP_JET),
        )
        _imwrite_unicode(
            pred_overlay_path,
            cv2.cvtColor(
                _overlay_heatmap(image_rgb, pred_image, gaze_xy, has_gaze),
                cv2.COLOR_RGB2BGR,
            ),
        )
        _imwrite_unicode(
            target_overlay_path,
            cv2.cvtColor(
                _overlay_heatmap(image_rgb, one_target_image, gaze_xy, has_gaze),
                cv2.COLOR_RGB2BGR,
            ),
        )
        _imwrite_unicode(
            comparison_path,
            cv2.cvtColor(
                _heatmap_comparison_strip(
                    image_rgb,
                    pred_image,
                    one_target_image,
                    gaze_xy,
                    has_gaze,
                ),
                cv2.COLOR_RGB2BGR,
            ),
        )

        paths = {
            "rgb": str(rgb_path),
            "pred_heatmap": str(pred_heatmap_path),
            "target_heatmap": str(target_heatmap_path),
            "pred_overlay": str(pred_overlay_path),
            "target_overlay": str(target_overlay_path),
            "comparison": str(comparison_path),
        }
        if sample_idx == 0:
            legacy_paths = {
                "rgb": preview_dir / "rgb.png",
                "pred_heatmap": preview_dir / "pred_heatmap.png",
                "target_heatmap": preview_dir / "target_heatmap.png",
                "pred_overlay": preview_dir / "pred_overlay.png",
                "target_overlay": preview_dir / "target_overlay.png",
                "comparison": preview_dir / "comparison.png",
            }
            _imwrite_unicode(
                legacy_paths["rgb"],
                cv2.cvtColor(_draw_gaze(image_rgb, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR),
            )
            _imwrite_unicode(
                legacy_paths["pred_heatmap"],
                cv2.applyColorMap(_heatmap_to_uint8(pred_image), cv2.COLORMAP_JET),
            )
            _imwrite_unicode(
                legacy_paths["target_heatmap"],
                cv2.applyColorMap(_heatmap_to_uint8(one_target_image), cv2.COLORMAP_JET),
            )
            _imwrite_unicode(
                legacy_paths["pred_overlay"],
                cv2.cvtColor(
                    _overlay_heatmap(image_rgb, pred_image, gaze_xy, has_gaze),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            _imwrite_unicode(
                legacy_paths["target_overlay"],
                cv2.cvtColor(
                    _overlay_heatmap(image_rgb, one_target_image, gaze_xy, has_gaze),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            _imwrite_unicode(
                legacy_paths["comparison"],
                cv2.cvtColor(
                    _heatmap_comparison_strip(
                        image_rgb,
                        pred_image,
                        one_target_image,
                        gaze_xy,
                        has_gaze,
                    ),
                    cv2.COLOR_RGB2BGR,
                ),
            )

        sample_summaries.append(
            {
                "index": int(sample_idx),
                "batch_index": int(selected_preview_indices[sample_idx].item()),
                "gaze_xy": [float(v) for v in gaze_xy.tolist()],
                "has_gaze_label": has_gaze,
                "use_gaze_condition": bool(sample_batch["use_gaze_condition"][0].item()),
                "is_open": bool(sample_batch["is_open"][0].item()),
                "pred_heatmap_shape": [int(v) for v in pred_image.shape],
                "target_heatmap_shape": [int(v) for v in one_target_image.shape],
                "pred_token_argmax": int(
                    pred["heatmap_tokens"][sample_idx].reshape(-1).argmax().item()
                ),
                "target_token_argmax": int(
                    target_tokens[sample_idx].reshape(-1).argmax().item()
                ),
                "paths": paths,
            }
        )

    heatmap_objective = str(policy.heatmap_objective)
    latent_mse_loss = heatmap_objective != "dsnt_js"
    diffusion_final_heatmap_loss = bool(
        heatmap_objective == "diffusion"
        and getattr(policy, "heatmap_diffusion_final_loss_enabled", False)
    )
    heatmap_supervision = (
        "full_resolution_dsnt_plus_js_after_frozen_decoder"
        if heatmap_objective == "dsnt_js"
        else "latent_diffusion_mse_plus_decoded_final_heatmap_loss"
        if diffusion_final_heatmap_loss
        else "latent_diffusion_mse_against_frozen_cosmos_target"
    )
    summary = {
        "epoch": int(epoch),
        "camera_key": camera_key,
        "num_samples": len(sample_summaries),
        "max_samples": max_samples,
        "sample_seed": None if sample_seed is None else int(sample_seed),
        "heatmap_objective": heatmap_objective,
        "heatmap_supervision": heatmap_supervision,
        "latent_mse_loss": latent_mse_loss,
        "diffusion_final_heatmap_loss": diffusion_final_heatmap_loss,
        "heatmap_diffusion_final_loss_enabled": bool(
            getattr(policy, "heatmap_diffusion_final_loss_enabled", False)
        ),
        "heatmap_final_loss_timestep_weighting": str(
            getattr(policy, "heatmap_final_loss_timestep_weighting", "none")
        ),
        "selected_batch_indices": [
            int(v) for v in selected_preview_indices.detach().cpu().tolist()
        ],
        "samples": sample_summaries,
    }
    if sample_summaries:
        first = sample_summaries[0]
        summary.update(
            {
                "gaze_xy": first["gaze_xy"],
                "has_gaze_label": first["has_gaze_label"],
                "use_gaze_condition": first["use_gaze_condition"],
                "pred_heatmap_shape": first["pred_heatmap_shape"],
                "target_heatmap_shape": first["target_heatmap_shape"],
                "pred_token_argmax": first["pred_token_argmax"],
                "target_token_argmax": first["target_token_argmax"],
                "paths": {
                    **{key: str(value) for key, value in legacy_paths.items()},
                    "summary": str(summary_path),
                    "samples": [sample["paths"] for sample in sample_summaries],
                },
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class TrainGazeWamWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg = _normalize_gaze_wam_early_bool_config(copy.deepcopy(cfg))
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: GazeWamPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: GazeWamPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0

        if not cfg.training.resume:
            self.exclude_keys = ["optimizer"]

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # Apply debug limits before validation, scheduler construction, and the
        # persisted training contract so smoke runs report their real budget.
        if cfg.training.debug:
            if cfg.training.num_epochs is None or cfg.training.num_epochs > 2:
                cfg.training.num_epochs = 2
            if cfg.training.max_train_steps is None or cfg.training.max_train_steps > 3:
                cfg.training.max_train_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.sample_every = 1
            cfg.training.val_every = 1
            if cfg.training.max_val_steps is None or cfg.training.max_val_steps > 2:
                cfg.training.max_val_steps = 2
        training_config = validate_gaze_wam_training_config(cfg)
        task_routing_config = validate_gaze_wam_task_routing_config(cfg)
        if not training_config["valid"] or not task_routing_config["valid"]:
            raise ValueError(
                "Invalid Gaze-WAM training/task routing config: "
                + "; ".join(
                    list(training_config["errors"])
                    + list(task_routing_config["errors"])
                )
            )
        cfg = _normalize_gaze_wam_training_config(cfg, training_config)
        cfg = _normalize_gaze_wam_task_routing_config(cfg, task_routing_config)

        transfer_load_summary = None
        transfer_load_path = str(training_config["transfer"]["load_path"] or "")
        if transfer_load_path:
            transfer_load_summary = load_gaze_wam_transfer_artifact(
                self.model,
                hydra.utils.to_absolute_path(transfer_load_path),
                scope=training_config["transfer"]["load_scope"],
            )
            if self.ema_model is not None:
                load_gaze_wam_transfer_artifact(
                    self.ema_model,
                    hydra.utils.to_absolute_path(transfer_load_path),
                    scope=training_config["transfer"]["load_scope"],
                )
            print(
                "Loaded Gaze-WAM transfer artifact: "
                f"{transfer_load_summary['path']} "
                f"scope={transfer_load_summary['scope']}"
            )

        accelerator = Accelerator(
            log_with="wandb",
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
            ],
        )
        _validate_gaze_wam_accumulation_flush_contract(accelerator)
        require_amp = bool(cfg.training.get("require_amp", True))
        mixed_precision = str(getattr(accelerator, "mixed_precision", "no") or "no")
        if require_amp and mixed_precision not in ("bf16", "fp16"):
            raise RuntimeError(
                "Gaze-WAM training requires AMP. Launch with "
                "`accelerate launch --mixed_precision bf16 train.py ...` or use "
                "`train_scripts/train_gaze_wam_open_only_amp.ps1` for the HOT3D open-only path. "
                "Set `training.require_amp=false` only for short CPU/shape debug runs."
            )
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path, trust_checkpoint=True)
                resume_epoch = cfg.training.get("resume_epoch", None)
                if resume_epoch is not None:
                    self.epoch = int(resume_epoch)

        robot_batch_size = cfg.robot_dataloader.batch_size
        open_batch_size = cfg.open_dataloader.batch_size
        val_robot_batch_size = cfg.val_robot_dataloader.batch_size
        open_val_batch_size = cfg.val_open_dataloader.batch_size
        val_open_batch_size = open_val_batch_size
        use_robot_data = robot_batch_size > 0
        use_open_data = open_batch_size > 0
        robot_dataset = None
        robot_val_dataset = None
        robot_dataloader = None
        robot_val_dataloader = None
        if use_robot_data:
            robot_dataset = hydra.utils.instantiate(cfg.task.robot_dataset)
            robot_val_dataset = robot_dataset.get_validation_dataset()
        open_dataloader = None
        open_val_dataloader = None
        open_dataset = None
        open_val_dataset = None
        use_open_val_data = False
        if use_open_data:
            open_dataset = hydra.utils.instantiate(cfg.task.open_dataset)
            open_val_dataset = open_dataset.get_validation_dataset()
            use_open_val_data = open_val_batch_size > 0 and len(open_val_dataset) > 0
            if use_open_val_data:
                open_val_dataloader = build_gaze_wam_dataloader(
                    open_val_dataset,
                    cfg.val_open_dataloader,
                    batch_size=open_val_batch_size,
                    tail_policy=training_config["batching"][
                        "validation_tail_policy"
                    ],
                    source_name="open_val",
                )
        use_robot_val_data = (
            use_robot_data
            and val_robot_batch_size > 0
            and robot_val_dataset is not None
        )

        _check_training_dataset_lengths(
            robot_dataset=robot_dataset,
            robot_val_dataset=robot_val_dataset,
            open_dataset=open_dataset,
            open_val_dataset=open_val_dataset,
            robot_batch_size=robot_batch_size,
            open_batch_size=open_batch_size,
        )
        if use_robot_data:
            robot_dataloader = build_gaze_wam_dataloader(
                robot_dataset,
                cfg.robot_dataloader,
                batch_size=robot_batch_size,
                tail_policy=training_config["batching"]["robot_tail_policy"],
                source_name="robot_train",
            )
        if use_robot_val_data:
            robot_val_dataloader = build_gaze_wam_dataloader(
                robot_val_dataset,
                cfg.val_robot_dataloader,
                batch_size=val_robot_batch_size,
                tail_policy=training_config["batching"][
                    "validation_tail_policy"
                ],
                source_name="robot_val",
            )
        if use_open_data:
            open_dataloader = build_gaze_wam_dataloader(
                open_dataset,
                cfg.open_dataloader,
                batch_size=open_batch_size,
                tail_policy=training_config["batching"]["open_tail_policy"],
                source_name="open_train",
            )
        dataloader_lengths = _check_training_dataloader_lengths(
            robot_dataloader=robot_dataloader,
            robot_val_dataloader=robot_val_dataloader,
            open_dataloader=open_dataloader,
            open_val_dataloader=open_val_dataloader,
            robot_batch_size=robot_batch_size,
            open_batch_size=open_batch_size,
        )
        if (
            use_open_val_data
            and dataloader_lengths.get("open_val_batches") is not None
            and dataloader_lengths["open_val_batches"] <= 0
        ):
            use_open_val_data = False
            open_val_dataloader = None
            dataloader_lengths["open_val_batches"] = 0

        os.makedirs(self.output_dir, exist_ok=True)
        epoch_driver_batches = (
            dataloader_lengths["robot_train_batches"]
            if use_robot_data
            else dataloader_lengths["open_train_batches"]
        )
        scheduler_step_multiplier = (
            int(accelerator.num_processes)
            if not bool(accelerator.split_batches)
            else 1
        )
        train_dataloader_for_planning = (
            robot_dataloader if use_robot_data else open_dataloader
        )
        if train_dataloader_for_planning is None:
            raise ValueError("The selected Gaze-WAM epoch driver dataloader is missing.")
        prepared_epoch_batches = _planned_prepared_epoch_batches(
            train_dataloader_for_planning,
            accelerator,
        )
        if prepared_epoch_batches <= 0:
            raise ValueError(
                "The selected Gaze-WAM epoch driver dataloader has no prepared batches."
            )
        planned_optimizer_steps = gaze_wam_planned_optimizer_steps(
            steps_per_epoch=prepared_epoch_batches,
            num_epochs=cfg.training.num_epochs,
            gradient_accumulate_every=cfg.training.gradient_accumulate_every,
            max_train_steps=cfg.training.max_train_steps,
        )
        training_contract = _build_training_contract_summary(
            cfg=cfg,
            robot_dataset=robot_dataset,
            robot_val_dataset=robot_val_dataset,
            open_dataset=open_dataset,
            open_val_dataset=open_val_dataset,
            dataloader_lengths=dataloader_lengths,
            policy=self.model,
            accelerator=accelerator,
            transfer_load_summary=transfer_load_summary,
            planned_optimizer_steps=planned_optimizer_steps,
            epoch_driver_batches=epoch_driver_batches,
            prepared_epoch_batches=prepared_epoch_batches,
        )
        training_contract_path = str(pathlib.Path(self.output_dir) / "training_contract.json")
        if accelerator.is_main_process:
            _write_training_contract_summary(training_contract, self.output_dir)
        training_contract_log_fields = {
            "training_contract_path": training_contract_path or "",
            "training_contract_canonical_main_config_ok": bool(
                training_contract["canonical_main_config_ok"]
            ),
            "training_contract_robot_ratio": float(
                training_contract["batching"]["robot_ratio"]
            ),
            "training_contract_open_ratio": float(
                training_contract["batching"]["open_ratio"]
            ),
            "training_contract_num_processes": int(
                training_contract["batching"]["num_processes"]
            ),
            "training_contract_effective_train_batch_size": int(
                training_contract["batching"][
                    "effective_train_batch_size_per_optimizer_step"
                ]
            ),
            "training_contract_robot_gaze_dropout_prob": float(
                training_contract["routing"]["robot_gaze_dropout_prob"]
            ),
            "training_contract_robot_heatmap_on_gaze_dropout": int(
                training_contract["routing"]["robot_heatmap_on_gaze_dropout"]
            ),
            "training_contract_stage": str(
                training_contract["schedule"]["stage"]
            ),
            "training_contract_epoch_driver": str(
                training_contract["schedule"]["epoch_driver"]
            ),
            "training_contract_max_train_steps": int(
                training_contract["schedule"]["max_train_steps"]
                or 0
            ),
            "training_contract_prepared_epoch_batches": int(
                prepared_epoch_batches
            ),
        }
        training_contract_log_pending = True
        cfg = _stamp_training_scale_into_cfg(cfg, training_contract)
        self.cfg = cfg
        normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
        normalizer_state_path = os.path.join(self.output_dir, "normalizer_state.pt")
        if accelerator.is_main_process:
            if use_robot_data:
                normalizer = robot_dataset.get_normalizer()
            else:
                normalizer = _make_heatmap_only_identity_normalizer(cfg.task.camera_key)
            with open(normalizer_path, "wb") as normalizer_file:
                pickle.dump(normalizer, normalizer_file)
            torch.save(normalizer.state_dict(), normalizer_state_path)

        accelerator.wait_for_everyone()
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(
            torch.load(
                normalizer_state_path,
                map_location="cpu",
                weights_only=True,
            )
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        train_epoch_dataloader = robot_dataloader if use_robot_data else open_dataloader
        if train_epoch_dataloader is None:
            raise ValueError(
                "No train dataloader was constructed. Enable robot_dataloader.batch_size "
                "or open_dataloader.batch_size before policy training."
            )
        if cfg.training.resume and self.global_step > 0:
            _ensure_optimizer_initial_lr_for_resume(
                self.optimizer,
                base_lr=cfg.optimizer.lr,
                obs_encoder_lr=cfg.optimizer.obs_encoder_lr,
            )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps)
            * (
                scheduler_step_multiplier
            ),
            num_training_steps=int(planned_optimizer_steps)
            * (
                scheduler_step_multiplier
            ),
            last_epoch=self.global_step * scheduler_step_multiplier - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        prepare_names = []
        prepare_items = []

        def add_prepare_item(name, value):
            if value is None:
                return
            prepare_names.append(name)
            prepare_items.append(value)

        add_prepare_item("robot_dataloader", robot_dataloader)
        add_prepare_item("robot_val_dataloader", robot_val_dataloader)
        add_prepare_item("open_dataloader", open_dataloader)
        add_prepare_item("open_val_dataloader", open_val_dataloader)
        add_prepare_item("model", self.model)
        add_prepare_item("optimizer", self.optimizer)
        add_prepare_item("lr_scheduler", lr_scheduler)
        prepared_items = accelerator.prepare(*prepare_items)
        if len(prepare_items) == 1:
            prepared_items = (prepared_items,)
        prepared = dict(zip(prepare_names, prepared_items))
        robot_dataloader = prepared.get("robot_dataloader")
        robot_val_dataloader = prepared.get("robot_val_dataloader")
        open_dataloader = prepared.get("open_dataloader")
        open_val_dataloader = prepared.get("open_val_dataloader")
        self.model = prepared["model"]
        self.optimizer = prepared["optimizer"]
        lr_scheduler = prepared["lr_scheduler"]
        train_epoch_dataloader = robot_dataloader if use_robot_data else open_dataloader
        if train_epoch_dataloader is None:
            raise ValueError(
                "Accelerate.prepare returned no epoch-driver dataloader."
            )
        actual_prepared_epoch_batches = _validate_prepared_epoch_driver_length(
            planned_batches=prepared_epoch_batches,
            actual_batches=len(train_epoch_dataloader),
        )
        training_contract["schedule"][
            "actual_prepared_epoch_batches_per_process"
        ] = actual_prepared_epoch_batches
        training_contract_log_fields[
            "training_contract_actual_prepared_epoch_batches"
        ] = actual_prepared_epoch_batches
        if accelerator.is_main_process:
            _write_training_contract_summary(training_contract, self.output_dir)
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        already_at_max_train_steps = (
            cfg.training.max_train_steps is not None
            and self.global_step >= cfg.training.max_train_steps
        )
        if already_at_max_train_steps:
            print(
                "Global training step budget already reached: "
                f"global_step={self.global_step}, "
                f"max_train_steps={cfg.training.max_train_steps}."
            )

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        json_logger_cls = JsonLogger if accelerator.is_main_process else _NullJsonLogger
        with json_logger_cls(log_path) as json_logger:
            while (
                not already_at_max_train_steps
                and self.epoch < cfg.training.num_epochs
            ):
                self.model.train()
                if cfg.training.freeze_encoder:
                    policy_module = accelerator.unwrap_model(self.model)
                    policy_module.obs_encoder.eval()
                    policy_module.obs_encoder.requires_grad_(False)

                step_log = {}
                step_log_has_optimizer_step = False
                stop_after_epoch = False
                train_window_log = _new_train_window_log_accumulator()
                open_iter = (
                    _RestartingDataLoaderIterator(open_dataloader, "open train")
                    if use_robot_data and use_open_data
                    else None
                )
                with tqdm.tqdm(
                    train_epoch_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, source_batch in enumerate(tepoch):
                        source_batch = dict_apply(
                            source_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        robot_batch = source_batch if use_robot_data else None
                        open_batch = None
                        if use_robot_data and use_open_data:
                            open_batch = open_iter.next()
                            open_batch = dict_apply(
                                open_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                        elif not use_robot_data:
                            open_batch = source_batch
                        batch = build_gaze_wam_mixed_batch(
                            robot_batch=robot_batch,
                            open_batch=open_batch,
                            robot_gaze_dropout_prob=cfg.task.robot_gaze_dropout_prob,
                            robot_heatmap_on_gaze_dropout=cfg.task.get(
                                "robot_heatmap_on_gaze_dropout",
                                True,
                            ),
                            shuffle=True,
                        )

                        optimizer_step_completed = False
                        with accelerator.accumulate(self.model):
                            components = self.model(
                                batch,
                                return_per_sample=True,
                            )
                            raw_loss = components["loss"]
                            accelerator.backward(raw_loss)

                            if accelerator.sync_gradients:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                                optimizer_step_completed = True

                                if cfg.training.use_ema:
                                    ema.step(accelerator.unwrap_model(self.model))

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_routing = loss_routing_summary(
                            mixed=batch,
                            action_loss_mask=components["action_loss_mask"],
                            heatmap_loss_mask=components["heatmap_loss_mask"],
                            use_distributed_counts=True,
                        )
                        _accumulate_train_window_log(
                            train_window_log,
                            raw_loss,
                            components,
                            train_routing,
                        )
                        current_step_log = {}
                        if optimizer_step_completed:
                            current_step_log = _finalize_train_window_log(train_window_log)
                            current_step_log.update(
                                {
                                    "global_step": self.global_step,
                                    "epoch": self.epoch,
                                    "lr": lr_scheduler.get_last_lr()[0],
                                }
                            )
                        gdr_every = cfg.training.get("gdr_every", 0)
                        should_log_gdr = (
                            optimizer_step_completed
                            and gdr_every is not None
                            and gdr_every > 0
                            and self.global_step % gdr_every == 0
                        )
                        if should_log_gdr:
                            metric_mask = (
                                batch["has_action"]
                                & batch["use_gaze_condition"]
                                & (~batch["is_open"])
                            )
                            global_gdr_count = distributed_mask_count(metric_mask)
                            if global_gdr_count.item() > 0:
                                local_feature_gdr_sum = torch.zeros(
                                    (),
                                    device=metric_mask.device,
                                    dtype=torch.float32,
                                )
                                local_output_gdr_sum = torch.zeros(
                                    (),
                                    device=metric_mask.device,
                                    dtype=torch.float32,
                                )
                            if metric_mask.any():
                                metric_batch = _slice_batch(batch, metric_mask)
                                with torch.no_grad():
                                    gdr = accelerator.unwrap_model(
                                        self.model
                                    ).compute_gaze_dependency_ratio(
                                        _obs_for_metric(metric_batch)
                                    )
                                local_feature_gdr_sum = gdr["feature_gdr"].detach().float().sum()
                                local_output_gdr_sum = gdr["output_gdr"].detach().float().sum()
                            if global_gdr_count.item() > 0:
                                global_feature_gdr_sum = _distributed_scalar_sum(
                                    local_feature_gdr_sum
                                )
                                global_output_gdr_sum = _distributed_scalar_sum(
                                    local_output_gdr_sum
                                )
                                current_step_log["train_feature_gdr"] = _to_float(
                                    global_feature_gdr_sum / global_gdr_count.clamp_min(1.0)
                                )
                                current_step_log["train_output_gdr"] = _to_float(
                                    global_output_gdr_sum / global_gdr_count.clamp_min(1.0)
                                )
                                current_step_log["train_gdr_count"] = _to_float(global_gdr_count)

                        is_last_batch = batch_idx == (len(train_epoch_dataloader) - 1)
                        reached_max_train_steps = False
                        if optimizer_step_completed:
                            reached_max_train_steps = (
                                cfg.training.max_train_steps is not None
                                and self.global_step + 1
                                >= cfg.training.max_train_steps
                            )
                            stop_after_epoch = reached_max_train_steps
                            step_log = current_step_log
                            step_log_has_optimizer_step = True
                            train_window_log = _new_train_window_log_accumulator()
                        if (
                            optimizer_step_completed
                            and not is_last_batch
                            and not reached_max_train_steps
                        ):
                            if training_contract_log_pending:
                                step_log.update(training_contract_log_fields)
                                training_contract_log_pending = False
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                            step_log = {}
                            step_log_has_optimizer_step = False

                        if reached_max_train_steps:
                            break

                val_epoch_dataloader = (
                    robot_val_dataloader if use_robot_data else open_val_dataloader
                )
                if (
                    cfg.training.val_every is not None
                    and cfg.training.val_every > 0
                    and self.epoch % cfg.training.val_every == 0
                    and val_epoch_dataloader is not None
                    and len(val_epoch_dataloader) > 0
                ):
                    if not step_log:
                        step_log = {
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }
                    self.model.eval()
                    val_metrics = {
                        "action_loss_sum": 0.0,
                        "action_mask_count": 0.0,
                        "heatmap_loss_sum": 0.0,
                        "heatmap_xy_loss_sum": 0.0,
                        "heatmap_point_nll_loss_sum": 0.0,
                        "heatmap_js_loss_sum": 0.0,
                        "heatmap_token_kl_loss_sum": 0.0,
                        "heatmap_mask_count": 0.0,
                        "heatmap_xy_mask_count": 0.0,
                        "robot_action_loss_sum": 0.0,
                        "robot_action_mask_count": 0.0,
                        "robot_heatmap_loss_sum": 0.0,
                        "robot_heatmap_mask_count": 0.0,
                        "robot_heatmap_xy_loss_sum": 0.0,
                        "robot_heatmap_xy_mask_count": 0.0,
                        "robot_heatmap_point_nll_loss_sum": 0.0,
                        "robot_heatmap_js_loss_sum": 0.0,
                        "robot_heatmap_token_kl_loss_sum": 0.0,
                        "open_heatmap_loss_sum": 0.0,
                        "open_heatmap_mask_count": 0.0,
                    }
                    val_preview_summary = None
                    should_save_val_preview = (
                        cfg.training.save_val_heatmap_preview
                        and accelerator.is_main_process
                        and (
                            cfg.training.sample_every is None
                            or cfg.training.sample_every <= 0
                            or self.epoch % cfg.training.sample_every == 0
                        )
                    )
                    open_val_iter = (
                        _RestartingDataLoaderIterator(open_val_dataloader, "open val")
                        if use_robot_data and use_open_val_data
                        else None
                    )
                    with torch.no_grad():
                        with tqdm.tqdm(
                            val_epoch_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, source_batch in enumerate(tepoch):
                                source_batch = dict_apply(
                                    source_batch,
                                    lambda x: x.to(device, non_blocking=True),
                                )
                                robot_batch = source_batch if use_robot_data else None
                                open_batch = None
                                if use_robot_data and use_open_val_data:
                                    open_batch = open_val_iter.next()
                                    open_batch = dict_apply(
                                        open_batch,
                                        lambda x: x.to(device, non_blocking=True),
                                    )
                                elif not use_robot_data:
                                    open_batch = source_batch
                                batch = build_gaze_wam_mixed_batch(
                                    robot_batch=robot_batch,
                                    open_batch=open_batch,
                                    robot_gaze_dropout_prob=cfg.task.robot_gaze_dropout_prob,
                                    robot_heatmap_on_gaze_dropout=cfg.task.get(
                                        "robot_heatmap_on_gaze_dropout",
                                        True,
                                    ),
                                    generator=_make_generator_for_device(
                                        int(cfg.training.val_mixing_seed)
                                        + int(self.epoch) * 100000
                                        + int(batch_idx),
                                        device=device,
                                    ),
                                    shuffle=False,
                                )
                                components = self.model(
                                    batch,
                                    return_per_sample=True,
                                )
                                (
                                    gather_action_loss,
                                    gather_heatmap_loss,
                                    gather_heatmap_xy_loss,
                                    gather_heatmap_point_nll_loss,
                                    gather_heatmap_js_loss,
                                    gather_heatmap_token_kl_loss,
                                    gather_action_mask,
                                    gather_heatmap_mask,
                                    gather_heatmap_xy_mask,
                                    gather_is_open,
                                ) = accelerator.gather_for_metrics(
                                    (
                                        components["per_sample_action_loss"],
                                        components["per_sample_heatmap_loss"],
                                        components["per_sample_heatmap_xy_loss"],
                                        components["per_sample_heatmap_point_nll_loss"],
                                        components["per_sample_heatmap_js_loss"],
                                        components["per_sample_heatmap_token_kl_loss"],
                                        components["action_loss_mask"].to(torch.float32),
                                        components["heatmap_loss_mask"].to(torch.float32),
                                        components["heatmap_xy_loss_mask"].to(torch.float32),
                                        batch["is_open"].to(torch.float32),
                                    )
                                )
                                gather_is_robot = 1.0 - gather_is_open
                                gather_robot_action_mask = gather_action_mask * gather_is_robot
                                gather_robot_heatmap_mask = gather_heatmap_mask * gather_is_robot
                                gather_robot_heatmap_xy_mask = (
                                    gather_heatmap_xy_mask * gather_is_robot
                                )
                                gather_open_heatmap_mask = gather_heatmap_mask * gather_is_open
                                val_metrics["action_loss_sum"] += float(
                                    (gather_action_loss * gather_action_mask).sum().item()
                                )
                                val_metrics["action_mask_count"] += float(
                                    gather_action_mask.sum().item()
                                )
                                val_metrics["heatmap_loss_sum"] += float(
                                    (gather_heatmap_loss * gather_heatmap_mask).sum().item()
                                )
                                val_metrics["heatmap_xy_loss_sum"] += float(
                                    (
                                        gather_heatmap_xy_loss
                                        * gather_heatmap_xy_mask
                                    ).sum().item()
                                )
                                val_metrics["heatmap_point_nll_loss_sum"] += float(
                                    (
                                        gather_heatmap_point_nll_loss
                                        * gather_heatmap_xy_mask
                                    ).sum().item()
                                )
                                val_metrics["heatmap_js_loss_sum"] += float(
                                    (gather_heatmap_js_loss * gather_heatmap_mask).sum().item()
                                )
                                val_metrics["heatmap_token_kl_loss_sum"] += float(
                                    (
                                        gather_heatmap_token_kl_loss
                                        * gather_heatmap_mask
                                    ).sum().item()
                                )
                                val_metrics["heatmap_mask_count"] += float(
                                    gather_heatmap_mask.sum().item()
                                )
                                val_metrics["heatmap_xy_mask_count"] += float(
                                    gather_heatmap_xy_mask.sum().item()
                                )
                                val_metrics["robot_action_loss_sum"] += float(
                                    (gather_action_loss * gather_robot_action_mask).sum().item()
                                )
                                val_metrics["robot_action_mask_count"] += float(
                                    gather_robot_action_mask.sum().item()
                                )
                                val_metrics["robot_heatmap_loss_sum"] += float(
                                    (gather_heatmap_loss * gather_robot_heatmap_mask).sum().item()
                                )
                                val_metrics["robot_heatmap_mask_count"] += float(
                                    gather_robot_heatmap_mask.sum().item()
                                )
                                val_metrics["robot_heatmap_xy_loss_sum"] += float(
                                    (
                                        gather_heatmap_xy_loss
                                        * gather_robot_heatmap_xy_mask
                                    ).sum().item()
                                )
                                val_metrics["robot_heatmap_xy_mask_count"] += float(
                                    gather_robot_heatmap_xy_mask.sum().item()
                                )
                                val_metrics["robot_heatmap_point_nll_loss_sum"] += float(
                                    (
                                        gather_heatmap_point_nll_loss
                                        * gather_robot_heatmap_xy_mask
                                    ).sum().item()
                                )
                                val_metrics["robot_heatmap_js_loss_sum"] += float(
                                    (
                                        gather_heatmap_js_loss
                                        * gather_robot_heatmap_mask
                                    ).sum().item()
                                )
                                val_metrics["robot_heatmap_token_kl_loss_sum"] += float(
                                    (
                                        gather_heatmap_token_kl_loss
                                        * gather_robot_heatmap_mask
                                    ).sum().item()
                                )
                                val_metrics["open_heatmap_loss_sum"] += float(
                                    (gather_heatmap_loss * gather_open_heatmap_mask).sum().item()
                                )
                                val_metrics["open_heatmap_mask_count"] += float(
                                    gather_open_heatmap_mask.sum().item()
                                )
                                if should_save_val_preview and val_preview_summary is None:
                                    val_preview_summary = _write_heatmap_preview(
                                        policy=accelerator.unwrap_model(self.model),
                                        batch=batch,
                                        output_dir=self.output_dir,
                                        epoch=self.epoch,
                                        camera_key=cfg.task.camera_key,
                                        max_samples=cfg.training.get(
                                            "val_heatmap_preview_max_samples",
                                            1,
                                        ),
                                        sample_seed=cfg.training.get(
                                            "val_heatmap_preview_sample_seed",
                                            None,
                                        ),
                                    )

                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1
                                    ):
                                    break
                    val_action_loss = None
                    val_heatmap_loss = None
                    val_heatmap_xy_loss = None
                    val_heatmap_point_nll_loss = None
                    val_heatmap_js_loss = None
                    val_heatmap_token_kl_loss = None
                    if val_metrics["action_mask_count"] > 0:
                        val_action_loss = (
                            val_metrics["action_loss_sum"] / val_metrics["action_mask_count"]
                        )
                        step_log["val_action_loss"] = val_action_loss
                    if val_metrics["heatmap_mask_count"] > 0:
                        val_heatmap_loss = (
                            val_metrics["heatmap_loss_sum"] / val_metrics["heatmap_mask_count"]
                        )
                        step_log["val_heatmap_loss"] = val_heatmap_loss
                        val_heatmap_js_loss = (
                            val_metrics["heatmap_js_loss_sum"]
                            / val_metrics["heatmap_mask_count"]
                        )
                        step_log["val_heatmap_js_loss"] = val_heatmap_js_loss
                        val_heatmap_token_kl_loss = (
                            val_metrics["heatmap_token_kl_loss_sum"]
                            / val_metrics["heatmap_mask_count"]
                        )
                        step_log["val_heatmap_token_kl_loss"] = val_heatmap_token_kl_loss
                    if val_metrics["heatmap_xy_mask_count"] > 0:
                        val_heatmap_xy_loss = (
                            val_metrics["heatmap_xy_loss_sum"]
                            / val_metrics["heatmap_xy_mask_count"]
                        )
                        step_log["val_heatmap_xy_loss"] = val_heatmap_xy_loss
                        val_heatmap_point_nll_loss = (
                            val_metrics["heatmap_point_nll_loss_sum"]
                            / val_metrics["heatmap_xy_mask_count"]
                        )
                        step_log["val_heatmap_point_nll_loss"] = (
                            val_heatmap_point_nll_loss
                        )
                    if (
                        val_action_loss is not None
                        or val_heatmap_loss is not None
                        or val_heatmap_xy_loss is not None
                        or val_heatmap_point_nll_loss is not None
                        or val_heatmap_js_loss is not None
                        or val_heatmap_token_kl_loss is not None
                    ):
                        loss = 0.0
                        policy_module = accelerator.unwrap_model(self.model)
                        if val_action_loss is not None:
                            loss += float(policy_module.action_loss_weight) * val_action_loss
                        if val_heatmap_loss is not None:
                            loss += float(policy_module.heatmap_loss_weight) * val_heatmap_loss
                        if val_heatmap_token_kl_loss is not None:
                            loss += (
                                float(policy_module.heatmap_token_kl_loss_weight)
                                * val_heatmap_token_kl_loss
                            )
                        if getattr(
                            policy_module,
                            "heatmap_diffusion_final_loss_enabled",
                            False,
                        ):
                            if val_heatmap_xy_loss is not None:
                                loss += (
                                    float(policy_module.heatmap_xy_loss_weight)
                                    * val_heatmap_xy_loss
                                )
                            if val_heatmap_point_nll_loss is not None:
                                loss += (
                                    float(policy_module.heatmap_point_nll_loss_weight)
                                    * val_heatmap_point_nll_loss
                                )
                            if val_heatmap_js_loss is not None:
                                loss += (
                                    float(policy_module.heatmap_js_loss_weight)
                                    * val_heatmap_js_loss
                                )
                        step_log["val_loss"] = loss
                    if val_metrics["robot_action_mask_count"] > 0:
                        step_log["val_robot_action_loss"] = (
                            val_metrics["robot_action_loss_sum"]
                            / val_metrics["robot_action_mask_count"]
                        )
                    if val_metrics["robot_heatmap_mask_count"] > 0:
                        step_log["val_robot_heatmap_loss"] = (
                            val_metrics["robot_heatmap_loss_sum"]
                            / val_metrics["robot_heatmap_mask_count"]
                        )
                    if (
                        val_metrics["robot_action_mask_count"] > 0
                        or val_metrics["robot_heatmap_mask_count"] > 0
                    ):
                        policy_module = accelerator.unwrap_model(self.model)

                        def _metric_mean(sum_key, count_key):
                            count = val_metrics[count_key]
                            if count <= 0:
                                return None
                            return val_metrics[sum_key] / count

                        robot_action_loss = _metric_mean(
                            "robot_action_loss_sum",
                            "robot_action_mask_count",
                        )
                        robot_heatmap_loss = _metric_mean(
                            "robot_heatmap_loss_sum",
                            "robot_heatmap_mask_count",
                        )
                        robot_heatmap_xy_loss = _metric_mean(
                            "robot_heatmap_xy_loss_sum",
                            "robot_heatmap_xy_mask_count",
                        )
                        robot_heatmap_point_nll_loss = _metric_mean(
                            "robot_heatmap_point_nll_loss_sum",
                            "robot_heatmap_xy_mask_count",
                        )
                        robot_heatmap_js_loss = _metric_mean(
                            "robot_heatmap_js_loss_sum",
                            "robot_heatmap_mask_count",
                        )
                        robot_heatmap_token_kl_loss = _metric_mean(
                            "robot_heatmap_token_kl_loss_sum",
                            "robot_heatmap_mask_count",
                        )
                        robot_loss = 0.0
                        if robot_action_loss is not None:
                            robot_loss += (
                                float(policy_module.action_loss_weight)
                                * robot_action_loss
                            )
                        if robot_heatmap_loss is not None:
                            robot_loss += (
                                float(policy_module.heatmap_loss_weight)
                                * robot_heatmap_loss
                            )
                        if robot_heatmap_token_kl_loss is not None:
                            robot_loss += (
                                float(policy_module.heatmap_token_kl_loss_weight)
                                * robot_heatmap_token_kl_loss
                            )
                        if getattr(
                            policy_module,
                            "heatmap_diffusion_final_loss_enabled",
                            False,
                        ):
                            if robot_heatmap_xy_loss is not None:
                                robot_loss += (
                                    float(policy_module.heatmap_xy_loss_weight)
                                    * robot_heatmap_xy_loss
                                )
                            if robot_heatmap_point_nll_loss is not None:
                                robot_loss += (
                                    float(policy_module.heatmap_point_nll_loss_weight)
                                    * robot_heatmap_point_nll_loss
                                )
                            if robot_heatmap_js_loss is not None:
                                robot_loss += (
                                    float(policy_module.heatmap_js_loss_weight)
                                    * robot_heatmap_js_loss
                                )
                        step_log["val_robot_loss"] = robot_loss
                        step_log["val_robot_heatmap_xy_loss"] = (
                            robot_heatmap_xy_loss
                            if robot_heatmap_xy_loss is not None
                            else 0.0
                        )
                        step_log["val_robot_heatmap_js_loss"] = (
                            robot_heatmap_js_loss
                            if robot_heatmap_js_loss is not None
                            else 0.0
                        )
                        step_log["val_robot_heatmap_token_kl_loss"] = (
                            robot_heatmap_token_kl_loss
                            if robot_heatmap_token_kl_loss is not None
                            else 0.0
                        )
                    if val_metrics["open_heatmap_mask_count"] > 0:
                        step_log["val_open_heatmap_loss"] = (
                            val_metrics["open_heatmap_loss_sum"]
                            / val_metrics["open_heatmap_mask_count"]
                        )
                    step_log["val_action_mask_count"] = val_metrics["action_mask_count"]
                    step_log["val_heatmap_mask_count"] = val_metrics["heatmap_mask_count"]
                    step_log["val_heatmap_xy_mask_count"] = val_metrics[
                        "heatmap_xy_mask_count"
                    ]
                    step_log["val_robot_action_mask_count"] = val_metrics[
                        "robot_action_mask_count"
                    ]
                    step_log["val_robot_heatmap_mask_count"] = val_metrics[
                        "robot_heatmap_mask_count"
                    ]
                    step_log["val_open_heatmap_mask_count"] = val_metrics[
                        "open_heatmap_mask_count"
                    ]
                    if val_preview_summary is not None:
                        step_log["val_heatmap_preview_saved"] = 1
                        step_log["val_heatmap_preview_num_samples"] = val_preview_summary[
                            "num_samples"
                        ]
                    self.model.train()

                should_save_checkpoint = (
                    _gaze_wam_checkpoint_due(
                        epoch=self.epoch,
                        checkpoint_every=cfg.training.checkpoint_every,
                        stop_after_epoch=stop_after_epoch,
                    )
                    and accelerator.is_main_process
                )
                metric_dict = {}
                if step_log:
                    if training_contract_log_pending:
                        step_log.update(training_contract_log_fields)
                        training_contract_log_pending = False
                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    if step_log_has_optimizer_step:
                        self.global_step += 1
                self.epoch += 1

                if should_save_checkpoint:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(
                            retain_last_n=int(
                                cfg.checkpoint.get("keep_last_n", 0)
                            ),
                            retained_tag=(
                                f"rolling-epoch={int(self.epoch):04d}"
                                f"-step={int(self.global_step):06d}"
                            ),
                        )
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    topk_ckpt_path = None
                    monitor_key = cfg.checkpoint.topk.monitor_key
                    if monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    self.model = model_ddp

                if stop_after_epoch:
                    break

        if accelerator.is_main_process:
            self.wait_for_pending_checkpoint()

        transfer_export_path = str(
            training_config["transfer"]["export_path"] or ""
        )
        if transfer_export_path:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                transfer_export_summary = export_gaze_wam_transfer_artifact(
                    accelerator.unwrap_model(self.model),
                    hydra.utils.to_absolute_path(transfer_export_path),
                    scope=training_config["transfer"]["export_scope"],
                    overwrite=bool(
                        training_config["transfer"]["export_overwrite"]
                    ),
                    metadata={
                        "training_stage": training_config["stage"],
                        "global_optimizer_steps": int(self.global_step),
                        "robot_ratio": float(
                            training_config["batching"]["robot_ratio"]
                        ),
                        "open_ratio": float(
                            training_config["batching"]["open_ratio"]
                        ),
                    },
                )
                training_contract["transfer"]["export"] = transfer_export_summary
                _write_training_contract_summary(training_contract, self.output_dir)
                print(
                    "Exported Gaze-WAM transfer artifact: "
                    f"{transfer_export_summary['path']} "
                    f"scope={transfer_export_summary['scope']}"
                )

        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainGazeWamWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
