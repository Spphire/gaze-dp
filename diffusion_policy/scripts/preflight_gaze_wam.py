from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import Dict, Optional, Sequence

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _shape(value) -> object:
    _ensure_preflight_runtime()
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.shape]
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    return None


def _to_float(value) -> float:
    _ensure_preflight_runtime()
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().item())
    return float(value)


def _make_heatmap_only_identity_normalizer(camera_key: str):
    _ensure_preflight_runtime()
    normalizer = LinearNormalizer()
    normalizer[str(camera_key)] = get_image_identity_normalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def _maybe_list(value):
    _ensure_preflight_runtime()
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _maybe_list(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return [_maybe_list(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _maybe_list(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, (list, tuple, dict, int, float, bool)):
            return _maybe_list(parsed)
        return value
    if isinstance(value, (int, float, bool)):
        return value
    try:
        values = list(value)
    except TypeError:
        pass
    else:
        return [_maybe_list(item) for item in values]
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if parsed is value:
        return text
    return _maybe_list(parsed)


def _ensure_preflight_runtime():
    global DataLoader
    global OmegaConf
    global as_optional_gaze_wam_key
    global build_gaze_wam_mixed_batch
    global dict_apply
    global hydra
    global load_cfg
    global load_policy_for_eval
    global loss_routing_summary
    global torch
    global _check_training_dataloader_lengths
    global _check_training_dataset_lengths
    global _normalize_gaze_wam_early_bool_config
    global _normalize_gaze_wam_training_config
    global _normalize_gaze_wam_task_routing_config
    global gaze_wam_action_normalizer_contract
    global gaze_wam_data_stream_contract
    global gaze_wam_loss_routing_validation_guardrails_ok
    global get_image_identity_normalizer
    global LinearNormalizer
    global normalize_gaze_wam_bool_field
    global normalize_gaze_wam_nonnegative_float_field
    global normalize_gaze_wam_nonnegative_int_field
    global normalize_gaze_wam_positive_float_field
    global normalize_gaze_wam_positive_int_sequence
    global normalize_gaze_wam_positive_int_field
    global SingleFieldLinearNormalizer
    global validate_gaze_wam_training_config
    global validate_gaze_wam_task_routing_config
    global validate_gaze_wam_zarr
    try:
        return torch
    except NameError:
        import hydra as _hydra
        import torch as _torch
        from omegaconf import OmegaConf as _OmegaConf
        from torch.utils.data import DataLoader as _DataLoader
        from diffusion_policy.common.gaze_utils import (
            as_optional_gaze_wam_key as _as_optional_gaze_wam_key,
        )
        from diffusion_policy.common.normalize_util import (
            get_image_identity_normalizer as _get_image_identity_normalizer,
        )
        from diffusion_policy.common.pytorch_util import dict_apply as _dict_apply
        from diffusion_policy.dataset.gaze_wam_mixing import (
            build_gaze_wam_mixed_batch as _build_gaze_wam_mixed_batch,
        )
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer as _LinearNormalizer,
            SingleFieldLinearNormalizer as _SingleFieldLinearNormalizer,
        )
        from diffusion_policy.model.gaze_wam.routing import (
            loss_routing_summary as _loss_routing_summary,
        )
        from diffusion_policy.scripts.eval_gaze_wam_metrics import (
            load_cfg as _load_cfg,
            load_policy_for_eval as _load_policy_for_eval,
        )
        from diffusion_policy.scripts.validate_gaze_wam_zarr import (
            validate_gaze_wam_zarr as _validate_gaze_wam_zarr,
        )
        from diffusion_policy.common.gaze_wam_dataloader_checks import (
            _check_training_dataloader_lengths as _common_check_training_dataloader_lengths,
            _check_training_dataset_lengths as _common_check_training_dataset_lengths,
        )
        from diffusion_policy.common.gaze_wam_training_config import (
            _normalize_gaze_wam_early_bool_config as _common_normalize_gaze_wam_early_bool_config,
            _normalize_gaze_wam_training_config as _common_normalize_gaze_wam_training_config,
            _normalize_gaze_wam_task_routing_config as _common_normalize_gaze_wam_task_routing_config,
            gaze_wam_action_normalizer_contract as _common_gaze_wam_action_normalizer_contract,
            gaze_wam_data_stream_contract as _common_gaze_wam_data_stream_contract,
            gaze_wam_loss_routing_validation_guardrails_ok as _common_gaze_wam_loss_routing_validation_guardrails_ok,
            normalize_gaze_wam_bool_field as _common_normalize_gaze_wam_bool_field,
            normalize_gaze_wam_nonnegative_float_field as _common_normalize_gaze_wam_nonnegative_float_field,
            normalize_gaze_wam_nonnegative_int_field as _common_normalize_gaze_wam_nonnegative_int_field,
            normalize_gaze_wam_positive_float_field as _common_normalize_gaze_wam_positive_float_field,
            normalize_gaze_wam_positive_int_sequence as _common_normalize_gaze_wam_positive_int_sequence,
            normalize_gaze_wam_positive_int_field as _common_normalize_gaze_wam_positive_int_field,
            validate_gaze_wam_training_config as _common_validate_gaze_wam_training_config,
            validate_gaze_wam_task_routing_config as _common_validate_gaze_wam_task_routing_config,
        )

        DataLoader = _DataLoader
        OmegaConf = _OmegaConf
        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        build_gaze_wam_mixed_batch = _build_gaze_wam_mixed_batch
        dict_apply = _dict_apply
        hydra = _hydra
        load_cfg = _load_cfg
        load_policy_for_eval = _load_policy_for_eval
        loss_routing_summary = _loss_routing_summary
        torch = _torch
        _check_training_dataloader_lengths = _common_check_training_dataloader_lengths
        _check_training_dataset_lengths = _common_check_training_dataset_lengths
        _normalize_gaze_wam_early_bool_config = _common_normalize_gaze_wam_early_bool_config
        _normalize_gaze_wam_training_config = _common_normalize_gaze_wam_training_config
        _normalize_gaze_wam_task_routing_config = _common_normalize_gaze_wam_task_routing_config
        gaze_wam_action_normalizer_contract = _common_gaze_wam_action_normalizer_contract
        gaze_wam_data_stream_contract = _common_gaze_wam_data_stream_contract
        gaze_wam_loss_routing_validation_guardrails_ok = (
            _common_gaze_wam_loss_routing_validation_guardrails_ok
        )
        get_image_identity_normalizer = _get_image_identity_normalizer
        LinearNormalizer = _LinearNormalizer
        normalize_gaze_wam_bool_field = _common_normalize_gaze_wam_bool_field
        normalize_gaze_wam_nonnegative_float_field = _common_normalize_gaze_wam_nonnegative_float_field
        normalize_gaze_wam_nonnegative_int_field = _common_normalize_gaze_wam_nonnegative_int_field
        normalize_gaze_wam_positive_float_field = _common_normalize_gaze_wam_positive_float_field
        normalize_gaze_wam_positive_int_sequence = _common_normalize_gaze_wam_positive_int_sequence
        normalize_gaze_wam_positive_int_field = _common_normalize_gaze_wam_positive_int_field
        SingleFieldLinearNormalizer = _SingleFieldLinearNormalizer
        validate_gaze_wam_training_config = _common_validate_gaze_wam_training_config
        validate_gaze_wam_task_routing_config = _common_validate_gaze_wam_task_routing_config
        validate_gaze_wam_zarr = _validate_gaze_wam_zarr
        return torch


def _obs_transform_summary(obs_encoder) -> Dict[str, object]:
    summaries = {}
    transform_map = getattr(obs_encoder, "key_transform_map", {})
    for key, transform in transform_map.items():
        modules = []
        children = list(transform.children()) if hasattr(transform, "children") else []
        if not children:
            children = [transform]
        for module in children:
            info = {"type": type(module).__name__}
            if hasattr(module, "mean") and hasattr(module, "std"):
                info["mean"] = _maybe_list(getattr(module, "mean"))
                info["std"] = _maybe_list(getattr(module, "std"))
            modules.append(info)
        summaries[str(key)] = modules
    return summaries


def _has_normalize_transform(summary: Dict[str, object], mean, std) -> bool:
    if mean is None or std is None:
        return True
    mean = [float(v) for v in mean]
    std = [float(v) for v in std]
    for modules in summary.get("obs_encoder_transforms", {}).values():
        for module in modules:
            if module.get("type") != "Normalize":
                continue
            module_mean = module.get("mean")
            module_std = module.get("std")
            if module_mean is None or module_std is None:
                continue
            module_mean = [float(v) for v in module_mean]
            module_std = [float(v) for v in module_std]
            if len(module_mean) != len(mean) or len(module_std) != len(std):
                continue
            if all(abs(a - b) < 1e-9 for a, b in zip(module_mean, mean)) and all(
                abs(a - b) < 1e-9 for a, b in zip(module_std, std)
            ):
                return True
    return False


def _sample_summary(sample: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "obs": _shape(sample["obs"]),
        "action": _shape(sample["action"]),
        "heatmap": _shape(sample["heatmap"]),
        "gaze_xy": _shape(sample["gaze_xy"]),
        "has_action": bool(sample["has_action"].item()),
        "has_heatmap": bool(sample["has_heatmap"].item()),
        "has_gaze_label": bool(sample["has_gaze_label"].item()),
        "use_gaze_condition": bool(sample["use_gaze_condition"].item()),
        "is_gaze_condition_dropped": bool(sample["is_gaze_condition_dropped"].item()),
    }
    for key in ("action_abs", "action_base_abs", "heatmap_image"):
        if key in sample:
            summary[key] = _shape(sample[key])
    for key in ("has_action_abs", "has_action_base_abs", "has_heatmap_image"):
        if key in sample:
            summary[key] = bool(sample[key].item())
    return summary


def _dataset_sampling_summary(dataset_cfg) -> Dict[str, object]:
    _ensure_preflight_runtime()
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
        "action_padding": normalize_gaze_wam_bool_field(
            "dataset.action_padding",
            dataset_cfg.get("action_padding", True),
            default=True,
        ),
    }


def _sampling_contract_summary(cfg) -> Dict[str, object]:
    _ensure_preflight_runtime()
    task_sampling = {
        "n_obs_steps": normalize_gaze_wam_positive_int_field(
            "task.n_obs_steps",
            cfg.task.n_obs_steps,
        ),
        "action_horizon": normalize_gaze_wam_positive_int_field(
            "task.action_horizon",
            cfg.task.action_horizon,
        ),
        "n_latency_steps": normalize_gaze_wam_nonnegative_int_field(
            "task.n_latency_steps",
            cfg.task.get("n_latency_steps", 0),
        ),
    }
    robot_sampling = _dataset_sampling_summary(cfg.task.robot_dataset)
    open_sampling = _dataset_sampling_summary(cfg.task.open_dataset)
    compare_keys = ["n_obs_steps", "action_horizon", "n_latency_steps"]
    return {
        "task": task_sampling,
        "robot_dataset": robot_sampling,
        "open_dataset": open_sampling,
        "compare_keys": compare_keys,
        "robot_matches_task": all(
            robot_sampling[key] == task_sampling[key] for key in compare_keys
        ),
        "open_matches_task": all(
            open_sampling[key] == task_sampling[key] for key in compare_keys
        ),
    }


def _check_sampling_contract(summary: Dict[str, object]) -> Sequence[str]:
    errors = []
    if not summary["robot_matches_task"]:
        errors.append(
            "Robot dataset temporal sampling must match task n_obs_steps, action_horizon, "
            f"and n_latency_steps; got task={summary['task']!r}, "
            f"robot_dataset={summary['robot_dataset']!r}."
        )
    if not summary["open_matches_task"]:
        errors.append(
            "Open dataset temporal sampling must match task n_obs_steps, action_horizon, "
            f"and n_latency_steps; got task={summary['task']!r}, "
            f"open_dataset={summary['open_dataset']!r}."
        )
    return errors


def _image_geometry_summary(cfg) -> Dict[str, object]:
    _ensure_preflight_runtime()
    task_resize_mode = str(cfg.task.get("image_resize_mode", "stretch"))
    robot_resize_mode = str(cfg.task.robot_dataset.get("image_resize_mode", task_resize_mode))
    open_resize_mode = str(cfg.task.open_dataset.get("image_resize_mode", task_resize_mode))
    image_shape = normalize_gaze_wam_positive_int_sequence(
        "task.image_shape",
        cfg.task.image_shape,
        length=3,
    )
    task_image_size = image_shape[-2:]
    robot_image_size = normalize_gaze_wam_positive_int_sequence(
        "task.robot_dataset.image_size",
        cfg.task.robot_dataset.image_size,
        length=2,
    )
    open_image_size = normalize_gaze_wam_positive_int_sequence(
        "task.open_dataset.image_size",
        cfg.task.open_dataset.image_size,
        length=2,
    )
    resize_modes = {
        "task": task_resize_mode,
        "robot_dataset": robot_resize_mode,
        "open_dataset": open_resize_mode,
    }
    image_sizes = {
        "task": task_image_size,
        "robot_dataset": robot_image_size,
        "open_dataset": open_image_size,
    }
    return {
        "image_shape": image_shape,
        "task_image_size": task_image_size,
        "robot_image_size": robot_image_size,
        "open_image_size": open_image_size,
        "task_image_resize_mode": task_resize_mode,
        "robot_image_resize_mode": robot_resize_mode,
        "open_image_resize_mode": open_resize_mode,
        "resize_modes": resize_modes,
        "image_sizes": image_sizes,
        "all_stretch": all(mode == "stretch" for mode in resize_modes.values()),
        "consistent": len(set(resize_modes.values())) == 1,
        "image_size_consistent": (
            robot_image_size == task_image_size and open_image_size == task_image_size
        ),
    }


def _check_image_geometry_contract(geometry: Dict[str, object]) -> Sequence[str]:
    errors = []
    resize_modes = geometry["resize_modes"]
    if not geometry["all_stretch"]:
        errors.append(
            "Gaze-WAM image_resize_mode must be 'stretch' for task, robot dataset, "
            f"and open dataset; got {resize_modes!r}."
        )
    if not geometry["consistent"]:
        errors.append(
            "Gaze-WAM task, robot dataset, and open dataset must use the same "
            f"image_resize_mode; got {resize_modes!r}."
        )
    if not geometry["image_size_consistent"]:
        errors.append(
            "Gaze-WAM task.image_shape H/W, robot dataset image_size, and open dataset image_size "
            f"must match; got {geometry['image_sizes']!r}."
        )
    return errors


def _data_stream_contract_summary(cfg, training_config: Dict[str, object]) -> Dict[str, object]:
    _ensure_preflight_runtime()
    return gaze_wam_data_stream_contract(
        robot_dataset_path=str(cfg.task.robot_dataset.dataset_path),
        open_dataset_path=str(cfg.task.open_dataset.dataset_path),
        robot_dataset_class=str(cfg.task.robot_dataset.get("_target_", "")),
        open_dataset_class=str(cfg.task.open_dataset.get("_target_", "")),
        robot_batch_size=int(training_config["robot_batch_size"]),
        open_batch_size=int(training_config["open_batch_size"]),
    )


def _check_data_stream_contract(contract: Dict[str, object]) -> Sequence[str]:
    errors = []
    robot = contract.get("robot") or {}
    open_source = contract.get("open") or {}
    mixing = contract.get("mixing") or {}
    if contract.get("source") != "two_zarr_two_dataset_online_mixed_batch":
        errors.append("Gaze-WAM data stream must use the two-zarr online mixed-batch contract.")
    if contract.get("separate_zarr_sources") is not True:
        errors.append("Gaze-WAM robot and open-source datasets must point to separate zarr roots.")
    if contract.get("offline_merged_zarr") is not False:
        errors.append("Gaze-WAM must not pre-merge robot and open-source rows into one zarr.")
    if robot.get("dataset_class_matches_expected") is not True:
        errors.append(
            "Robot data stream must instantiate GazeWamRobotDataset, got "
            f"{robot.get('dataset_class')!r}."
        )
    if open_source.get("dataset_class_matches_expected") is not True:
        errors.append(
            "Open-source data stream must instantiate GazeWamOpenDataset, got "
            f"{open_source.get('dataset_class')!r}."
        )
    if mixing.get("builder") != (
        "diffusion_policy.dataset.gaze_wam_mixing.build_gaze_wam_mixed_batch"
    ):
        errors.append("Gaze-WAM data stream must use build_gaze_wam_mixed_batch.")
    if mixing.get("mode") != "online_per_step_concat_after_fetch":
        errors.append("Gaze-WAM robot/open rows must be mixed online after per-step fetch.")
    if mixing.get("ratio_source") != "robot_dataloader.batch_size/open_dataloader.batch_size":
        errors.append("Gaze-WAM source ratio must be defined by the two dataloader batch sizes.")
    return errors


def _path_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _summary_bool(name: str, value, default: bool = False) -> bool:
    _ensure_preflight_runtime()
    return normalize_gaze_wam_bool_field(name, value, default=default)


def _policy_contract_summary(policy, cfg) -> Dict[str, object]:
    _ensure_preflight_runtime()
    obs_shape = tuple(int(v) for v in policy.obs_encoder.output_shape())
    n_obs_steps = normalize_gaze_wam_positive_int_field(
        "task.n_obs_steps",
        cfg.task.n_obs_steps,
    )
    heatmap_num_tokens = normalize_gaze_wam_positive_int_field(
        "task.heatmap_num_tokens",
        cfg.task.heatmap_num_tokens,
    )
    heatmap_dim = normalize_gaze_wam_positive_int_field(
        "task.heatmap_dim",
        cfg.task.heatmap_dim,
    )
    heatmap_token_grid = normalize_gaze_wam_positive_int_sequence(
        "task.heatmap_token_grid",
        cfg.task.heatmap_token_grid,
        length=2,
    )
    image_shape = normalize_gaze_wam_positive_int_sequence(
        "task.image_shape",
        cfg.task.image_shape,
        length=3,
    )
    image_hw = image_shape[-2:]
    patch_size = None
    patch_area = 0
    if len(heatmap_token_grid) == 2 and all(v > 0 for v in heatmap_token_grid):
        if image_hw[0] % heatmap_token_grid[0] == 0 and image_hw[1] % heatmap_token_grid[1] == 0:
            patch_h = image_hw[0] // heatmap_token_grid[0]
            patch_w = image_hw[1] // heatmap_token_grid[1]
            patch_area = int(patch_h * patch_w)
            if patch_h == patch_w:
                patch_size = int(patch_h)
            else:
                patch_size = [int(patch_h), int(patch_w)]
    latent_grid = [int(heatmap_token_grid[0]), int(heatmap_token_grid[1])]
    visual_tokens_per_frame = int(obs_shape[-2]) / n_obs_steps if n_obs_steps > 0 else 0.0
    robot_batch_size = int(cfg.robot_dataloader.batch_size)
    open_batch_size = int(cfg.open_dataloader.batch_size)
    total_batch_size = robot_batch_size + open_batch_size
    if total_batch_size > 0:
        robot_ratio = robot_batch_size / total_batch_size
        open_ratio = open_batch_size / total_batch_size
    else:
        robot_ratio = 0.0
        open_ratio = 0.0
    attention_contract = None
    if hasattr(policy.model, "attention_contract_summary"):
        attention_contract = policy.model.attention_contract_summary(
            num_image_tokens=int(policy.model.max_image_tokens)
        )
    loss_routing_contract = None
    if hasattr(policy, "loss_routing_contract_summary"):
        loss_routing_contract = policy.loss_routing_contract_summary()
    obs_encoder_checkpoint_path = _path_str(getattr(policy.obs_encoder, "checkpoint_path", ""))
    obs_encoder_cache_dir = _path_str(getattr(policy.obs_encoder, "cache_dir", ""))
    obs_encoder_checkpoint_path_exists = _configured_path_exists(obs_encoder_checkpoint_path)
    obs_encoder_cache_dir_exists = _configured_path_exists(obs_encoder_cache_dir)
    obs_encoder_checkpoint_path_is_file = _configured_path_is_file(obs_encoder_checkpoint_path)
    obs_encoder_cache_dir_is_dir = _configured_path_is_dir(obs_encoder_cache_dir)
    obs_encoder_local_weight_source_configured = bool(
        obs_encoder_checkpoint_path or obs_encoder_cache_dir
    )
    obs_encoder_local_weight_source_exists = (
        (not obs_encoder_checkpoint_path or obs_encoder_checkpoint_path_exists)
        and (not obs_encoder_cache_dir or obs_encoder_cache_dir_exists)
    )
    obs_encoder_local_weight_source_valid = (
        (not obs_encoder_checkpoint_path or obs_encoder_checkpoint_path_is_file)
        and (not obs_encoder_cache_dir or obs_encoder_cache_dir_is_dir)
    )
    obs_encoder_pretrained_cfg = _maybe_list(
        getattr(policy.obs_encoder, "timm_pretrained_cfg", None)
    )
    obs_encoder_hf_hub_id = None
    if isinstance(obs_encoder_pretrained_cfg, dict):
        obs_encoder_hf_hub_id = obs_encoder_pretrained_cfg.get("hf_hub_id")
    heatmap_latent_scale = float(getattr(policy, "heatmap_latent_scale", 1.0))
    heatmap_latent_offset = float(getattr(policy, "heatmap_latent_offset", 0.0))
    heatmap_latent_stats_path = str(getattr(policy, "heatmap_latent_stats_path", ""))
    heatmap_latent_stats = _latent_stats_summary(heatmap_latent_stats_path)
    heatmap_latent_scaled_range = _scaled_latent_range_from_stats(
        heatmap_latent_stats,
        heatmap_latent_scale,
        heatmap_latent_offset,
    )

    return {
        "obs_encoder_model_name": getattr(policy.obs_encoder, "model_name", None),
        "obs_encoder_pretrained": bool(getattr(policy.obs_encoder, "pretrained", False)),
        "obs_encoder_pretrained_cfg_name": _maybe_list(
            getattr(policy.obs_encoder, "pretrained_cfg_name", None)
        ),
        "obs_encoder_pretrained_cfg": obs_encoder_pretrained_cfg,
        "obs_encoder_hf_hub_id": obs_encoder_hf_hub_id,
        "obs_encoder_checkpoint_path": obs_encoder_checkpoint_path,
        "obs_encoder_checkpoint_path_exists": obs_encoder_checkpoint_path_exists,
        "obs_encoder_checkpoint_path_is_file": obs_encoder_checkpoint_path_is_file,
        "obs_encoder_cache_dir": obs_encoder_cache_dir,
        "obs_encoder_cache_dir_exists": obs_encoder_cache_dir_exists,
        "obs_encoder_cache_dir_is_dir": obs_encoder_cache_dir_is_dir,
        "obs_encoder_local_weight_source_configured": obs_encoder_local_weight_source_configured,
        "obs_encoder_local_weight_source_exists": obs_encoder_local_weight_source_exists,
        "obs_encoder_local_weight_source_valid": obs_encoder_local_weight_source_valid,
        "obs_encoder_feature_aggregation": getattr(
            policy.obs_encoder,
            "feature_aggregation",
            None,
        ),
        "obs_encoder_downsample_ratio": getattr(policy.obs_encoder, "downsample_ratio", None),
        "obs_encoder_patch_size": _maybe_list(getattr(policy.obs_encoder, "patch_size", None)),
        "obs_encoder_model_input_size": _maybe_list(
            getattr(policy.obs_encoder, "model_input_size", None)
        ),
        "obs_encoder_transforms": _obs_transform_summary(policy.obs_encoder),
        "obs_output_shape": list(obs_shape),
        "visual_tokens_per_batch_item": int(obs_shape[-2]),
        "visual_tokens_per_frame": float(visual_tokens_per_frame),
        "expected_visual_tokens_per_frame": heatmap_num_tokens,
        "expected_visual_tokens_total": n_obs_steps * heatmap_num_tokens,
        "visual_embed_dim": int(obs_shape[-1]),
        "image_size": image_hw,
        "image_resize_mode": str(cfg.task.get("image_resize_mode", "stretch")),
        "inferred_patch_size_from_heatmap_grid": patch_size,
        "gaze_mask_token_shape": _shape(policy.gaze_encoder.mask_token),
        "gaze_encoder_grid": list(policy.gaze_encoder.spatial_encoder.grid_size),
        "gaze_encoder_sigma": float(policy.gaze_encoder.spatial_encoder.sigma),
        "model_n_emb": int(policy.model.n_emb),
        "model_max_image_tokens": int(policy.model.max_image_tokens),
        "model_action_dim": int(policy.model.action_dim),
        "model_action_horizon": int(policy.model.action_horizon),
        "model_heatmap_num_tokens": int(policy.model.heatmap_num_tokens),
        "model_heatmap_dim": int(policy.model.heatmap_dim),
        "heatmap_representation": "channel_latent_decoder_to_full_resolution",
        "heatmap_spatial_decoder": str(getattr(policy, "heatmap_spatial_decoder", "unknown")),
        "heatmap_latent_grid": latent_grid,
        "heatmap_latent_scale": heatmap_latent_scale,
        "heatmap_latent_offset": heatmap_latent_offset,
        "heatmap_latent_stats_path": heatmap_latent_stats_path,
        "heatmap_latent_stats": heatmap_latent_stats,
        "heatmap_latent_scaled_range": heatmap_latent_scaled_range,
        "heatmap_scheduler_clip_sample": getattr(
            policy,
            "heatmap_scheduler_clip_sample",
            None,
        ),
        "heatmap_patch_size": _maybe_list(patch_size),
        "heatmap_patch_area": int(patch_area),
        "action_head_out_features": int(policy.model.action_head.out_features),
        "heatmap_head_out_features": int(policy.model.heatmap_head.out_features),
        "use_block_attention_mask": bool(policy.model.use_block_attention_mask),
        "attention_contract": attention_contract,
        "loss_routing_contract": loss_routing_contract,
        "normalizer_contract": gaze_wam_action_normalizer_contract(
            action_dim=int(policy.model.action_dim),
            camera_key=str(
                cfg.task.robot_dataset.get(
                    "camera_key",
                    cfg.task.get("camera_key", "camera0_rgb"),
                )
            ),
            heatmap_only=robot_batch_size <= 0,
        ),
        "heatmap_codec_token_grid": list(policy.heatmap_codec.token_grid),
        "heatmap_codec_image_size": list(policy.heatmap_codec.image_size),
        "heatmap_decode_method": str(policy.heatmap_decode_method),
        "heatmap_objective": str(policy.heatmap_objective),
        "heatmap_token_kl_loss_weight": float(policy.heatmap_token_kl_loss_weight),
        "heatmap_xy_loss_weight": float(policy.heatmap_xy_loss_weight),
        "heatmap_point_nll_loss_weight": float(policy.heatmap_point_nll_loss_weight),
        "heatmap_js_loss_weight": float(policy.heatmap_js_loss_weight),
        "heatmap_diffusion_final_loss_enabled": bool(
            getattr(policy, "heatmap_diffusion_final_loss_enabled", False)
        ),
        "heatmap_final_loss_timestep_weighting": str(
            getattr(policy, "heatmap_final_loss_timestep_weighting", "none")
        ),
        "heatmap_dsnt_temperature": float(policy.heatmap_dsnt_temperature),
        "heatmap_distribution_mode": str(policy.heatmap_distribution_mode),
        "num_inference_steps": int(policy.num_inference_steps),
        "robot_batch_size": robot_batch_size,
        "open_batch_size": open_batch_size,
        "robot_ratio": float(robot_ratio),
        "open_ratio": float(open_ratio),
    }


def _check_policy_contract(summary: Dict[str, object], cfg) -> Sequence[str]:
    _ensure_preflight_runtime()
    errors = []
    task_n_obs_steps = normalize_gaze_wam_positive_int_field(
        "task.n_obs_steps",
        cfg.task.n_obs_steps,
    )
    task_action_horizon = normalize_gaze_wam_positive_int_field(
        "task.action_horizon",
        cfg.task.action_horizon,
    )
    task_action_dim = normalize_gaze_wam_positive_int_field(
        "task.action_dim",
        cfg.task.action_dim,
    )
    task_heatmap_num_tokens = normalize_gaze_wam_positive_int_field(
        "task.heatmap_num_tokens",
        cfg.task.heatmap_num_tokens,
    )
    task_heatmap_dim = normalize_gaze_wam_positive_int_field(
        "task.heatmap_dim",
        cfg.task.heatmap_dim,
    )
    expected_visual_tokens = task_n_obs_steps * task_heatmap_num_tokens
    expected_embed_dim = int(cfg.policy.n_emb)
    expected_grid = normalize_gaze_wam_positive_int_sequence(
        "task.heatmap_token_grid",
        cfg.task.heatmap_token_grid,
        length=2,
    )
    expected_image_size = normalize_gaze_wam_positive_int_sequence(
        "task.image_shape",
        cfg.task.image_shape,
        length=3,
    )[-2:]
    expected_patch_area = 0
    if expected_image_size[0] % expected_grid[0] == 0 and expected_image_size[1] % expected_grid[1] == 0:
        expected_patch_area = (
            expected_image_size[0]
            // expected_grid[0]
            * expected_image_size[1]
            // expected_grid[1]
        )
    expected_open_batch = int(cfg.open_dataloader.batch_size)
    total_batch = int(cfg.robot_dataloader.batch_size) + expected_open_batch
    expected_robot_ratio = (
        int(cfg.robot_dataloader.batch_size) / total_batch if total_batch > 0 else 0.0
    )
    expected_open_ratio = expected_open_batch / total_batch if total_batch > 0 else 0.0
    expected_camera_key = str(
        cfg.task.robot_dataset.get(
            "camera_key",
            cfg.task.get("camera_key", "camera0_rgb"),
        )
    )
    expected_normalizer_contract = gaze_wam_action_normalizer_contract(
        action_dim=task_action_dim,
        camera_key=expected_camera_key,
        heatmap_only=int(cfg.robot_dataloader.batch_size) <= 0,
    )
    expected_heatmap_token_kl_loss_weight = normalize_gaze_wam_nonnegative_float_field(
        "policy.heatmap_token_kl_loss_weight",
        cfg.policy.get("heatmap_token_kl_loss_weight", 0.0),
        default=0.0,
    )
    expected_heatmap_xy_loss_weight = normalize_gaze_wam_nonnegative_float_field(
        "policy.heatmap_xy_loss_weight",
        cfg.policy.get("heatmap_xy_loss_weight", 1.0),
        default=1.0,
    )
    expected_heatmap_point_nll_loss_weight = normalize_gaze_wam_nonnegative_float_field(
        "policy.heatmap_point_nll_loss_weight",
        cfg.policy.get("heatmap_point_nll_loss_weight", 0.0),
        default=0.0,
    )
    expected_heatmap_js_loss_weight = normalize_gaze_wam_nonnegative_float_field(
        "policy.heatmap_js_loss_weight",
        cfg.policy.get("heatmap_js_loss_weight", 1.0),
        default=1.0,
    )
    expected_heatmap_dsnt_temperature = normalize_gaze_wam_positive_float_field(
        "policy.heatmap_dsnt_temperature",
        cfg.policy.get("heatmap_dsnt_temperature", 0.1),
        default=0.1,
    )
    expected_heatmap_distribution_mode = str(
        cfg.policy.get("heatmap_distribution_mode", "intensity_softplus")
        or "intensity_softplus"
    ).strip()
    expected_heatmap_latent_scale = normalize_gaze_wam_positive_float_field(
        "policy.heatmap_latent_scale",
        cfg.policy.get("heatmap_latent_scale", 1.0),
        default=1.0,
    )
    expected_heatmap_latent_offset = float(
        cfg.policy.get("heatmap_latent_offset", 0.0)
    )
    expected_heatmap_latent_stats_path = str(
        cfg.policy.get("heatmap_latent_stats_path", "") or ""
    )
    expected_heatmap_scheduler_clip_sample = cfg.policy.get(
        "heatmap_scheduler_clip_sample",
        None,
    )
    if expected_heatmap_scheduler_clip_sample is not None:
        expected_heatmap_scheduler_clip_sample = normalize_gaze_wam_bool_field(
            "policy.heatmap_scheduler_clip_sample",
            expected_heatmap_scheduler_clip_sample,
            default=False,
        )
    expected_train_sequence_tokens = (
        int(summary["model_max_image_tokens"])
        + 1
        + task_action_horizon
        + task_heatmap_num_tokens
    )
    expected_inference_sequence_tokens = (
        int(summary["model_max_image_tokens"])
        + 1
        + task_action_horizon
    )
    inferred_patch_size = summary["inferred_patch_size_from_heatmap_grid"]
    obs_patch_size = summary.get("obs_encoder_patch_size")
    if isinstance(obs_patch_size, list) and len(obs_patch_size) == 2 and obs_patch_size[0] == obs_patch_size[1]:
        obs_patch_size_for_check = obs_patch_size[0]
    else:
        obs_patch_size_for_check = obs_patch_size
    pretrained_cfg = summary.get("obs_encoder_pretrained_cfg") or {}
    expected_normalize_mean = pretrained_cfg.get("mean")
    expected_normalize_std = pretrained_cfg.get("std")
    obs_encoder_pretrained = _summary_bool(
        "policy_contract.obs_encoder_pretrained",
        summary.get("obs_encoder_pretrained", False),
        default=False,
    )
    use_block_attention_mask = _summary_bool(
        "policy_contract.use_block_attention_mask",
        summary.get("use_block_attention_mask", False),
        default=False,
    )
    obs_encoder_local_weight_source_configured = _summary_bool(
        "policy_contract.obs_encoder_local_weight_source_configured",
        summary.get("obs_encoder_local_weight_source_configured", False),
        default=False,
    )
    obs_encoder_local_weight_source_exists = _summary_bool(
        "policy_contract.obs_encoder_local_weight_source_exists",
        summary.get("obs_encoder_local_weight_source_exists", False),
        default=False,
    )
    obs_encoder_local_weight_source_valid = _summary_bool(
        "policy_contract.obs_encoder_local_weight_source_valid",
        summary.get("obs_encoder_local_weight_source_valid", False),
        default=False,
    )
    obs_encoder_hf_hub_id = str(summary.get("obs_encoder_hf_hub_id") or "").strip()
    obs_encoder_hf_weight_source_configured = bool(obs_encoder_hf_hub_id)
    obs_encoder_weight_source_configured = (
        obs_encoder_local_weight_source_configured
        or obs_encoder_hf_weight_source_configured
    )
    obs_encoder_weight_source_exists = (
        obs_encoder_local_weight_source_exists
        if obs_encoder_local_weight_source_configured
        else obs_encoder_hf_weight_source_configured
    )
    obs_encoder_weight_source_valid = (
        obs_encoder_local_weight_source_valid
        if obs_encoder_local_weight_source_configured
        else obs_encoder_hf_weight_source_configured
    )
    obs_encoder_checkpoint_path_is_file = _summary_bool(
        "policy_contract.obs_encoder_checkpoint_path_is_file",
        summary.get("obs_encoder_checkpoint_path_is_file", False),
        default=False,
    )
    obs_encoder_cache_dir_is_dir = _summary_bool(
        "policy_contract.obs_encoder_cache_dir_is_dir",
        summary.get("obs_encoder_cache_dir_is_dir", False),
        default=False,
    )
    loss_routing_contract = summary.get("loss_routing_contract") or {}
    heatmap_latent_scaled_range = summary.get("heatmap_latent_scaled_range") or {}

    checks = [
        (
            expected_grid[0] * expected_grid[1] == task_heatmap_num_tokens,
            "Heatmap token grid product does not match task.heatmap_num_tokens.",
        ),
        (
            summary["visual_tokens_per_batch_item"] == expected_visual_tokens,
            "Visual token count does not match n_obs_steps * heatmap_num_tokens.",
        ),
        (
            abs(summary["visual_tokens_per_frame"] - task_heatmap_num_tokens) < 1e-9,
            "Visual tokens per frame do not match heatmap tokens per frame.",
        ),
        (
            summary["visual_embed_dim"] == expected_embed_dim,
            "Visual embedding dim does not match policy.n_emb.",
        ),
        (
            summary["model_n_emb"] == expected_embed_dim,
            "Transformer hidden dim does not match policy.n_emb.",
        ),
        (
            summary["model_max_image_tokens"] >= expected_visual_tokens,
            "Transformer max image-token capacity is smaller than visual token count.",
        ),
        (
            summary["model_action_dim"] == task_action_dim,
            "Transformer action dim does not match task.action_dim.",
        ),
        (
            summary["model_action_horizon"] == task_action_horizon,
            "Transformer action horizon does not match task.action_horizon.",
        ),
        (
            summary["model_heatmap_num_tokens"] == task_heatmap_num_tokens,
            "Transformer heatmap token count does not match task.heatmap_num_tokens.",
        ),
        (
            task_heatmap_dim == 16,
            (
                "Canonical FastWAM/LDM-aligned heatmap path requires "
                "task.heatmap_dim=16 latent channels on a 16x16 heatmap grid."
            ),
        ),
        (
            task_heatmap_dim > 0,
            "Heatmap latent dim must be a positive channel count.",
        ),
        (
            task_heatmap_dim != expected_patch_area,
            (
                "Canonical heatmap path must not use the old lossless patch area "
                f"({expected_patch_area}) as task.heatmap_dim."
            ),
        ),
        (
            summary.get("heatmap_spatial_decoder") == "cosmos_tokenizer",
            (
                "Canonical heatmap path requires "
                "policy.heatmap_spatial_decoder='cosmos_tokenizer'."
            ),
        ),
        (
            summary["model_heatmap_dim"] == int(cfg.policy.heatmap_dim),
            "Transformer heatmap dim does not match policy.heatmap_dim.",
        ),
        (
            summary["action_head_out_features"] == task_action_dim,
            "Action head output dim does not match task.action_dim.",
        ),
        (
            summary["heatmap_head_out_features"] == int(cfg.policy.heatmap_dim),
            "Heatmap head output dim does not match policy.heatmap_dim.",
        ),
        (
            summary.get("attention_contract") is not None,
            "Transformer attention contract summary is missing.",
        ),
        (
            (summary.get("attention_contract") or {}).get("use_block_attention_mask")
            == use_block_attention_mask,
            "Attention contract block-mask flag does not match transformer setting.",
        ),
        (
            (summary.get("attention_contract") or {}).get("train_sequence_tokens")
            == expected_train_sequence_tokens,
            "Attention contract train sequence length is inconsistent with model/token config.",
        ),
        (
            (summary.get("attention_contract") or {}).get("inference_sequence_tokens")
            == expected_inference_sequence_tokens,
            "Attention contract inference sequence length is inconsistent with model/token config.",
        ),
        (
            (summary.get("attention_contract") or {}).get("condition_reads_targets") is False
            and (summary.get("attention_contract") or {}).get("action_reads_heatmap") is False
            and (summary.get("attention_contract") or {}).get("heatmap_reads_action") is False
            and (summary.get("attention_contract") or {}).get(
                "action_inference_drops_heatmap"
            )
            is True,
            "Attention contract does not match the Gaze-WAM block-mask policy.",
        ),
        (
            summary.get("loss_routing_contract") is not None,
            "Policy loss routing contract summary is missing.",
        ),
        (
            loss_routing_contract.get("source") == "policy"
            and loss_routing_contract.get("dynamic_head_freezing")
            is False
            and loss_routing_contract.get("action_loss_mask") == "(~is_open) & has_action"
            and loss_routing_contract.get("heatmap_loss_mask")
            in (
                "has_heatmap",
                "has_heatmap & has_gaze_label",
                "has_heatmap & has_gaze_label for dsnt_js",
            ),
            "Loss routing contract does not match the Gaze-WAM mask policy.",
        ),
        (
            (loss_routing_contract.get("open_rows") or {}).get("trains_action")
            is False
            and (loss_routing_contract.get("open_rows") or {}).get("trains_heatmap")
            in (
                True,
                "xy DSNT plus generated Gaussian JS target",
                "latent diffusion MSE against generated Cosmos target",
                "latent diffusion MSE plus decoded final heatmap XY/NLL/JS loss",
            )
            and (loss_routing_contract.get("robot_real_gaze_rows") or {}).get("trains_action")
            is True
            and (loss_routing_contract.get("robot_real_gaze_rows") or {}).get("trains_heatmap")
            is False
            and (loss_routing_contract.get("robot_masked_gaze_rows") or {}).get("trains_action")
            is True,
            "Loss routing row semantics do not match the Gaze-WAM supervision policy.",
        ),
        (
            gaze_wam_loss_routing_validation_guardrails_ok(loss_routing_contract),
            "Loss routing validation guardrails do not match the Gaze-WAM tensor contract.",
        ),
        (
            summary.get("normalizer_contract") is not None,
            "Policy normalizer contract summary is missing.",
        ),
        (
            (summary.get("normalizer_contract") or {}).get("source")
            == expected_normalizer_contract["source"]
            and (summary.get("normalizer_contract") or {}).get("action_normalizer_source")
            == expected_normalizer_contract["action_normalizer_source"]
            and (summary.get("normalizer_contract") or {}).get("normalizer_keys")
            == expected_normalizer_contract["normalizer_keys"]
            and (summary.get("normalizer_contract") or {}).get("action_dim")
            == expected_normalizer_contract["action_dim"]
            and (summary.get("normalizer_contract") or {}).get(
                "action_representation"
            )
            == expected_normalizer_contract["action_representation"]
            and (summary.get("normalizer_contract") or {}).get(
                "robot_zarr_action_storage"
            )
            == expected_normalizer_contract["robot_zarr_action_storage"]
            and (summary.get("normalizer_contract") or {}).get(
                "excludes_open_source_dummy_actions"
            )
            is True
            and (summary.get("normalizer_contract") or {}).get(
                "open_source_get_normalizer_allowed"
            )
            is False,
            (
                "Policy normalizer contract does not match robot-relative-only "
                "Gaze-WAM provenance."
            ),
        ),
        (
            summary["heatmap_codec_token_grid"] == expected_grid,
            "Heatmap codec token grid does not match task.heatmap_token_grid.",
        ),
        (
            summary["heatmap_codec_image_size"] == expected_image_size,
            "Heatmap codec image size does not match task.image_shape.",
        ),
        (
            abs(
                float(summary.get("heatmap_latent_scale", 1.0))
                - expected_heatmap_latent_scale
            )
            < 1e-9,
            "Policy heatmap latent scale does not match config.",
        ),
        (
            abs(
                float(summary.get("heatmap_latent_offset", 0.0))
                - expected_heatmap_latent_offset
            )
            < 1e-9,
            "Policy heatmap latent offset does not match config.",
        ),
        (
            str(summary.get("heatmap_latent_stats_path", ""))
            == expected_heatmap_latent_stats_path,
            "Policy heatmap latent stats path does not match config.",
        ),
        (
            (not expected_heatmap_latent_stats_path)
            or (summary.get("heatmap_latent_stats") or {}).get("exists") is True,
            "Configured heatmap latent stats file is missing.",
        ),
        (
            (not expected_heatmap_latent_stats_path)
            or heatmap_latent_scaled_range.get("within_clip") is True,
            (
                "Policy heatmap latent scale/offset would clip the observed raw "
                "Cosmos label range under scheduler clip_sample=[-1, 1]."
            ),
        ),
        (
            expected_heatmap_scheduler_clip_sample is None
            or summary.get("heatmap_scheduler_clip_sample")
            is expected_heatmap_scheduler_clip_sample,
            "Policy heatmap scheduler clip_sample override does not match config.",
        ),
        (
            abs(
                float(summary.get("heatmap_token_kl_loss_weight", 0.0))
                - expected_heatmap_token_kl_loss_weight
            )
            < 1e-9,
            "Policy heatmap token KL loss weight does not match config.",
        ),
        (
            abs(
                float(summary.get("heatmap_xy_loss_weight", 0.0))
                - expected_heatmap_xy_loss_weight
            )
            < 1e-9,
            "Policy heatmap DSNT xy loss weight does not match config.",
        ),
        (
            abs(
                float(summary.get("heatmap_point_nll_loss_weight", 0.0))
                - expected_heatmap_point_nll_loss_weight
            )
            < 1e-9,
            "Policy heatmap point NLL loss weight does not match config.",
        ),
        (
            abs(
                float(summary.get("heatmap_js_loss_weight", 0.0))
                - expected_heatmap_js_loss_weight
            )
            < 1e-9,
            "Policy heatmap JS loss weight does not match config.",
        ),
        (
            abs(
                float(summary.get("heatmap_dsnt_temperature", 0.0))
                - expected_heatmap_dsnt_temperature
            )
            < 1e-9,
            "Policy heatmap DSNT temperature does not match config.",
        ),
        (
            str(summary.get("heatmap_distribution_mode", ""))
            == expected_heatmap_distribution_mode,
            "Policy heatmap distribution mode does not match config.",
        ),
        (
            str(summary.get("heatmap_distribution_mode", ""))
            == "intensity_softplus",
            (
                "Canonical frozen-Cosmos heatmap path interprets decoded outputs "
                "as nonnegative intensity distributions; set "
                "policy.heatmap_distribution_mode='intensity_softplus'."
            ),
        ),
        (
            obs_patch_size_for_check in (None, inferred_patch_size),
            "Obs encoder patch size does not match heatmap-grid patch size.",
        ),
        (
            _has_normalize_transform(summary, expected_normalize_mean, expected_normalize_std),
            "Obs encoder transforms do not include the pretrained checkpoint normalization stats.",
        ),
        (
            (not obs_encoder_pretrained)
            or obs_encoder_weight_source_configured,
            (
                "Preflight requires a DINO weight source when "
                "policy.obs_encoder.pretrained=true; set policy.obs_encoder.checkpoint_path, "
                "policy.obs_encoder.cache_dir, or use a timm pretrained cfg with hf_hub_id."
            ),
        ),
        (
            (not obs_encoder_pretrained)
            or obs_encoder_weight_source_exists,
            (
                "Configured local DINO weight source does not exist; check "
                "policy.obs_encoder.checkpoint_path and policy.obs_encoder.cache_dir."
            ),
        ),
        (
            (not obs_encoder_pretrained)
            or obs_encoder_weight_source_valid,
            (
                "Configured local DINO weight source is not structurally valid; "
                "policy.obs_encoder.checkpoint_path must be a file and "
                "policy.obs_encoder.cache_dir must be a directory."
            ),
        ),
        (
            (not obs_encoder_pretrained)
            or not summary.get("obs_encoder_checkpoint_path")
            or obs_encoder_checkpoint_path_is_file,
            "Configured policy.obs_encoder.checkpoint_path must point to a file.",
        ),
        (
            (not obs_encoder_pretrained)
            or not summary.get("obs_encoder_cache_dir")
            or obs_encoder_cache_dir_is_dir,
            "Configured policy.obs_encoder.cache_dir must point to a directory.",
        ),
        (
            summary["gaze_mask_token_shape"] == [1, 1, expected_embed_dim],
            "Gaze mask token shape does not match policy.n_emb.",
        ),
        (
            abs(summary["robot_ratio"] - expected_robot_ratio) < 1e-9,
            "Robot/open dataloader ratio summary is inconsistent with config batch sizes.",
        ),
        (
            abs(summary["open_ratio"] - expected_open_ratio) < 1e-9,
            "Open dataloader ratio summary is inconsistent with config batch sizes.",
        ),
    ]
    for ok, message in checks:
        if not ok:
            errors.append(message)
    return errors


def _preflight_loader_kwargs(dataloader_cfg) -> Dict[str, object]:
    _ensure_preflight_runtime()
    loader_kwargs = OmegaConf.to_container(dataloader_cfg, resolve=True)
    loader_kwargs["shuffle"] = False
    loader_kwargs["num_workers"] = 0
    loader_kwargs["pin_memory"] = False
    loader_kwargs["persistent_workers"] = False
    return loader_kwargs


def _preflight_dataloader(dataset, dataloader_cfg):
    return DataLoader(dataset, **_preflight_loader_kwargs(dataloader_cfg))


def _first_batch(dataset, dataloader_cfg) -> Dict[str, torch.Tensor]:
    loader = _preflight_dataloader(dataset, dataloader_cfg)
    return next(iter(loader))


def _load_policy_for_contract(
    cfg,
    checkpoint: Optional[str],
    device: str,
    use_ema: bool,
    overrides: Optional[Sequence[str]] = None,
    trust_checkpoint: bool = False,
):
    _ensure_preflight_runtime()
    if checkpoint is not None:
        policy, _ = load_policy_for_eval(
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
            use_ema=use_ema,
            overrides=overrides,
            trust_checkpoint=trust_checkpoint,
        )
    else:
        policy = hydra.utils.instantiate(cfg.policy)
        policy.eval().to(torch.device(device))
    return policy


def _record_zarr_validation_warnings(
    summary: Dict[str, object],
    source_name: str,
    validation: Dict[str, object],
    fail_on_zarr_warning: bool,
) -> None:
    warnings = list(validation.get("warnings", []) or [])
    for warning in warnings:
        summary["warnings"].append(f"{source_name} zarr warning: {warning}")
    if fail_on_zarr_warning and warnings:
        summary["errors"].append(
            f"{source_name} zarr validation produced {len(warnings)} warning(s)."
        )


def _zarr_presence_mask_summary(summary: Dict[str, object]) -> Dict[str, object]:
    presence = {}
    for source_name, validation_key in (
        ("robot", "robot_zarr_validation"),
        ("open", "open_zarr_validation"),
    ):
        validation = summary.get(validation_key)
        if not isinstance(validation, dict):
            continue
        masks = validation.get("presence_masks", {})
        if not isinstance(masks, dict):
            masks = {}
        presence[source_name] = {
            "validation_key": validation_key,
            "mask_keys": sorted(str(key) for key in masks.keys()),
            "masks": masks,
        }
    return presence


def preflight_gaze_wam(
    config_name: str,
    overrides: Optional[Sequence[str]] = None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    validate_zarr: bool = True,
    run_loss_smoke: bool = True,
    use_ema: bool = True,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    fail_on_zarr_warning: bool = False,
    trust_checkpoint: bool = False,
) -> Dict[str, object]:
    """Run local sanity checks before long Gaze-WAM training/evaluation jobs."""
    _ensure_preflight_runtime()
    cfg = load_cfg(config_name, overrides=overrides)
    cfg = _normalize_gaze_wam_early_bool_config(cfg)
    device_obj = torch.device(device)
    training_config = validate_gaze_wam_training_config(cfg)
    task_routing_config = validate_gaze_wam_task_routing_config(cfg)
    if training_config["valid"] and task_routing_config["valid"]:
        cfg = _normalize_gaze_wam_training_config(cfg, training_config)
        cfg = _normalize_gaze_wam_task_routing_config(cfg, task_routing_config)

    summary: Dict[str, object] = {
        "config_name": config_name,
        "overrides": list(overrides or []),
        "checkpoint": checkpoint or "",
        "device": str(device_obj),
        "ok": True,
        "errors": [],
        "warnings": [],
        "config": {
            "task_name": str(cfg.task.name),
            "n_obs_steps": normalize_gaze_wam_positive_int_field(
                "task.n_obs_steps",
                cfg.task.n_obs_steps,
            ),
            "n_latency_steps": normalize_gaze_wam_nonnegative_int_field(
                "task.n_latency_steps",
                cfg.task.get("n_latency_steps", 0),
            ),
            "action_horizon": normalize_gaze_wam_positive_int_field(
                "task.action_horizon",
                cfg.task.action_horizon,
            ),
            "action_dim": normalize_gaze_wam_positive_int_field(
                "task.action_dim",
                cfg.task.action_dim,
            ),
            "heatmap_num_tokens": normalize_gaze_wam_positive_int_field(
                "task.heatmap_num_tokens",
                cfg.task.heatmap_num_tokens,
            ),
            "heatmap_token_grid": normalize_gaze_wam_positive_int_sequence(
                "task.heatmap_token_grid",
                cfg.task.heatmap_token_grid,
                length=2,
            ),
            "robot_batch_size": int(training_config["robot_batch_size"]),
            "open_batch_size": int(training_config["open_batch_size"]),
            "robot_dataset_sampling": _dataset_sampling_summary(cfg.task.robot_dataset),
            "open_dataset_sampling": _dataset_sampling_summary(cfg.task.open_dataset),
            "policy_target": str(cfg.policy._target_),
            "fail_on_zarr_warning": bool(fail_on_zarr_warning),
        },
    }
    summary["training_config"] = training_config
    summary["task_routing_config"] = task_routing_config
    summary["errors"].extend(training_config["errors"])
    summary["errors"].extend(task_routing_config["errors"])
    summary["sampling_contract"] = _sampling_contract_summary(cfg)
    summary["errors"].extend(_check_sampling_contract(summary["sampling_contract"]))
    summary["image_geometry"] = _image_geometry_summary(cfg)
    summary["errors"].extend(_check_image_geometry_contract(summary["image_geometry"]))
    training_loop_config_valid = bool(training_config["valid"])
    task_routing_config_valid = bool(task_routing_config["valid"])
    if not training_loop_config_valid or not task_routing_config_valid:
        skip_reason = (
            "invalid training_config"
            if not training_loop_config_valid
            else "invalid task_routing_config"
        )
        summary["skipped_checks"] = [
            "robot_dataset",
            "open_dataset",
            "data_stream_contract",
            "dataset_lengths",
            "dataloader_batches",
            "zarr_validation",
            "policy_contract",
            "loss_smoke",
        ]
        summary["robot_dataset"] = {"skipped": skip_reason}
        summary["open_dataset"] = {"skipped": skip_reason}
        summary["data_stream_contract"] = {"skipped": skip_reason}
        summary["dataset_lengths"] = {"skipped": skip_reason}
        summary["dataloader_batches"] = {"skipped": skip_reason}
        if validate_zarr:
            summary["robot_zarr_validation"] = {"skipped": skip_reason}
            summary["open_zarr_validation"] = {"skipped": skip_reason}
        summary["policy_contract"] = {"skipped": skip_reason}
        summary["zarr_presence_masks"] = {}
        summary["ok"] = len(summary["errors"]) == 0
        return summary

    summary["data_stream_contract"] = _data_stream_contract_summary(cfg, training_config)
    summary["errors"].extend(
        _check_data_stream_contract(summary["data_stream_contract"])
    )

    robot_dataset = None
    robot_val_dataset = []
    robot_batch_size = int(training_config["robot_batch_size"])
    open_dataset = None
    open_val_dataset = []
    open_batch_size = int(training_config["open_batch_size"])
    if robot_batch_size > 0:
        try:
            robot_dataset = hydra.utils.instantiate(cfg.task.robot_dataset)
            try:
                robot_val_dataset = robot_dataset.get_validation_dataset()
            except Exception as exc:
                robot_val_dataset = []
                summary["errors"].append(
                    f"Robot validation dataset check failed: {type(exc).__name__}: {exc}"
                )
            summary["robot_dataset"] = {
                "path": str(cfg.task.robot_dataset.dataset_path),
                "length": int(len(robot_dataset)),
                "val_length": int(len(robot_val_dataset)),
                "sample": _sample_summary(robot_dataset[0]) if len(robot_dataset) > 0 else None,
            }
        except Exception as exc:
            summary["errors"].append(f"Robot dataset check failed: {type(exc).__name__}: {exc}")
            robot_dataset = None
            robot_val_dataset = []
    else:
        summary["robot_dataset"] = {
            "path": str(cfg.task.robot_dataset.dataset_path),
            "length": 0,
            "skipped": "robot_dataloader.batch_size <= 0",
        }

    if open_batch_size > 0:
        try:
            open_dataset = hydra.utils.instantiate(cfg.task.open_dataset)
            try:
                open_val_dataset = open_dataset.get_validation_dataset()
            except Exception as exc:
                open_val_dataset = []
                summary["errors"].append(
                    f"Open validation dataset check failed: {type(exc).__name__}: {exc}"
                )
            summary["open_dataset"] = {
                "path": str(cfg.task.open_dataset.dataset_path),
                "length": int(len(open_dataset)),
                "val_length": int(len(open_val_dataset)),
                "sample": _sample_summary(open_dataset[0]) if len(open_dataset) > 0 else None,
            }
        except Exception as exc:
            summary["errors"].append(f"Open dataset check failed: {type(exc).__name__}: {exc}")
    else:
        summary["open_dataset"] = {
            "path": str(cfg.task.open_dataset.dataset_path),
            "length": 0,
            "skipped": "open_dataloader.batch_size <= 0",
        }

    if (
        (robot_batch_size <= 0 or robot_dataset is not None)
        and (open_batch_size <= 0 or open_dataset is not None)
    ):
        try:
            summary["dataset_lengths"] = _check_training_dataset_lengths(
                robot_dataset=robot_dataset,
                robot_val_dataset=robot_val_dataset,
                open_dataset=open_dataset,
                open_val_dataset=open_val_dataset,
                robot_batch_size=robot_batch_size,
                open_batch_size=open_batch_size,
            )
        except Exception as exc:
            summary["dataset_lengths"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["errors"].append(
                f"Preflight dataset-length check failed: {type(exc).__name__}: {exc}"
            )

    robot_preflight_dataloader = None
    open_preflight_dataloader = None
    if robot_dataset is not None:
        try:
            robot_preflight_dataloader = _preflight_dataloader(
                robot_dataset,
                cfg.robot_dataloader,
            )
        except Exception as exc:
            summary["errors"].append(
                f"Robot preflight dataloader construction failed: {type(exc).__name__}: {exc}"
            )
    if open_dataset is not None:
        try:
            open_preflight_dataloader = _preflight_dataloader(
                open_dataset,
                cfg.open_dataloader,
            )
        except Exception as exc:
            summary["errors"].append(
                f"Open preflight dataloader construction failed: {type(exc).__name__}: {exc}"
            )
    if (
        (robot_batch_size <= 0 or robot_dataset is not None)
        and (open_batch_size <= 0 or open_dataset is not None)
    ):
        try:
            summary["dataloader_batches"] = _check_training_dataloader_lengths(
                robot_dataloader=robot_preflight_dataloader,
                robot_val_dataloader=None,
                open_dataloader=open_preflight_dataloader,
                open_val_dataloader=None,
                robot_batch_size=robot_batch_size,
                open_batch_size=open_batch_size,
            )
        except Exception as exc:
            summary["dataloader_batches"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["errors"].append(
                f"Preflight dataloader batch-count check failed: {type(exc).__name__}: {exc}"
            )

    if validate_zarr:
        if robot_dataset is not None:
            try:
                summary["robot_zarr_validation"] = validate_gaze_wam_zarr(
                    dataset_path=str(cfg.task.robot_dataset.dataset_path),
                    dataset_type="robot",
                    camera_key=str(cfg.task.robot_dataset.camera_key),
                    gaze_key=as_optional_gaze_wam_key(cfg.task.robot_dataset.get("gaze_key", None)),
                    heatmap_key=as_optional_gaze_wam_key(cfg.task.robot_dataset.get("heatmap_key", None)),
                    action_abs_key=str(cfg.task.robot_dataset.action_abs_key),
                    tcp_pose_key=str(cfg.task.robot_dataset.tcp_pose_key),
                    gripper_key=str(cfg.task.robot_dataset.gripper_key),
                    n_obs_steps=normalize_gaze_wam_positive_int_field(
                        "task.n_obs_steps",
                        cfg.task.n_obs_steps,
                    ),
                    action_horizon=normalize_gaze_wam_positive_int_field(
                        "task.action_horizon",
                        cfg.task.action_horizon,
                    ),
                    n_latency_steps=normalize_gaze_wam_nonnegative_int_field(
                        "task.n_latency_steps",
                        cfg.task.n_latency_steps,
                    ),
                    image_size=cfg.task.robot_dataset.image_size,
                    image_resize_mode=str(cfg.task.robot_dataset.get("image_resize_mode", "stretch")),
                    heatmap_token_grid=cfg.task.heatmap_token_grid,
                    heatmap_dim=normalize_gaze_wam_positive_int_field(
                        "task.heatmap_dim",
                        cfg.task.heatmap_dim,
                    ),
                    require_timestamps=require_timestamps,
                    timestamp_max_delta=timestamp_max_delta,
                    timestamp_max_step=timestamp_max_step,
                    check_dataset_sample=True,
                )
                if not summary["robot_zarr_validation"]["valid"]:
                    summary["errors"].append("Robot zarr validation failed.")
                _record_zarr_validation_warnings(
                    summary=summary,
                    source_name="Robot",
                    validation=summary["robot_zarr_validation"],
                    fail_on_zarr_warning=fail_on_zarr_warning,
                )
            except Exception as exc:
                summary["errors"].append(f"Robot zarr validation failed: {type(exc).__name__}: {exc}")
        if open_dataset is not None:
            try:
                summary["open_zarr_validation"] = validate_gaze_wam_zarr(
                    dataset_path=str(cfg.task.open_dataset.dataset_path),
                    dataset_type="open",
                    camera_key=str(cfg.task.open_dataset.camera_key),
                    gaze_key=as_optional_gaze_wam_key(cfg.task.open_dataset.get("gaze_key", None)),
                    heatmap_key=as_optional_gaze_wam_key(cfg.task.open_dataset.get("heatmap_key", None)),
                    n_obs_steps=normalize_gaze_wam_positive_int_field(
                        "task.n_obs_steps",
                        cfg.task.n_obs_steps,
                    ),
                    action_horizon=normalize_gaze_wam_positive_int_field(
                        "task.action_horizon",
                        cfg.task.action_horizon,
                    ),
                    n_latency_steps=normalize_gaze_wam_nonnegative_int_field(
                        "task.n_latency_steps",
                        cfg.task.n_latency_steps,
                    ),
                    image_size=cfg.task.open_dataset.image_size,
                    image_resize_mode=str(cfg.task.open_dataset.get("image_resize_mode", "stretch")),
                    heatmap_token_grid=cfg.task.heatmap_token_grid,
                    heatmap_dim=normalize_gaze_wam_positive_int_field(
                        "task.heatmap_dim",
                        cfg.task.heatmap_dim,
                    ),
                    action_dim=normalize_gaze_wam_positive_int_field(
                        "task.action_dim",
                        cfg.task.action_dim,
                    ),
                    require_timestamps=require_timestamps,
                    timestamp_max_delta=timestamp_max_delta,
                    timestamp_max_step=timestamp_max_step,
                    check_dataset_sample=True,
                )
                if not summary["open_zarr_validation"]["valid"]:
                    summary["errors"].append("Open zarr validation failed.")
                _record_zarr_validation_warnings(
                    summary=summary,
                    source_name="Open",
                    validation=summary["open_zarr_validation"],
                    fail_on_zarr_warning=fail_on_zarr_warning,
                )
            except Exception as exc:
                summary["errors"].append(f"Open zarr validation failed: {type(exc).__name__}: {exc}")

    summary["zarr_presence_masks"] = _zarr_presence_mask_summary(summary)

    policy = None
    try:
        policy = _load_policy_for_contract(
            cfg=cfg,
            checkpoint=checkpoint,
            device=device,
            use_ema=use_ema,
            overrides=overrides,
            trust_checkpoint=trust_checkpoint,
        )
        contract = _policy_contract_summary(policy, cfg)
        summary["policy_contract"] = contract
        summary["errors"].extend(_check_policy_contract(contract, cfg))
    except Exception as exc:
        summary["errors"].append(f"Policy contract check failed: {type(exc).__name__}: {exc}")

    if policy is not None and checkpoint is None:
        try:
            if robot_dataset is not None:
                policy.set_normalizer(robot_dataset.get_normalizer())
            elif robot_batch_size <= 0:
                camera_key = str(
                    cfg.task.robot_dataset.get(
                        "camera_key",
                        cfg.task.get("camera_key", "camera0_rgb"),
                    )
                )
                policy.set_normalizer(_make_heatmap_only_identity_normalizer(camera_key))
        except Exception as exc:
            summary["errors"].append(f"Policy normalizer setup failed: {type(exc).__name__}: {exc}")
            policy = None

    if (
        run_loss_smoke
        and policy is not None
        and (robot_dataset is not None or open_dataset is not None)
    ):
        try:
            robot_batch = (
                _first_batch(robot_dataset, cfg.robot_dataloader)
                if robot_dataset is not None
                else None
            )
            open_batch = _first_batch(open_dataset, cfg.open_dataloader) if open_dataset is not None else None
            if robot_batch is not None:
                robot_batch = dict_apply(robot_batch, lambda x: x.to(device_obj))
            if open_batch is not None:
                open_batch = dict_apply(open_batch, lambda x: x.to(device_obj))
            mixed = build_gaze_wam_mixed_batch(
                robot_batch=robot_batch,
                open_batch=open_batch,
                robot_gaze_dropout_prob=float(cfg.task.robot_gaze_dropout_prob),
                robot_heatmap_on_gaze_dropout=normalize_gaze_wam_bool_field(
                    "task.robot_heatmap_on_gaze_dropout",
                    cfg.task.get("robot_heatmap_on_gaze_dropout", True),
                    default=True,
                ),
                shuffle=False,
            )
            with torch.no_grad():
                components = policy.compute_loss_components(mixed, return_per_sample=True)
            summary["loss_smoke"] = {
                "loss": _to_float(components["loss"]),
                "action_loss": _to_float(components["action_loss"]),
                "heatmap_loss": _to_float(components["heatmap_loss"]),
                "heatmap_xy_loss": _to_float(components["heatmap_xy_loss"]),
                "heatmap_point_nll_loss": _to_float(
                    components["heatmap_point_nll_loss"]
                ),
                "heatmap_js_loss": _to_float(components["heatmap_js_loss"]),
                "heatmap_token_kl_loss": _to_float(
                    components["heatmap_token_kl_loss"]
                ),
                "heatmap_token_kl_loss_weight": float(
                    components["heatmap_token_kl_loss_weight"]
                ),
                "heatmap_xy_loss_weight": float(
                    components["heatmap_xy_loss_weight"]
                ),
                "heatmap_point_nll_loss_weight": float(
                    components["heatmap_point_nll_loss_weight"]
                ),
                "heatmap_js_loss_weight": float(
                    components["heatmap_js_loss_weight"]
                ),
                "heatmap_diffusion_final_loss_enabled": bool(
                    components["heatmap_diffusion_final_loss_enabled"]
                ),
                "heatmap_final_loss_timestep_weighting": str(
                    components["heatmap_final_loss_timestep_weighting"]
                ),
                "heatmap_dsnt_temperature": float(
                    components["heatmap_dsnt_temperature"]
                ),
                "heatmap_distribution_mode": str(
                    components["heatmap_distribution_mode"]
                ),
                "action_loss_mask_count": _to_float(components["action_loss_mask_count"]),
                "heatmap_loss_mask_count": _to_float(components["heatmap_loss_mask_count"]),
                "heatmap_xy_loss_mask_count": _to_float(
                    components["heatmap_xy_loss_mask_count"]
                ),
                "mixed_batch_size": int(mixed["gaze_xy"].shape[0]),
                "mixed_obs": _shape(mixed["obs"]),
                "mixed_action": _shape(mixed["action"]),
                "mixed_heatmap": _shape(mixed["heatmap"]),
                "routing": loss_routing_summary(
                    mixed=mixed,
                    action_loss_mask=components["action_loss_mask"],
                    heatmap_loss_mask=components["heatmap_loss_mask"],
                ),
            }
            if not torch.isfinite(components["loss"]):
                summary["errors"].append("Loss smoke produced non-finite loss.")
        except Exception as exc:
            summary["errors"].append(f"Loss smoke failed: {type(exc).__name__}: {exc}")

    summary["ok"] = len(summary["errors"]) == 0
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Run local preflight checks for Gaze-WAM configs.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Acknowledge that the supplied dill checkpoint is trusted and may execute code.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-zarr-validation", action="store_true")
    parser.add_argument("--skip-loss-smoke", action="store_true")
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--fail-on-zarr-warning", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = preflight_gaze_wam(
        config_name=args.config_name,
        overrides=args.override,
        checkpoint=args.checkpoint,
        device=args.device,
        validate_zarr=not args.skip_zarr_validation,
        run_loss_smoke=not args.skip_loss_smoke,
        use_ema=args.use_ema,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        fail_on_zarr_warning=args.fail_on_zarr_warning,
        trust_checkpoint=args.trust_checkpoint,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if not summary["ok"]:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
