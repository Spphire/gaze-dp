from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, Optional, Sequence, Tuple


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def add_values(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.sums[name] = self.sums.get(name, 0.0) + float(values.sum().item())
        self.counts[name] = self.counts.get(name, 0.0) + float(values.numel())

    def add_mean(self, name: str, mean_value: torch.Tensor, count: torch.Tensor) -> None:
        count_value = float(count.detach().float().item())
        if count_value <= 0:
            return
        self.sums[name] = self.sums.get(name, 0.0) + float(mean_value.detach().float().item()) * count_value
        self.counts[name] = self.counts.get(name, 0.0) + count_value

    def add_scalar(self, name: str, value: torch.Tensor) -> None:
        self.add_values(name, value.reshape(1))

    def summary(self, prefix: str) -> Dict[str, float]:
        result = {}
        for key in sorted(self.sums):
            count = self.counts[key]
            if count > 0:
                result[f"{prefix}_{key}"] = self.sums[key] / count
                result[f"{prefix}_{key}_count"] = count
        return result


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _ensure_torch_runtime():
    global DataLoader, F, dict_apply, torch
    try:
        return torch
    except NameError:
        import torch as _torch
        import torch.nn.functional as _F
        from torch.utils.data import DataLoader as _DataLoader
        from diffusion_policy.common.pytorch_util import dict_apply as _dict_apply

        torch = _torch
        F = _F
        DataLoader = _DataLoader
        dict_apply = _dict_apply
        return torch


def _ensure_hydra_runtime():
    global OmegaConf, hydra
    try:
        hydra_module = hydra
    except NameError:
        import hydra as _hydra
        from omegaconf import OmegaConf as _OmegaConf

        hydra = _hydra
        OmegaConf = _OmegaConf
        hydra_module = hydra
    from diffusion_policy.common.omegaconf_resolvers import (
        register_safe_omegaconf_resolvers,
    )

    register_safe_omegaconf_resolvers()
    return hydra_module


def _ensure_gaze_wam_eval_runtime():
    _ensure_torch_runtime()
    global build_gaze_wam_mixed_batch
    try:
        build_gaze_wam_mixed_batch
    except NameError:
        from diffusion_policy.dataset.gaze_wam_mixing import (
            build_gaze_wam_mixed_batch as _build_gaze_wam_mixed_batch,
        )

        build_gaze_wam_mixed_batch = _build_gaze_wam_mixed_batch


def _ensure_validation_runtime():
    global as_optional_gaze_wam_key, validate_gaze_wam_zarr
    try:
        validate_gaze_wam_zarr
    except NameError:
        from diffusion_policy.common.gaze_utils import (
            as_optional_gaze_wam_key as _as_optional_gaze_wam_key,
        )
        from diffusion_policy.scripts.validate_gaze_wam_zarr import (
            validate_gaze_wam_zarr as _validate_gaze_wam_zarr,
        )

        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        validate_gaze_wam_zarr = _validate_gaze_wam_zarr


def _move_batch_to_device(batch, device: torch.device):
    _ensure_torch_runtime()
    return dict_apply(batch, lambda x: x.to(device, non_blocking=True))


def _slice_batch(batch, mask: torch.Tensor):
    def apply_mask(value):
        if isinstance(value, dict):
            return {key: apply_mask(item) for key, item in value.items()}
        return value[mask]

    return {key: apply_mask(value) for key, value in batch.items()}


def _obs_from_batch(batch, include_action_base: bool = False) -> Dict[str, torch.Tensor]:
    obs = dict(batch["obs"])
    obs["gaze_xy"] = batch["gaze_xy"]
    obs["has_gaze_label"] = batch["has_gaze_label"]
    obs["use_gaze_condition"] = batch["use_gaze_condition"]
    if include_action_base and "action_base_abs" in batch:
        obs["action_base_abs"] = batch["action_base_abs"]
        if "has_action_base_abs" in batch:
            obs["has_action_base_abs"] = batch["has_action_base_abs"]
    return obs


def _require_bool_vector(name: str, value: torch.Tensor, batch_size: int) -> torch.Tensor:
    _ensure_torch_runtime()
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be a BoolTensor, got dtype {value.dtype}.")
    if value.shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape [B] with B={batch_size}, got {tuple(value.shape)}."
        )
    return value


def _action_abs_mask(batch, has_action: torch.Tensor) -> torch.Tensor:
    _ensure_torch_runtime()
    if "action_abs" not in batch or "action_base_abs" not in batch:
        return torch.zeros_like(has_action)
    batch_size = int(has_action.shape[0])
    has_action_abs = batch.get("has_action_abs")
    if has_action_abs is None:
        has_action_abs = has_action
    else:
        has_action_abs = (
            _require_bool_vector("batch['has_action_abs']", has_action_abs, batch_size)
            .to(device=has_action.device)
            & has_action
        )
    has_action_base_abs = batch.get("has_action_base_abs")
    if has_action_base_abs is None:
        has_action_base_abs = has_action
    else:
        has_action_base_abs = (
            _require_bool_vector(
                "batch['has_action_base_abs']",
                has_action_base_abs,
                batch_size,
            ).to(device=has_action.device)
            & has_action
        )
    return has_action_abs & has_action_base_abs


def _optional_presence_mask(
    batch,
    optional_key: str,
    default_mask: torch.Tensor,
) -> torch.Tensor:
    _ensure_torch_runtime()
    if optional_key not in batch:
        return torch.zeros_like(default_mask)
    batch_size = int(default_mask.shape[0])
    mask_key = f"has_{optional_key}"
    mask = batch.get(mask_key)
    if mask is None:
        return default_mask
    return (
        _require_bool_vector(f"batch['{mask_key}']", mask, batch_size)
        .to(device=default_mask.device)
        & default_mask
    )


def _heatmap_target(batch, dtype: torch.dtype) -> torch.Tensor:
    heatmap = batch["heatmap"].to(dtype=dtype)
    if heatmap.ndim == 4:
        if heatmap.shape[1] != 1:
            raise ValueError(
                "Only heatmap_horizon=1 is supported by the offline evaluator, "
                f"got {heatmap.shape}."
            )
        heatmap = heatmap[:, 0]
    return heatmap


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _ensure_torch_runtime()
    return F.mse_loss(pred, target, reduction="none").flatten(start_dim=1).mean(dim=1)


def _heatmap_kl(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.squeeze(-1).clamp_min(eps)
    target = target.squeeze(-1).clamp_min(0.0)
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (target * (target.clamp_min(eps).log() - pred.log())).sum(dim=-1)


def _heatmap_argmax_xy(tokens: torch.Tensor, token_grid: Tuple[int, int]) -> torch.Tensor:
    _ensure_torch_runtime()
    height, width = token_grid
    index = tokens.squeeze(-1).argmax(dim=-1)
    row = torch.div(index, width, rounding_mode="floor")
    col = index.remainder(width)
    x = (col.to(tokens.dtype) + 0.5) / width
    y = (row.to(tokens.dtype) + 0.5) / height
    return torch.stack([x, y], dim=-1)


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _require_bool_vector("metric mask", mask, int(values.shape[0])).to(device=values.device)
    return values[mask]


def _mask_count(mask: torch.Tensor) -> float:
    _ensure_torch_runtime()
    return float(mask.detach().to(dtype=torch.float32).sum().item())


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    _ensure_torch_runtime()
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def evaluate_gaze_wam_dataset(
    *args,
    **kwargs,
) -> Dict[str, float]:
    """Evaluate one Gaze-WAM source dataset with action, heatmap, and GDR metrics."""
    torch = _ensure_torch_runtime()
    _ensure_gaze_wam_eval_runtime()
    with torch.no_grad():
        return _evaluate_gaze_wam_dataset_impl(*args, **kwargs)


def _evaluate_gaze_wam_dataset_impl(
    policy: GazeWamPolicy,
    dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
    source_name: str = "eval",
    cfg_scale: Optional[float] = None,
    compute_denoising_loss: bool = True,
    compute_sampling: bool = True,
    compute_heatmap: bool = True,
    compute_gdr: bool = True,
    source_is_robot: bool = False,
    robot_gaze_dropout_prob: float = 0.0,
    robot_heatmap_on_gaze_dropout: bool = True,
    robot_gaze_dropout_seed: Optional[int] = None,
) -> Dict[str, float]:
    if device is None:
        device = policy.device
    device = torch.device(device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    was_training = policy.training
    policy.eval().to(device)
    accumulator = MetricAccumulator()
    num_batches = 0
    num_samples = 0
    coverage_counts = {
        "action_supervision_count": 0.0,
        "action_abs_supervision_count": 0.0,
        "action_abs_metric_eligible_count": 0.0,
        "heatmap_supervision_count": 0.0,
        "has_action_abs_count": 0.0,
        "has_action_base_abs_count": 0.0,
        "has_heatmap_image_count": 0.0,
        "gaze_label_count": 0.0,
        "gaze_condition_count": 0.0,
        "gdr_eligible_count": 0.0,
        "denoise_action_mask_count": 0.0,
        "denoise_heatmap_mask_count": 0.0,
    }

    robot_dropout_generator = _make_generator(device, robot_gaze_dropout_seed)
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        if source_is_robot:
            batch = build_gaze_wam_mixed_batch(
                robot_batch=batch,
                open_batch=None,
                robot_gaze_dropout_prob=robot_gaze_dropout_prob,
                robot_heatmap_on_gaze_dropout=robot_heatmap_on_gaze_dropout,
                generator=robot_dropout_generator,
                shuffle=False,
            )
        batch_size_actual = int(batch["gaze_xy"].shape[0])
        num_batches += 1
        num_samples += batch_size_actual

        has_action = _require_bool_vector(
            "batch['has_action']",
            batch["has_action"],
            batch_size_actual,
        )
        has_heatmap = _require_bool_vector(
            "batch['has_heatmap']",
            batch["has_heatmap"],
            batch_size_actual,
        )
        has_gaze_label = _require_bool_vector(
            "batch['has_gaze_label']",
            batch["has_gaze_label"],
            batch_size_actual,
        )
        use_gaze_condition = _require_bool_vector(
            "batch['use_gaze_condition']",
            batch["use_gaze_condition"],
            batch_size_actual,
        )
        has_action_abs = _action_abs_mask(batch, has_action)
        row_mask = torch.ones_like(has_action)
        has_action_abs_tensor = _optional_presence_mask(batch, "action_abs", has_action)
        has_action_base_abs_tensor = _optional_presence_mask(
            batch,
            "action_base_abs",
            has_action,
        )
        has_heatmap_image = _optional_presence_mask(batch, "heatmap_image", row_mask)
        gdr_mask = has_action & has_gaze_label & use_gaze_condition

        coverage_counts["action_supervision_count"] += _mask_count(has_action)
        coverage_counts["action_abs_supervision_count"] += _mask_count(has_action_abs)
        coverage_counts["action_abs_metric_eligible_count"] += _mask_count(has_action_abs)
        coverage_counts["heatmap_supervision_count"] += _mask_count(has_heatmap)
        coverage_counts["has_action_abs_count"] += _mask_count(has_action_abs_tensor)
        coverage_counts["has_action_base_abs_count"] += _mask_count(has_action_base_abs_tensor)
        coverage_counts["has_heatmap_image_count"] += _mask_count(has_heatmap_image)
        coverage_counts["gaze_label_count"] += _mask_count(has_gaze_label)
        coverage_counts["gaze_condition_count"] += _mask_count(use_gaze_condition)
        coverage_counts["gdr_eligible_count"] += _mask_count(gdr_mask)

        if compute_denoising_loss:
            components = policy.compute_loss_components(batch)
            coverage_counts["denoise_action_mask_count"] += float(
                components["action_loss_mask_count"].detach().float().item()
            )
            coverage_counts["denoise_heatmap_mask_count"] += float(
                components["heatmap_loss_mask_count"].detach().float().item()
            )
            accumulator.add_scalar("denoise_loss", components["loss"])
            accumulator.add_mean(
                "denoise_action_loss",
                components["action_loss"],
                components["action_loss_mask_count"],
            )
            accumulator.add_mean(
                "denoise_heatmap_loss",
                components["heatmap_loss"],
                components["heatmap_loss_mask_count"],
            )

        if compute_sampling and has_action.any():
            pred = policy.predict_action(
                _obs_from_batch(batch, include_action_base=False),
                cfg_scale=cfg_scale,
            )
            pred_relative = pred["action_pred_relative"].to(dtype=batch["action"].dtype)
            action_mse = _per_sample_mse(pred_relative, batch["action"])
            accumulator.add_values("action_mse", _masked_values(action_mse, has_action))

            if has_action_abs.any():
                abs_batch = _slice_batch(batch, has_action_abs)
                abs_pred = policy.predict_action(
                    _obs_from_batch(abs_batch, include_action_base=True),
                    cfg_scale=cfg_scale,
                )
                action_abs_mse = _per_sample_mse(
                    abs_pred["action_pred_abs"].to(dtype=abs_batch["action_abs"].dtype),
                    abs_batch["action_abs"],
                )
                accumulator.add_values(
                    "action_abs_mse",
                    action_abs_mse,
                )

        if compute_heatmap and has_heatmap.any():
            heatmap_target = _heatmap_target(batch, dtype=policy.dtype)
            pred_heatmap = policy.predict_heatmap(
                _obs_from_batch(batch),
                use_gaze_condition=batch["use_gaze_condition"],
                timestep=torch.zeros(batch_size_actual, device=device, dtype=torch.long),
                decode=False,
            )["heatmap_tokens"]
            heatmap_mse = _per_sample_mse(pred_heatmap, heatmap_target)
            accumulator.add_values("heatmap_mse", _masked_values(heatmap_mse, has_heatmap))
            heatmap_kl = _heatmap_kl(pred_heatmap, heatmap_target)
            accumulator.add_values("heatmap_kl", _masked_values(heatmap_kl, has_heatmap))

            pred_xy = _heatmap_argmax_xy(pred_heatmap, policy.heatmap_codec.token_grid)
            target_xy = _heatmap_argmax_xy(heatmap_target, policy.heatmap_codec.token_grid)
            argmax_l2 = torch.linalg.vector_norm(pred_xy - target_xy, dim=-1)
            accumulator.add_values("heatmap_argmax_l2", _masked_values(argmax_l2, has_heatmap))

        if compute_gdr:
            if gdr_mask.any():
                gdr_batch = _slice_batch(batch, gdr_mask)
                gdr = policy.compute_gaze_dependency_ratio(_obs_from_batch(gdr_batch))
                accumulator.add_values("feature_gdr", gdr["feature_gdr"])
                accumulator.add_values("output_gdr", gdr["output_gdr"])

    if was_training:
        policy.train()

    result = accumulator.summary(source_name)
    result[f"{source_name}_num_batches"] = float(num_batches)
    result[f"{source_name}_num_samples"] = float(num_samples)
    for name, count in sorted(coverage_counts.items()):
        result[f"{source_name}_{name}"] = float(count)
    return result


def load_cfg(config_name: str, overrides: Optional[Sequence[str]] = None):
    hydra = _ensure_hydra_runtime()
    from diffusion_policy.common.gaze_wam_training_config import (
        _normalize_gaze_wam_early_bool_config,
    )

    config_dir = pathlib.Path(__file__).resolve().parents[1].joinpath("config")
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides or []))
    OmegaConf.resolve(cfg)
    return _normalize_gaze_wam_early_bool_config(cfg)


def load_policy_for_eval(
    cfg=None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    use_ema: bool = True,
    overrides: Optional[Sequence[str]] = None,
    trust_checkpoint: bool = False,
) -> Tuple[GazeWamPolicy, object]:
    hydra = _ensure_hydra_runtime()
    torch = _ensure_torch_runtime()
    import dill
    from diffusion_policy.common.checkpoint_security import require_trusted_pickle_artifact

    overrides = list(overrides or [])
    device_obj = torch.device(device)
    if checkpoint is not None:
        checkpoint = require_trusted_pickle_artifact(
            checkpoint,
            trusted=trust_checkpoint,
            artifact_name="Gaze-WAM checkpoint",
        )
        payload = torch.load(
            checkpoint.open("rb"),
            pickle_module=dill,
            map_location=device_obj,
        )
        cfg = payload["cfg"]
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
            OmegaConf.resolve(cfg)
        from diffusion_policy.common.gaze_wam_training_config import (
            _normalize_gaze_wam_early_bool_config,
        )

        cfg = _normalize_gaze_wam_early_bool_config(cfg)
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model if use_ema and cfg.training.use_ema else workspace.model
    else:
        if cfg is None:
            raise ValueError("cfg is required when checkpoint is not provided.")
        robot_dataset = hydra.utils.instantiate(cfg.task.robot_dataset)
        policy = hydra.utils.instantiate(cfg.policy)
        policy.set_normalizer(robot_dataset.get_normalizer())

    policy.eval().to(device_obj)
    return policy, cfg


def _validation_error(source_name: str, summary: Dict[str, object]) -> ValueError:
    errors = summary.get("errors", [])
    message = "; ".join(str(error) for error in errors) if errors else "unknown validation error"
    return ValueError(f"{source_name} zarr validation failed: {message}")


def _validate_eval_source_zarr(
    cfg,
    source_name: str,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    robot_gaze_dropout_seed: Optional[int] = None,
) -> Dict[str, object]:
    _ensure_validation_runtime()
    common_kwargs = dict(
        n_obs_steps=int(cfg.task.n_obs_steps),
        action_horizon=int(cfg.task.action_horizon),
        n_latency_steps=int(cfg.task.n_latency_steps),
        heatmap_token_grid=cfg.task.heatmap_token_grid,
        heatmap_dim=int(cfg.task.heatmap_dim),
        action_dim=int(cfg.task.action_dim),
        timestamp_key=timestamp_key,
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
        check_dataset_sample=True,
    )
    if source_name == "robot":
        dataset_cfg = cfg.task.robot_dataset
        summary = validate_gaze_wam_zarr(
            dataset_path=str(dataset_cfg.dataset_path),
            dataset_type="robot",
            camera_key=str(dataset_cfg.camera_key),
            gaze_key=as_optional_gaze_wam_key(dataset_cfg.get("gaze_key", None)),
            heatmap_key=as_optional_gaze_wam_key(dataset_cfg.get("heatmap_key", None)),
            action_abs_key=str(dataset_cfg.action_abs_key),
            tcp_pose_key=str(dataset_cfg.tcp_pose_key),
            gripper_key=str(dataset_cfg.gripper_key),
            image_size=dataset_cfg.image_size,
            image_resize_mode=str(dataset_cfg.get("image_resize_mode", "stretch")),
            **common_kwargs,
        )
    elif source_name == "open":
        dataset_cfg = cfg.task.open_dataset
        summary = validate_gaze_wam_zarr(
            dataset_path=str(dataset_cfg.dataset_path),
            dataset_type="open",
            camera_key=str(dataset_cfg.camera_key),
            gaze_key=as_optional_gaze_wam_key(dataset_cfg.get("gaze_key", None)),
            heatmap_key=as_optional_gaze_wam_key(dataset_cfg.get("heatmap_key", None)),
            image_size=dataset_cfg.image_size,
            image_resize_mode=str(dataset_cfg.get("image_resize_mode", "stretch")),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown evaluation source {source_name!r}.")

    if not summary["valid"]:
        raise _validation_error(source_name, summary)
    return summary


def evaluate_gaze_wam_sources(
    policy: GazeWamPolicy,
    cfg,
    sources: Iterable[str] = ("robot", "open"),
    batch_size: int = 16,
    num_workers: int = 0,
    max_batches: Optional[int] = None,
    device: str = "cpu",
    cfg_scale: Optional[float] = None,
    compute_denoising_loss: bool = True,
    compute_sampling: bool = True,
    compute_heatmap: bool = True,
    compute_gdr: bool = True,
    validate_zarr: bool = True,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    robot_gaze_dropout_seed: Optional[int] = None,
) -> Dict[str, object]:
    hydra = _ensure_hydra_runtime()
    torch = _ensure_torch_runtime()
    from diffusion_policy.common.gaze_wam_training_config import (
        normalize_gaze_wam_bool_field,
    )

    metrics: Dict[str, object] = {}
    source_set = set(sources)
    if validate_zarr:
        for source_name in sorted(source_set):
            metrics[f"{source_name}_zarr_validation"] = _validate_eval_source_zarr(
                cfg=cfg,
                source_name=source_name,
                timestamp_key=timestamp_key,
                require_timestamps=require_timestamps,
                timestamp_max_delta=timestamp_max_delta,
                timestamp_max_step=timestamp_max_step,
            )
    if "robot" in source_set:
        robot_dataset = hydra.utils.instantiate(cfg.task.robot_dataset)
        metrics.update(
            evaluate_gaze_wam_dataset(
                policy=policy,
                dataset=robot_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                max_batches=max_batches,
                device=torch.device(device),
                source_name="robot",
                cfg_scale=cfg_scale,
                compute_denoising_loss=compute_denoising_loss,
                compute_sampling=compute_sampling,
                compute_heatmap=compute_heatmap,
                compute_gdr=compute_gdr,
                source_is_robot=True,
                robot_gaze_dropout_prob=float(cfg.task.get("robot_gaze_dropout_prob", 0.0)),
                robot_heatmap_on_gaze_dropout=normalize_gaze_wam_bool_field(
                    "task.robot_heatmap_on_gaze_dropout",
                    cfg.task.get("robot_heatmap_on_gaze_dropout", True),
                    default=True,
                ),
                robot_gaze_dropout_seed=robot_gaze_dropout_seed,
            )
        )
    if "open" in source_set:
        open_dataset = hydra.utils.instantiate(cfg.task.open_dataset)
        metrics.update(
            evaluate_gaze_wam_dataset(
                policy=policy,
                dataset=open_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                max_batches=max_batches,
                device=torch.device(device),
                source_name="open",
                cfg_scale=cfg_scale,
                compute_denoising_loss=compute_denoising_loss,
                compute_sampling=compute_sampling,
                compute_heatmap=compute_heatmap,
                compute_gdr=compute_gdr,
            )
        )
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Offline Gaze-WAM action/heatmap/GDR evaluation.")
    parser.add_argument("--checkpoint", default=None, help="Optional workspace checkpoint path.")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Acknowledge that the supplied dill checkpoint is trusted and may execute code.",
    )
    parser.add_argument("--config-name", default=None, help="Hydra config name when no checkpoint is used.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override. Repeat this flag for multiple overrides.",
    )
    parser.add_argument("--device", default="cuda:0" if _torch_cuda_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sources", default="robot,open", help="Comma-separated subset of robot,open.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Optional global CFG scale override. Defaults to each policy config's cfg_scale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-denoising-loss", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-gdr", action="store_true")
    parser.add_argument("--validate-zarr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = parse_args(argv)
    torch = _ensure_torch_runtime()
    torch.manual_seed(args.seed)

    cfg = None
    if args.config_name is not None:
        cfg = load_cfg(args.config_name, args.override)
    if args.checkpoint is None and cfg is None:
        raise ValueError("Either --checkpoint or --config-name must be provided.")

    policy, cfg = load_policy_for_eval(
        cfg=cfg,
        checkpoint=args.checkpoint,
        device=args.device,
        use_ema=args.use_ema,
        overrides=args.override,
        trust_checkpoint=args.trust_checkpoint,
    )
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    metrics = evaluate_gaze_wam_sources(
        policy=policy,
        cfg=cfg,
        sources=sources,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        device=args.device,
        cfg_scale=args.cfg_scale,
        compute_denoising_loss=not args.skip_denoising_loss,
        compute_sampling=not args.skip_sampling,
        compute_heatmap=not args.skip_heatmap,
        compute_gdr=not args.skip_gdr,
        validate_zarr=args.validate_zarr,
        timestamp_key=args.timestamp_key,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
        robot_gaze_dropout_seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output is not None:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


if __name__ == "__main__":
    main()
