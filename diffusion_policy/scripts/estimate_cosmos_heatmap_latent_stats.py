from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from diffusion_policy.dataset.gaze_wam_dataset import (
    _build_sample_indices,
    _infer_image_hw,
    _normalize_gaze_xy,
    _open_zarr_root,
    _presence_mask_values_to_vector,
    _resolve_data_group,
)
from diffusion_policy.model.gaze_wam.heatmap_decoder import CosmosHeatmapCodec


def _positive_int_pair(name: str, value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers, got {value!r}.")
    parsed = (int(value[0]), int(value[1]))
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError(f"{name} must contain positive integers, got {value!r}.")
    return parsed


def _positive_int(name: str, value: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return parsed


def _nonnegative_int(name: str, value: int) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return parsed


def _positive_float(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}.")
    return parsed


def _optional_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in ("none", "null"):
        return None
    return value


def _shape_to_image_hw(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) == 4:
        if shape[-1] in (1, 3, 4):
            return int(shape[1]), int(shape[2])
        if shape[1] in (1, 3, 4):
            return int(shape[2]), int(shape[3])
    if len(shape) == 5:
        if shape[-1] in (1, 3, 4):
            return int(shape[2]), int(shape[3])
        if shape[2] in (1, 3, 4):
            return int(shape[3]), int(shape[4])
    raise ValueError(f"Cannot infer image H/W from camera array shape {tuple(shape)}.")


def _valid_mask(data_group, key: str, current_idx: int) -> bool:
    if key not in data_group:
        return True
    mask = _presence_mask_values_to_vector(key, data_group[key][current_idx])
    if mask.size != 1:
        raise ValueError(
            f"{key} must provide exactly one value for row {current_idx}, got {mask.shape}."
        )
    return bool(mask.item())


def _target_heatmap_from_xy(
    gaze_xy: torch.Tensor,
    image_size: tuple[int, int],
    sigma_px: float,
) -> torch.Tensor:
    height, width = image_size
    y = (torch.arange(height, device=gaze_xy.device, dtype=gaze_xy.dtype) + 0.5) / height
    x = (torch.arange(width, device=gaze_xy.device, dtype=gaze_xy.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma_x = float(sigma_px) / float(width)
    sigma_y = float(sigma_px) / float(height)
    gx = gaze_xy[:, 0].reshape(-1, 1, 1)
    gy = gaze_xy[:, 1].reshape(-1, 1, 1)
    dist = ((xx - gx) / sigma_x) ** 2 + ((yy - gy) / sigma_y) ** 2
    target = torch.exp(-0.5 * dist)
    target = target / target.flatten(start_dim=1).sum(dim=-1, keepdim=True).reshape(
        -1,
        1,
        1,
    ).clamp_min(1e-12)
    return target


def _select_dataset_indices(
    total: int,
    max_samples: int,
    stride: int,
    seed: Optional[int],
) -> np.ndarray:
    stride = _positive_int("stride", stride)
    base = np.arange(0, int(total), stride, dtype=np.int64)
    if max_samples > 0 and base.size > int(max_samples):
        if seed is None:
            base = base[: int(max_samples)]
        else:
            rng = np.random.default_rng(int(seed))
            base = np.sort(rng.choice(base, size=int(max_samples), replace=False))
    return base.astype(np.int64)


def _summary_from_values(values: np.ndarray, prefix: str = "") -> Dict[str, object]:
    if values.size == 0:
        raise ValueError("Cannot summarize zero latent values.")
    quantiles = {
        "p00_1": 0.001,
        "p00_5": 0.005,
        "p01": 0.01,
        "p05": 0.05,
        "p50": 0.50,
        "p95": 0.95,
        "p99": 0.99,
        "p99_5": 0.995,
        "p99_9": 0.999,
    }
    result = {
        f"{prefix}count": int(values.size),
        f"{prefix}min": float(np.min(values)),
        f"{prefix}max": float(np.max(values)),
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}std": float(np.std(values)),
    }
    for name, q in quantiles.items():
        result[f"{prefix}{name}"] = float(np.quantile(values, q))
    return result


def _scale_recommendations(values: np.ndarray, clip_target: float) -> Dict[str, object]:
    abs_values = np.abs(values.astype(np.float64, copy=False))
    levels = {
        "abs_p95": 0.95,
        "abs_p99": 0.99,
        "abs_p99_5": 0.995,
        "abs_p99_9": 0.999,
    }
    rec = {"clip_target": float(clip_target)}
    for name, q in levels.items():
        value = float(np.quantile(abs_values, q))
        rec[name] = value
        rec[f"scale_for_{name}"] = float(clip_target / max(value, 1e-12))
    abs_max = float(np.max(abs_values))
    rec["abs_max"] = abs_max
    rec["scale_for_abs_max"] = float(clip_target / max(abs_max, 1e-12))
    rounded_safe = float(np.floor(rec["scale_for_abs_max"] * 100.0) / 100.0)
    rec["scale_for_abs_max_rounded_down_0p01"] = (
        rounded_safe if rounded_safe > 0.0 else rec["scale_for_abs_max"]
    )
    rec["recommended_default"] = rec["scale_for_abs_max_rounded_down_0p01"]
    rec["recommended_default_basis"] = "scale_for_abs_max_rounded_down_0p01"
    return rec


def _iter_batches(values: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def estimate_cosmos_heatmap_latent_stats(
    dataset_path: str,
    encoder_path: str,
    decoder_path: str,
    output_path: str,
    camera_key: str = "camera0_rgb",
    gaze_key: str = "gaze_xy",
    has_gaze_label_key: str = "has_gaze_label",
    image_size: Sequence[int] = (256, 256),
    token_grid: Sequence[int] = (16, 16),
    latent_channels: int = 16,
    sigma_px: float = 6.0,
    action_horizon: int = 16,
    n_latency_steps: int = 0,
    action_downsample_steps: int = 1,
    action_padding: bool = True,
    val_ratio: float = 0.0,
    seed: Optional[int] = None,
    max_samples: int = 2048,
    stride: int = 1,
    batch_size: int = 32,
    device: str = "cuda:0",
    dtype: str = "float32",
    gaze_is_normalized: bool = True,
    input_range: str = "minus_one_one",
    output_range: str = "minus_one_one",
    input_normalization: str = "max",
    clip_target: float = 1.0,
) -> Dict[str, object]:
    image_size = _positive_int_pair("image_size", image_size)
    token_grid = _positive_int_pair("token_grid", token_grid)
    latent_channels = _positive_int("latent_channels", latent_channels)
    sigma_px = _positive_float("sigma_px", sigma_px)
    action_horizon = _positive_int("action_horizon", action_horizon)
    n_latency_steps = _nonnegative_int("n_latency_steps", n_latency_steps)
    action_downsample_steps = _positive_int("action_downsample_steps", action_downsample_steps)
    max_samples = int(max_samples)
    if max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {max_samples}.")
    batch_size = _positive_int("batch_size", batch_size)
    clip_target = _positive_float("clip_target", clip_target)
    gaze_key = _optional_key(gaze_key)
    if gaze_key is None:
        raise ValueError("gaze_key must not be null for latent statistics.")

    root, store = _open_zarr_root(dataset_path)
    try:
        data_group, episode_ends = _resolve_data_group(root)
        if camera_key not in data_group:
            raise KeyError(f"Missing camera key {camera_key!r} in {dataset_path}.")
        if gaze_key not in data_group:
            raise KeyError(f"Missing gaze key {gaze_key!r} in {dataset_path}.")
        camera_shape = tuple(int(v) for v in data_group[camera_key].shape)
        source_image_hw = _shape_to_image_hw(camera_shape)
        episode_mask = None
        if val_ratio > 0.0:
            from diffusion_policy.common.sampler import get_val_mask

            episode_mask = ~get_val_mask(
                n_episodes=len(episode_ends),
                val_ratio=float(val_ratio),
                seed=0 if seed is None else int(seed),
            )
        sample_indices = _build_sample_indices(
            episode_ends=episode_ends,
            action_horizon=action_horizon,
            n_latency_steps=n_latency_steps,
            action_downsample_steps=action_downsample_steps,
            action_padding=bool(action_padding),
            episode_mask=episode_mask,
        )
        selected_dataset_indices = _select_dataset_indices(
            total=len(sample_indices),
            max_samples=max_samples,
            stride=stride,
            seed=seed,
        )
        if selected_dataset_indices.size == 0:
            raise ValueError("No samples selected for latent statistics.")

        device_obj = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        torch_dtype = torch.float32
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype != "float32":
            raise ValueError("dtype must be one of: float32, float16, bfloat16.")
        codec = CosmosHeatmapCodec(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            token_grid=token_grid,
            image_size=image_size,
            latent_channels=latent_channels,
            input_range=input_range,
            output_range=output_range,
            input_normalization=input_normalization,
        ).to(device_obj)
        codec.eval()

        raw_chunks = []
        channel_sum = np.zeros(latent_channels, dtype=np.float64)
        channel_sum_sq = np.zeros(latent_channels, dtype=np.float64)
        channel_min = np.full(latent_channels, np.inf, dtype=np.float64)
        channel_max = np.full(latent_channels, -np.inf, dtype=np.float64)
        channel_count = 0
        valid_rows = 0
        skipped_no_gaze = 0
        selected_current_indices = []
        with torch.no_grad():
            for batch_indices in _iter_batches(selected_dataset_indices.tolist(), batch_size):
                gaze_batch = []
                current_batch = []
                for dataset_idx in batch_indices:
                    current_idx = int(sample_indices[int(dataset_idx)][0])
                    if not _valid_mask(data_group, has_gaze_label_key, current_idx):
                        skipped_no_gaze += 1
                        continue
                    gaze_xy = _normalize_gaze_xy(
                        data_group[gaze_key][current_idx],
                        source_image_size=source_image_hw,
                        gaze_is_normalized=bool(gaze_is_normalized),
                    )
                    gaze_batch.append(gaze_xy)
                    current_batch.append(current_idx)
                if not gaze_batch:
                    continue
                gaze_tensor = torch.from_numpy(np.stack(gaze_batch, axis=0)).to(
                    device=device_obj,
                    dtype=torch_dtype,
                )
                target_heatmap = _target_heatmap_from_xy(
                    gaze_xy=gaze_tensor,
                    image_size=image_size,
                    sigma_px=sigma_px,
                )
                tokens = codec.encode_image(target_heatmap).detach().float().cpu().numpy()
                raw_chunks.append(tokens.reshape(-1).astype(np.float32, copy=False))
                per_channel = tokens.reshape(-1, latent_channels).astype(np.float64, copy=False)
                channel_sum += per_channel.sum(axis=0)
                channel_sum_sq += (per_channel * per_channel).sum(axis=0)
                channel_min = np.minimum(channel_min, per_channel.min(axis=0))
                channel_max = np.maximum(channel_max, per_channel.max(axis=0))
                channel_count += int(per_channel.shape[0])
                valid_rows += int(tokens.shape[0])
                selected_current_indices.extend(int(v) for v in current_batch)

        if not raw_chunks:
            raise ValueError("No valid gaze rows were encoded.")
        raw_values = np.concatenate(raw_chunks, axis=0)
        channel_mean = channel_sum / max(channel_count, 1)
        channel_var = channel_sum_sq / max(channel_count, 1) - channel_mean * channel_mean
        channel_std = np.sqrt(np.maximum(channel_var, 0.0))
        summary = {
            "dataset_path": str(dataset_path),
            "encoder_path": str(encoder_path),
            "decoder_path": str(decoder_path),
            "camera_key": str(camera_key),
            "gaze_key": str(gaze_key),
            "has_gaze_label_key": str(has_gaze_label_key),
            "codec": {
                "name": "CosmosHeatmapCodec",
                "input_range": input_range,
                "output_range": output_range,
                "input_normalization": input_normalization,
                "token_grid": [int(v) for v in token_grid],
                "image_size": [int(v) for v in image_size],
                "latent_channels": int(latent_channels),
            },
            "target_heatmap": {
                "source": "gaze_xy_to_fullres_gaussian",
                "sigma_px": float(sigma_px),
                "sum_normalized": True,
                "policy_matches": "GazeWamPolicy._target_heatmap_image_from_xy",
            },
            "sampling": {
                "dataset_sample_count": int(len(sample_indices)),
                "selected_rows": int(selected_dataset_indices.size),
                "valid_gaze_rows_encoded": int(valid_rows),
                "skipped_no_gaze": int(skipped_no_gaze),
                "max_samples": int(max_samples),
                "stride": int(stride),
                "seed": None if seed is None else int(seed),
                "batch_size": int(batch_size),
                "current_indices_head": selected_current_indices[:20],
            },
            "raw_latent": _summary_from_values(raw_values),
            "raw_latent_abs": _summary_from_values(np.abs(raw_values), prefix="abs_"),
            "per_channel": {
                "count_per_channel": int(channel_count),
                "min": [float(v) for v in channel_min.tolist()],
                "max": [float(v) for v in channel_max.tolist()],
                "mean": [float(v) for v in channel_mean.tolist()],
                "std": [float(v) for v in channel_std.tolist()],
            },
            "scale_recommendations": _scale_recommendations(raw_values, clip_target=clip_target),
            "notes": [
                "Stats are computed before any policy heatmap_latent_scale/offset.",
                "Use scale on raw Cosmos latents before DDIM/DDPM scheduler training.",
                "Frozen Cosmos decoder should receive denormalized raw latents.",
            ],
        }
        output = pathlib.Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        if store is not None:
            store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate raw Cosmos heatmap latent statistics from the same "
            "gaze_xy -> full-resolution Gaussian heatmap -> frozen Cosmos encoder "
            "path used by Gaze-WAM DSNT/JS training."
        )
    )
    parser.add_argument("--dataset-path", default="data/hot3d_open.zarr")
    parser.add_argument("--encoder-path", default="data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/encoder.jit")
    parser.add_argument("--decoder-path", default="data/checkpoints/cosmos_tokenizer/Cosmos-Tokenizer-CI16x16/decoder.jit")
    parser.add_argument("--output-path", default="data/outputs/cosmos_heatmap_latent_stats/summary.json")
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--has-gaze-label-key", default="has_gaze_label")
    parser.add_argument("--image-size", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--token-grid", nargs=2, type=int, default=(16, 16))
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--sigma-px", type=float, default=6.0)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--n-latency-steps", type=int, default=0)
    parser.add_argument("--action-downsample-steps", type=int, default=1)
    parser.add_argument("--action-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-range", choices=("zero_one", "minus_one_one"), default="minus_one_one")
    parser.add_argument("--output-range", choices=("zero_one", "minus_one_one"), default="minus_one_one")
    parser.add_argument("--input-normalization", choices=("none", "max", "mass"), default="max")
    parser.add_argument("--clip-target", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = estimate_cosmos_heatmap_latent_stats(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
