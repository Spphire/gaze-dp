from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


EVAL_MODES = ("image_only", "image_gt_gaze")


def _resolve_checkpoint_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "checkpoints" / "latest.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_episode(
    episode_ends: Sequence[int],
    episode_index: Optional[int] = None,
) -> Tuple[int, int, int]:
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("episode_ends must be a non-empty one-dimensional array.")
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    lengths = ends - starts
    if np.any(lengths <= 1):
        raise ValueError("Every evaluation episode must contain at least two frames.")
    if episode_index is None:
        episode_index = int(np.argmin(lengths))
    episode_index = int(episode_index)
    if episode_index < 0:
        episode_index += int(ends.size)
    if episode_index < 0 or episode_index >= int(ends.size):
        raise IndexError(
            f"episode_index {episode_index} is outside [0, {int(ends.size) - 1}]."
        )
    return episode_index, int(starts[episode_index]), int(ends[episode_index])


def build_valid_horizon_mask(
    current_indices: Sequence[int],
    episode_end: int,
    action_horizon: int,
    action_downsample_steps: int = 1,
    n_latency_steps: int = 0,
) -> np.ndarray:
    action_horizon = int(action_horizon)
    action_downsample_steps = int(action_downsample_steps)
    n_latency_steps = int(n_latency_steps)
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if action_downsample_steps <= 0:
        raise ValueError("action_downsample_steps must be positive.")
    if n_latency_steps < 0:
        raise ValueError("n_latency_steps cannot be negative.")
    current = np.asarray(current_indices, dtype=np.int64).reshape(-1, 1)
    horizon = np.arange(action_horizon, dtype=np.int64).reshape(1, -1)
    targets = current + 1 + (n_latency_steps + horizon) * action_downsample_steps
    return targets < int(episode_end)


def rotation_geodesic_degrees(pred_action: np.ndarray, gt_action: np.ndarray) -> np.ndarray:
    from diffusion_policy.common.pose_util import pose10d_to_mat

    pred = np.asarray(pred_action, dtype=np.float64)
    gt = np.asarray(gt_action, dtype=np.float64)
    if pred.shape != gt.shape or pred.shape[-1] < 9:
        raise ValueError(
            "pred_action and gt_action must have matching shapes with at least 9 action dims."
        )
    leading_shape = pred.shape[:-1]
    pred_rot = pose10d_to_mat(pred[..., :9].reshape(-1, 9))[:, :3, :3].reshape(
        leading_shape + (3, 3)
    )
    gt_rot = pose10d_to_mat(gt[..., :9].reshape(-1, 9))[:, :3, :3].reshape(
        leading_shape + (3, 3)
    )
    relative = np.swapaxes(gt_rot, -1, -2) @ pred_rot
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def compute_action_errors(
    pred_action_abs: np.ndarray,
    gt_action_abs: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, object]:
    pred = np.asarray(pred_action_abs, dtype=np.float64)
    gt = np.asarray(gt_action_abs, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 10:
        raise ValueError("Absolute action arrays must have matching shape [N, H, 10].")
    if mask.shape != pred.shape[:2]:
        raise ValueError(f"valid_mask must have shape {pred.shape[:2]}, got {mask.shape}.")

    translation = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)
    rotation = rotation_geodesic_degrees(pred, gt)
    gripper = np.abs(pred[..., 9] - gt[..., 9])
    per_horizon = []
    for horizon_idx in range(pred.shape[1]):
        horizon_mask = mask[:, horizon_idx]
        count = int(horizon_mask.sum())
        per_horizon.append(
            {
                "horizon_step": horizon_idx + 1,
                "count": count,
                "translation_mae_m": (
                    float(translation[horizon_mask, horizon_idx].mean()) if count else None
                ),
                "rotation_mae_deg": (
                    float(rotation[horizon_mask, horizon_idx].mean()) if count else None
                ),
                "gripper_mae_m": (
                    float(gripper[horizon_mask, horizon_idx].mean()) if count else None
                ),
            }
        )

    valid_translation = translation[mask]
    valid_rotation = rotation[mask]
    valid_gripper = gripper[mask]
    first_mask = mask[:, 0]
    return {
        "valid_action_count": int(mask.sum()),
        "translation_mae_m": float(valid_translation.mean()),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(valid_translation)))),
        "rotation_mae_deg": float(valid_rotation.mean()),
        "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(valid_rotation)))),
        "gripper_mae_m": float(valid_gripper.mean()),
        "gripper_rmse_m": float(np.sqrt(np.mean(np.square(valid_gripper)))),
        "first_step": {
            "count": int(first_mask.sum()),
            "translation_mae_m": float(translation[first_mask, 0].mean()),
            "rotation_mae_deg": float(rotation[first_mask, 0].mean()),
            "gripper_mae_m": float(gripper[first_mask, 0].mean()),
        },
        "per_horizon": per_horizon,
    }


def _to_device(batch, device):
    import torch

    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    return batch


def _policy_obs(batch, mode: str):
    import torch

    if mode not in EVAL_MODES:
        raise ValueError(f"Unsupported evaluation mode {mode!r}.")
    obs = dict(batch["obs"])
    batch_size = int(batch["gaze_xy"].shape[0])
    if mode == "image_only":
        obs["gaze_xy"] = torch.zeros_like(batch["gaze_xy"])
        obs["has_gaze_label"] = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=batch["gaze_xy"].device,
        )
        obs["use_gaze_condition"] = torch.zeros_like(obs["has_gaze_label"])
    else:
        has_gaze_label = batch["has_gaze_label"].to(dtype=torch.bool)
        obs["gaze_xy"] = batch["gaze_xy"]
        obs["has_gaze_label"] = has_gaze_label
        obs["use_gaze_condition"] = has_gaze_label.clone()
    return obs


def _seed_everything(seed: int) -> None:
    import torch

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def predict_episode_mode(
    policy,
    dataloader: Iterable,
    mode: str,
    device,
    seed: int,
    cfg_scale: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    import torch

    from diffusion_policy.common.action_utils import relative_actions_to_absolute_actions

    if hasattr(policy, "reset"):
        policy.reset()
    _seed_everything(seed)
    relative_rows = []
    absolute_rows = []
    gt_absolute_rows = []
    base_rows = []
    gaze_rows = []
    gaze_label_rows = []
    started = time.perf_counter()
    policy.eval().to(device)

    with torch.inference_mode():
        for batch in dataloader:
            batch = _to_device(batch, device)
            prediction = policy.predict_action(
                _policy_obs(batch, mode),
                cfg_scale=cfg_scale,
            )
            relative = prediction["action_pred_relative"].detach().float().cpu().numpy()
            base_abs = batch["action_base_abs"].detach().float().cpu().numpy()
            absolute = relative_actions_to_absolute_actions(relative, base_abs).astype(
                np.float32
            )
            relative_rows.append(relative.astype(np.float32))
            absolute_rows.append(absolute)
            gt_absolute_rows.append(
                batch["action_abs"].detach().float().cpu().numpy().astype(np.float32)
            )
            base_rows.append(base_abs.astype(np.float32))
            gaze_rows.append(batch["gaze_xy"].detach().float().cpu().numpy().astype(np.float32))
            gaze_label_rows.append(
                batch["has_gaze_label"].detach().cpu().numpy().astype(bool)
            )

    return {
        "action_relative": np.concatenate(relative_rows, axis=0),
        "action_abs": np.concatenate(absolute_rows, axis=0),
        "gt_action_abs": np.concatenate(gt_absolute_rows, axis=0),
        "action_base_abs": np.concatenate(base_rows, axis=0),
        "gaze_xy": np.concatenate(gaze_rows, axis=0),
        "has_gaze_label": np.concatenate(gaze_label_rows, axis=0),
        "elapsed_seconds": np.asarray(time.perf_counter() - started, dtype=np.float64),
    }


def validate_episode_predictions(
    results: Dict[str, Dict[str, np.ndarray]],
    expected_samples: int,
    action_horizon: int,
) -> None:
    expected_action_shape = (int(expected_samples), int(action_horizon), 10)
    expected_base_shape = (int(expected_samples), 10)
    for mode in EVAL_MODES:
        if mode not in results:
            raise KeyError(f"Missing evaluation mode {mode!r}.")
        mode_result = results[mode]
        for key in ("action_relative", "action_abs", "gt_action_abs"):
            value = np.asarray(mode_result[key])
            if value.shape != expected_action_shape:
                raise ValueError(
                    f"{mode}.{key} must have shape {expected_action_shape}, got {value.shape}."
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{mode}.{key} contains non-finite values.")
        base_abs = np.asarray(mode_result["action_base_abs"])
        if base_abs.shape != expected_base_shape:
            raise ValueError(
                f"{mode}.action_base_abs must have shape {expected_base_shape}, "
                f"got {base_abs.shape}."
            )
        if not np.all(np.isfinite(base_abs)):
            raise ValueError(f"{mode}.action_base_abs contains non-finite values.")
        gaze_xy = np.asarray(mode_result["gaze_xy"])
        has_gaze_label = np.asarray(mode_result["has_gaze_label"])
        if gaze_xy.shape != (int(expected_samples), 2):
            raise ValueError(
                f"{mode}.gaze_xy must have shape {(int(expected_samples), 2)}, "
                f"got {gaze_xy.shape}."
            )
        if has_gaze_label.shape != (int(expected_samples),):
            raise ValueError(
                f"{mode}.has_gaze_label must have shape {(int(expected_samples),)}, "
                f"got {has_gaze_label.shape}."
            )
        if not np.all(np.isfinite(gaze_xy)):
            raise ValueError(f"{mode}.gaze_xy contains non-finite values.")


def _continuous_euler_xyz_degrees(action_abs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    from diffusion_policy.common.pose_util import pose10d_to_mat

    matrices = pose10d_to_mat(np.asarray(action_abs)[..., :9])[..., :3, :3]
    euler = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_euler(
        "xyz", degrees=False
    )
    euler = np.unwrap(euler, axis=0)
    return np.degrees(euler).reshape(matrices.shape[:-2] + (3,))


def assemble_nonoverlapping_action_chunks(
    gt_action_abs: np.ndarray,
    predictions: Dict[str, np.ndarray],
    valid_horizon_mask: np.ndarray,
    action_downsample_steps: int = 1,
) -> Dict[str, object]:
    gt = np.asarray(gt_action_abs)
    valid = np.asarray(valid_horizon_mask, dtype=bool)
    if gt.ndim != 3 or gt.shape[-1] != 10:
        raise ValueError("gt_action_abs must have shape [N, H, 10].")
    if valid.shape != gt.shape[:2]:
        raise ValueError(f"valid_horizon_mask must have shape {gt.shape[:2]}, got {valid.shape}.")
    action_downsample_steps = int(action_downsample_steps)
    if action_downsample_steps <= 0:
        raise ValueError("action_downsample_steps must be positive.")
    for mode in EVAL_MODES:
        value = np.asarray(predictions[mode])
        if value.shape != gt.shape:
            raise ValueError(
                f"predictions[{mode!r}] must have shape {gt.shape}, got {value.shape}."
            )

    num_samples, action_horizon, _ = gt.shape
    chunk_stride = action_horizon * action_downsample_steps
    anchor_indices = np.arange(0, num_samples, chunk_stride, dtype=np.int64)
    curve_indices = []
    boundary_indices = []
    gt_chunks = []
    prediction_chunks = {mode: [] for mode in EVAL_MODES}
    curve_size = 0
    for anchor_idx in anchor_indices:
        valid_steps = np.flatnonzero(valid[anchor_idx])
        if not np.array_equal(valid_steps, np.arange(valid_steps.size, dtype=np.int64)):
            raise ValueError(
                "Each chunk anchor must have a prefix-valid horizon mask before padding."
            )
        target_curve_indices = anchor_idx + valid_steps * action_downsample_steps
        in_episode = target_curve_indices < num_samples
        valid_steps = valid_steps[in_episode]
        target_curve_indices = target_curve_indices[in_episode]
        if valid_steps.size == 0:
            continue
        boundary_indices.append(curve_size)
        curve_indices.append(target_curve_indices)
        gt_chunks.append(gt[anchor_idx, valid_steps])
        for mode in EVAL_MODES:
            prediction_chunks[mode].append(np.asarray(predictions[mode])[anchor_idx, valid_steps])
        curve_size += int(valid_steps.size)

    if not curve_indices:
        raise ValueError("No valid non-overlapping action chunks were produced.")
    curve_indices = np.concatenate(curve_indices, axis=0)
    if np.any(np.diff(curve_indices) <= 0):
        raise ValueError("Non-overlapping chunk target indices must be strictly increasing.")
    return {
        "curve_indices": curve_indices,
        "chunk_anchor_indices": anchor_indices,
        "chunk_boundary_indices": np.asarray(boundary_indices, dtype=np.int64),
        "gt_action_abs": np.concatenate(gt_chunks, axis=0),
        "predictions": {
            mode: np.concatenate(prediction_chunks[mode], axis=0) for mode in EVAL_MODES
        },
    }


def plot_absolute_action_curves(
    output_path: pathlib.Path,
    time_seconds: np.ndarray,
    gt_action_abs: np.ndarray,
    predictions: Dict[str, np.ndarray],
    chunk_boundary_indices: Sequence[int],
    modes: Optional[Sequence[str]] = None,
    figure_title: str = "Absolute non-overlapping action chunks vs ground truth",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_modes = tuple(EVAL_MODES if modes is None else modes)
    if not selected_modes:
        raise ValueError("At least one evaluation mode must be selected for plotting.")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError("Evaluation modes selected for plotting must be unique.")
    unsupported_modes = [mode for mode in selected_modes if mode not in EVAL_MODES]
    if unsupported_modes:
        raise ValueError(f"Unsupported evaluation modes for plotting: {unsupported_modes}.")

    gt_curve = np.asarray(gt_action_abs)
    pred_curves = {name: np.asarray(value) for name, value in predictions.items()}
    if gt_curve.ndim != 2 or gt_curve.shape[-1] != 10:
        raise ValueError("Chunked gt_action_abs must have shape [N, 10].")
    if np.asarray(time_seconds).shape != (gt_curve.shape[0],):
        raise ValueError("time_seconds must match the chunked action curve length.")
    for mode in selected_modes:
        if mode not in pred_curves:
            raise KeyError(f"Missing chunked predictions for evaluation mode {mode!r}.")
        if pred_curves[mode].shape != gt_curve.shape:
            raise ValueError(
                f"Chunked predictions[{mode!r}] must have shape {gt_curve.shape}."
            )
    boundaries = np.asarray(chunk_boundary_indices, dtype=np.int64)
    if boundaries.ndim != 1 or boundaries.size == 0 or boundaries[0] != 0:
        raise ValueError("chunk_boundary_indices must be a non-empty vector starting at zero.")
    if np.any(boundaries < 0) or np.any(boundaries >= gt_curve.shape[0]):
        raise ValueError("chunk_boundary_indices contains an out-of-range curve index.")

    gt_euler = _continuous_euler_xyz_degrees(gt_curve)
    pred_euler = {
        mode: _continuous_euler_xyz_degrees(pred_curves[mode]) for mode in selected_modes
    }
    colors = {"image_only": "#d97706", "image_gt_gaze": "#2563eb"}
    labels = {
        "image_only": "image only",
        "image_gt_gaze": "image + GT gaze",
    }
    series = [
        ("TCP x", "m", gt_curve[:, 0], {key: value[:, 0] for key, value in pred_curves.items()}),
        ("TCP y", "m", gt_curve[:, 1], {key: value[:, 1] for key, value in pred_curves.items()}),
        ("TCP z", "m", gt_curve[:, 2], {key: value[:, 2] for key, value in pred_curves.items()}),
        ("roll", "deg", gt_euler[:, 0], {key: value[:, 0] for key, value in pred_euler.items()}),
        ("pitch", "deg", gt_euler[:, 1], {key: value[:, 1] for key, value in pred_euler.items()}),
        ("yaw", "deg", gt_euler[:, 2], {key: value[:, 2] for key, value in pred_euler.items()}),
        ("gripper", "m", gt_curve[:, 9], {key: value[:, 9] for key, value in pred_curves.items()}),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(15, 18), sharex=True)
    for axis_idx, (axis, (title, unit, gt_values, mode_values)) in enumerate(zip(axes, series)):
        axis.plot(time_seconds, gt_values, color="#111827", linewidth=1.6, label="GT")
        for mode in selected_modes:
            axis.plot(
                time_seconds,
                mode_values[mode],
                color=colors[mode],
                linewidth=1.0,
                alpha=0.9,
                label=labels[mode],
            )
        for boundary_idx, curve_idx in enumerate(boundaries[1:], start=1):
            axis.axvline(
                time_seconds[curve_idx],
                color="#6b7280",
                linestyle="--",
                linewidth=0.8,
                alpha=0.55,
                label=(
                    "inference chunk boundary"
                    if axis_idx == 0 and boundary_idx == 1
                    else None
                ),
            )
        axis.set_ylabel(unit)
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=min(4, 2 + len(selected_modes)), loc="upper right")
    axes[-1].set_xlabel("episode time (s)")
    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_horizon_errors(
    output_path: pathlib.Path,
    metrics: Dict[str, Dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"image_only": "#d97706", "image_gt_gaze": "#2563eb"}
    labels = {
        "image_only": "image only",
        "image_gt_gaze": "image + GT gaze",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fields = [
        ("translation_mae_m", "translation MAE (cm)", 100.0),
        ("rotation_mae_deg", "rotation MAE (deg)", 1.0),
        ("gripper_mae_m", "gripper MAE (mm)", 1000.0),
    ]
    for axis, (field, ylabel, scale) in zip(axes, fields):
        for mode in EVAL_MODES:
            rows = metrics[mode]["per_horizon"]
            x = [row["horizon_step"] for row in rows]
            y = [np.nan if row[field] is None else row[field] * scale for row in rows]
            axis.plot(x, y, color=colors[mode], linewidth=1.8, label=labels[mode])
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("prediction horizon (frames at 30 Hz)")
    fig.suptitle("Absolute action error by prediction horizon", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _episode_time_seconds(
    dataset,
    current_indices: np.ndarray,
    sample_fps: float,
    action_downsample_steps: int = 1,
    n_latency_steps: int = 0,
) -> np.ndarray:
    target_offset = 1 + int(n_latency_steps) * int(action_downsample_steps)
    target_indices = current_indices + target_offset
    if "timestamp" in dataset.data_group:
        timestamps = np.asarray(dataset.data_group["timestamp"][target_indices], dtype=np.float64)
        if np.all(np.isfinite(timestamps)) and timestamps.size:
            return timestamps - timestamps[0]
    return np.arange(target_indices.size, dtype=np.float64) / float(sample_fps)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_episode(args) -> Dict[str, object]:
    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict
    from torch.utils.data import DataLoader, Subset

    from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval

    checkpoint = _resolve_checkpoint_path(args.checkpoint)
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    policy, cfg = load_policy_for_eval(
        checkpoint=str(checkpoint),
        device=str(device),
        use_ema=args.use_ema,
        trust_checkpoint=args.trust_checkpoint,
    )
    if args.num_inference_steps is not None:
        policy.num_inference_steps = policy._validate_positive_int(
            "num_inference_steps", args.num_inference_steps
        )

    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.task.robot_dataset, resolve=True)
    )
    with open_dict(dataset_cfg):
        dataset_cfg.dataset_path = str(pathlib.Path(args.dataset).expanduser().resolve())
        dataset_cfg.val_ratio = 0.0
        dataset_cfg.seed = int(args.seed)
    dataset = hydra.utils.instantiate(dataset_cfg)
    episode_index, episode_start, episode_end = select_episode(
        dataset.episode_ends,
        episode_index=args.episode_index,
    )
    sample_indices = np.flatnonzero(
        (dataset.indices[:, 3] == episode_index)
        & (dataset.indices[:, 0] >= episode_start)
        & (dataset.indices[:, 0] < episode_end - 1)
    )
    if sample_indices.size != episode_end - episode_start - 1:
        raise RuntimeError(
            "Selected dataset samples do not cover every evaluable episode frame: "
            f"expected {episode_end - episode_start - 1}, got {sample_indices.size}."
        )
    current_indices = dataset.indices[sample_indices, 0].astype(np.int64)
    dataloader = DataLoader(
        Subset(dataset, sample_indices.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    results = {}
    for mode in EVAL_MODES:
        results[mode] = predict_episode_mode(
            policy=policy,
            dataloader=dataloader,
            mode=mode,
            device=device,
            seed=int(args.seed),
            cfg_scale=args.cfg_scale,
        )

    validate_episode_predictions(
        results,
        expected_samples=int(current_indices.size),
        action_horizon=int(cfg.task.action_horizon),
    )

    gt_action_abs = results[EVAL_MODES[0]]["gt_action_abs"]
    if not np.array_equal(gt_action_abs, results[EVAL_MODES[1]]["gt_action_abs"]):
        raise RuntimeError("Ground-truth action rows changed between evaluation modes.")
    valid_horizon_mask = build_valid_horizon_mask(
        current_indices=current_indices,
        episode_end=episode_end,
        action_horizon=gt_action_abs.shape[1],
        action_downsample_steps=int(cfg.task.action_downsample_steps),
        n_latency_steps=int(cfg.task.n_latency_steps),
    )
    metrics = {
        mode: compute_action_errors(
            results[mode]["action_abs"],
            gt_action_abs,
            valid_horizon_mask,
        )
        for mode in EVAL_MODES
    }
    time_seconds = _episode_time_seconds(
        dataset,
        current_indices=current_indices,
        sample_fps=float(args.sample_fps),
        action_downsample_steps=int(cfg.task.action_downsample_steps),
        n_latency_steps=int(cfg.task.n_latency_steps),
    )
    chunked_curves = assemble_nonoverlapping_action_chunks(
        gt_action_abs=gt_action_abs,
        predictions={mode: results[mode]["action_abs"] for mode in EVAL_MODES},
        valid_horizon_mask=valid_horizon_mask,
        action_downsample_steps=int(cfg.task.action_downsample_steps),
    )
    chunked_time_seconds = time_seconds[chunked_curves["curve_indices"]]

    plot_absolute_action_curves(
        output_dir / "absolute_action_curves.png",
        time_seconds=chunked_time_seconds,
        gt_action_abs=chunked_curves["gt_action_abs"],
        predictions=chunked_curves["predictions"],
        chunk_boundary_indices=chunked_curves["chunk_boundary_indices"],
    )
    plot_absolute_action_curves(
        output_dir / "absolute_action_curves_image_only.png",
        time_seconds=chunked_time_seconds,
        gt_action_abs=chunked_curves["gt_action_abs"],
        predictions=chunked_curves["predictions"],
        chunk_boundary_indices=chunked_curves["chunk_boundary_indices"],
        modes=("image_only",),
        figure_title="Image-only absolute action chunks vs ground truth",
    )
    plot_absolute_action_curves(
        output_dir / "absolute_action_curves_image_gt_gaze.png",
        time_seconds=chunked_time_seconds,
        gt_action_abs=chunked_curves["gt_action_abs"],
        predictions=chunked_curves["predictions"],
        chunk_boundary_indices=chunked_curves["chunk_boundary_indices"],
        modes=("image_gt_gaze",),
        figure_title="Image + GT gaze absolute action chunks vs ground truth",
    )
    plot_horizon_errors(output_dir / "horizon_error_curves.png", metrics=metrics)

    np.savez_compressed(
        output_dir / "episode_predictions.npz",
        episode_index=np.asarray(episode_index, dtype=np.int64),
        episode_start=np.asarray(episode_start, dtype=np.int64),
        episode_end=np.asarray(episode_end, dtype=np.int64),
        current_indices=current_indices,
        time_seconds=time_seconds,
        chunked_curve_indices=chunked_curves["curve_indices"],
        chunked_time_seconds=chunked_time_seconds,
        chunk_anchor_indices=chunked_curves["chunk_anchor_indices"],
        chunk_boundary_indices=chunked_curves["chunk_boundary_indices"],
        chunked_gt_action_abs=chunked_curves["gt_action_abs"],
        chunked_image_only_action_abs=chunked_curves["predictions"]["image_only"],
        chunked_image_gt_gaze_action_abs=chunked_curves["predictions"]["image_gt_gaze"],
        valid_horizon_mask=valid_horizon_mask,
        gaze_xy=results[EVAL_MODES[0]]["gaze_xy"],
        has_gaze_label=results[EVAL_MODES[0]]["has_gaze_label"],
        action_base_abs=results[EVAL_MODES[0]]["action_base_abs"],
        gt_action_abs=gt_action_abs,
        image_only_action_relative=results["image_only"]["action_relative"],
        image_only_action_abs=results["image_only"]["action_abs"],
        image_gt_gaze_action_relative=results["image_gt_gaze"]["action_relative"],
        image_gt_gaze_action_abs=results["image_gt_gaze"]["action_abs"],
    )

    summary = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": None if args.skip_checkpoint_hash else _sha256(checkpoint),
            "use_ema": bool(args.use_ema),
            "num_inference_steps": int(policy.num_inference_steps),
            "cfg_scale": float(policy.cfg_scale if args.cfg_scale is None else args.cfg_scale),
        },
        "dataset": {
            "path": str(pathlib.Path(args.dataset).expanduser().resolve()),
            "episode_count": int(len(dataset.episode_ends)),
            "episode_index": episode_index,
            "episode_start": episode_start,
            "episode_end": episode_end,
            "episode_frames": episode_end - episode_start,
            "evaluated_observation_frames": int(current_indices.size),
            "gaze_label_frames": int(results[EVAL_MODES[0]]["has_gaze_label"].sum()),
            "sample_fps": float(args.sample_fps),
        },
        "action_contract": {
            "relative_definition": "inverse(tcp_pose_abs[t]) @ action_abs_tcp[t+h]",
            "target_start_offset_frames": int(
                1 + int(cfg.task.n_latency_steps) * int(cfg.task.action_downsample_steps)
            ),
            "action_downsample_steps": int(cfg.task.action_downsample_steps),
            "n_latency_steps": int(cfg.task.n_latency_steps),
            "action_horizon": int(gt_action_abs.shape[1]),
            "action_dim": int(gt_action_abs.shape[2]),
            "pose_encoding": "xyz_plus_rotation6d",
            "gripper_encoding": "absolute_width_m",
            "curve_strategy": "nonoverlapping_action_chunks",
            "curve_chunk_size_frames": int(gt_action_abs.shape[1]),
            "curve_chunk_count": int(len(chunked_curves["chunk_boundary_indices"])),
            "curve_action_count": int(len(chunked_curves["curve_indices"])),
            "curve_boundaries": "vertical_dashed_lines_at_inference_chunk_starts",
        },
        "modes": {
            mode: {
                "gaze_condition": mode == "image_gt_gaze",
                "elapsed_seconds": float(results[mode]["elapsed_seconds"]),
                "metrics": metrics[mode],
            }
            for mode in EVAL_MODES
        },
        "artifacts": {
            "absolute_action_curves_combined": "absolute_action_curves.png",
            "absolute_action_curves_image_only": (
                "absolute_action_curves_image_only.png"
            ),
            "absolute_action_curves_image_gt_gaze": (
                "absolute_action_curves_image_gt_gaze.png"
            ),
            "horizon_error_curves": "horizon_error_curves.png",
            "predictions": "episode_predictions.npz",
        },
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one complete robot training episode with image-only and "
            "image-plus-GT-gaze conditioning, then compare absolute TCP actions to GT."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode index. Defaults to the shortest complete episode.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-fps", type=float, default=30.0)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    if not args.trust_checkpoint:
        raise ValueError(
            "--trust-checkpoint is required because Gaze-WAM checkpoints use pickle/dill."
        )
    if args.batch_size <= 0 or args.num_workers < 0 or args.sample_fps <= 0:
        raise ValueError("batch-size and sample-fps must be positive; num-workers cannot be negative.")
    summary = evaluate_episode(args)
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
