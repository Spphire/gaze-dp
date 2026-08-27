from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, Optional, Sequence

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval
from diffusion_policy.scripts.preview_gaze_wam_checkpoint import (
    _checkpoint_ema_summary,
    _comparison_strip,
    _instantiate_dataset,
    _latest_obs_rgb,
)


def _imwrite_unicode(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image for '{path}'.")
    encoded.tofile(str(path))


def _parse_int_list(value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    return [int(token) for token in tokens] if tokens else None


def _episode_lengths(dataset) -> np.ndarray:
    ends = np.asarray(dataset.episode_ends, dtype=np.int64)
    starts = np.concatenate([np.asarray([0], dtype=np.int64), ends[:-1]])
    return ends - starts


def _select_episode(dataset, requested_episode: Optional[int]) -> int:
    if requested_episode is not None:
        episode = int(requested_episode)
        if episode < 0 or episode >= len(dataset.episode_ends):
            raise ValueError(
                f"episode {episode} is out of range for {len(dataset.episode_ends)} episodes."
            )
        available = np.unique(np.asarray(dataset.indices)[:, 3]).astype(np.int64)
        if episode not in set(int(v) for v in available.tolist()):
            raise ValueError(
                f"episode {episode} is not present in this dataset split; available={available.tolist()}."
            )
        return episode

    available = np.unique(np.asarray(dataset.indices)[:, 3]).astype(np.int64)
    lengths = _episode_lengths(dataset)
    order = sorted(int(ep) for ep in available.tolist())
    return min(order, key=lambda ep: int(lengths[ep]))


def _episode_dataset_indices(dataset, episode: int) -> Sequence[int]:
    sample_indices = []
    raw_indices = np.asarray(dataset.indices)
    for dataset_idx, row in enumerate(raw_indices):
        if int(row[3]) == int(episode):
            sample_indices.append(int(dataset_idx))
    if not sample_indices:
        raise ValueError(f"episode {episode} has no dataset samples in this split.")
    return sample_indices


def _make_writer(path: pathlib.Path, frame_shape, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    candidates = [
        ("mp4v", path),
        ("avc1", path),
    ]
    for fourcc_name, candidate_path in candidates:
        writer = cv2.VideoWriter(
            str(candidate_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open MP4 writer for '{path}'.")


def _obs_from_batch(batch, device_obj: torch.device, use_gaze_condition: bool):
    obs = dict(batch["obs"])
    obs["gaze_xy"] = batch["gaze_xy"]
    obs["has_gaze_label"] = batch.get(
        "has_gaze_label",
        torch.ones(batch["gaze_xy"].shape[0], device=device_obj, dtype=torch.bool),
    )
    obs["has_gaze_condition"] = batch.get(
        "has_gaze_condition", obs["has_gaze_label"]
    )
    obs["use_gaze_condition"] = torch.full(
        (batch["gaze_xy"].shape[0],),
        bool(use_gaze_condition),
        device=device_obj,
        dtype=torch.bool,
    )
    return obs


def preview_gaze_wam_episode(
    checkpoint: str,
    output_dir: str,
    source: str = "open",
    split: str = "val",
    episode: Optional[int] = None,
    device: str = "cuda:0",
    use_ema: bool = True,
    use_gaze_condition: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    fps: float = 15.0,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    still_indices: Optional[Sequence[int]] = None,
    sample_seed: Optional[int] = None,
    overrides: Optional[Sequence[str]] = None,
    trust_checkpoint: bool = False,
) -> Dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    policy, cfg = load_policy_for_eval(
        checkpoint=checkpoint,
        device=device,
        use_ema=use_ema,
        overrides=overrides,
        trust_checkpoint=trust_checkpoint,
    )
    ema_summary = _checkpoint_ema_summary(cfg, use_ema_requested=use_ema)
    dataset = _instantiate_dataset(cfg, source=source, split=split)
    selected_episode = _select_episode(dataset, requested_episode=episode)
    episode_dataset_indices = list(_episode_dataset_indices(dataset, selected_episode))
    episode_dataset_indices = episode_dataset_indices[:: int(frame_stride)]
    if max_frames is not None:
        episode_dataset_indices = episode_dataset_indices[: int(max_frames)]

    if not episode_dataset_indices:
        raise ValueError(f"episode {selected_episode} produced no frames.")

    raw_rows = np.asarray(dataset.indices)[episode_dataset_indices]
    current_indices = raw_rows[:, 0].astype(np.int64)
    episode_start = int(raw_rows[0, 1])
    episode_end = int(raw_rows[0, 2])
    episode_length = int(episode_end - episode_start)

    out_dir = pathlib.Path(output_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "comparison.mp4"
    still_indices = list(still_indices or [])
    if not still_indices:
        count = len(episode_dataset_indices)
        still_indices = sorted(set([0, count // 4, count // 2, (count * 3) // 4, count - 1]))
    still_index_set = set(int(idx) for idx in still_indices if 0 <= int(idx) < len(episode_dataset_indices))

    generator = None
    device_obj = torch.device(device)
    if sample_seed is not None:
        generator = torch.Generator(device=device_obj)
        generator.manual_seed(int(sample_seed))

    loader = DataLoader(
        Subset(dataset, episode_dataset_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device_obj.type == "cuda",
    )

    writer = None
    first_frame_shape = None
    frame_summaries = []
    camera_key = str(cfg.task.camera_key)
    start_time = time.time()
    processed = 0

    with torch.no_grad():
        for batch in loader:
            batch = dict_apply(batch, lambda x: x.to(device_obj, non_blocking=True))
            obs = _obs_from_batch(batch, device_obj=device_obj, use_gaze_condition=use_gaze_condition)
            pred = policy.predict_heatmap(obs, decode=True, generator=generator)
            target_mask = obs["has_gaze_label"].to(device=policy.device)
            target_image = policy._target_heatmap_image_from_batch_or_xy(
                batch=batch,
                gaze_xy=batch["gaze_xy"].to(device=policy.device, dtype=policy.dtype),
                valid_mask=target_mask,
            )

            pred_images = pred["heatmap_image"].detach().float().cpu().numpy()
            target_images = target_image.detach().float().cpu().numpy()
            gaze_xy_all = batch["gaze_xy"].detach().float().cpu().numpy()
            has_gaze_all = obs["has_gaze_label"].detach().cpu().numpy().astype(bool)

            for local_idx in range(pred_images.shape[0]):
                frame_idx = processed
                image_rgb = _latest_obs_rgb(batch, camera_key=camera_key, sample_idx=local_idx)
                frame_rgb = _comparison_strip(
                    image_rgb=image_rgb,
                    pred_heatmap=pred_images[local_idx],
                    target_heatmap=target_images[local_idx],
                    gaze_xy=gaze_xy_all[local_idx],
                    valid=bool(has_gaze_all[local_idx]),
                )
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if writer is None:
                    first_frame_shape = list(frame_bgr.shape)
                    writer = _make_writer(video_path, frame_bgr.shape, fps=float(fps))
                writer.write(frame_bgr)

                if frame_idx in still_index_set:
                    still_path = frames_dir / f"frame_{frame_idx:06d}.png"
                    _imwrite_unicode(still_path, frame_bgr)
                    source_dataset_index = int(episode_dataset_indices[frame_idx])
                    frame_summaries.append(
                        {
                            "frame_index": int(frame_idx),
                            "dataset_index": source_dataset_index,
                            "zarr_index": int(current_indices[frame_idx]),
                            "path": str(still_path),
                            "gaze_xy": [float(v) for v in gaze_xy_all[local_idx].tolist()],
                            "has_gaze_label": bool(has_gaze_all[local_idx]),
                        }
                    )
                processed += 1

    if writer is None:
        raise RuntimeError("No frames were rendered.")
    writer.release()

    heatmap_objective = "diffusion"
    latent_mse_loss = True
    diffusion_final_heatmap_loss = False
    temporal_mode = str(cfg.task.get("temporal_heatmap_mode", "off"))
    heatmap_label_source = (
        "temporal_window_dense_heatmap"
        if temporal_mode != "off"
        else "single_point_gaussian_from_gaze_xy"
    )
    summary = {
        "checkpoint": str(checkpoint),
        "source": source,
        "split": split,
        **ema_summary,
        "episode": int(selected_episode),
        "episode_start_zarr_index": episode_start,
        "episode_end_zarr_index": episode_end,
        "episode_length_frames": episode_length,
        "rendered_frames": int(processed),
        "frame_stride": int(frame_stride),
        "max_frames": None if max_frames is None else int(max_frames),
        "dataset_index_start": int(episode_dataset_indices[0]),
        "dataset_index_end": int(episode_dataset_indices[-1]),
        "zarr_index_start": int(current_indices[0]),
        "zarr_index_end": int(current_indices[-1]),
        "camera_key": camera_key,
        "video_fps": float(fps),
        "video_path": str(video_path),
        "first_frame_shape": first_frame_shape,
        "batch_size": int(batch_size),
        "device": str(device),
        "use_gaze_condition": bool(use_gaze_condition),
        "heatmap_objective": heatmap_objective,
        "latent_mse_loss": bool(latent_mse_loss),
        "diffusion_final_heatmap_loss": diffusion_final_heatmap_loss,
        "heatmap_label_source": heatmap_label_source,
        "temporal_heatmap": {
            "mode": temporal_mode,
            "window_radius": int(cfg.task.get("temporal_heatmap_window_radius", 0)),
            "beta": float(cfg.task.get("temporal_heatmap_beta", 0.0)),
            "sigma_px": float(cfg.task.get("temporal_heatmap_sigma_px", 0.0)),
            "current_weight": float(
                cfg.task.get("temporal_heatmap_current_weight", 1.0)
            ),
        },
        "heatmap_num_tokens": int(policy.heatmap_num_tokens),
        "heatmap_dim": int(policy.heatmap_dim),
        "heatmap_image_size": [int(v) for v in policy.heatmap_codec.image_size],
        "heatmap_prediction_mode": "iterative_denoise",
        "heatmap_distribution_mode": str(getattr(policy, "heatmap_distribution_mode", "unknown")),
        "heatmap_latent_scale": float(getattr(policy, "heatmap_latent_scale", 1.0)),
        "heatmap_latent_offset": float(getattr(policy, "heatmap_latent_offset", 0.0)),
        "heatmap_latent_stats_path": str(getattr(policy, "heatmap_latent_stats_path", "")),
        "heatmap_scheduler_clip_sample": getattr(policy, "heatmap_scheduler_clip_sample", None),
        "num_inference_steps": int(policy.num_inference_steps),
        "sample_seed": None if sample_seed is None else int(sample_seed),
        "render_seconds": float(time.time() - start_time),
        "still_frames": frame_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Render a full Gaze-WAM episode with predicted vs GT heatmap comparison."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Acknowledge that the dill checkpoint is trusted and may execute code.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["open", "robot"], default="open")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--episode", type=int, default=None, help="Raw episode id. Defaults to shortest episode in split.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gaze-condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--still-indices", default=None, help="Comma-separated rendered-frame indices to save as PNG.")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    summary = preview_gaze_wam_episode(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        source=args.source,
        split=args.split,
        episode=args.episode,
        device=args.device,
        use_ema=args.use_ema,
        use_gaze_condition=args.use_gaze_condition,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fps=args.fps,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        still_indices=_parse_int_list(args.still_indices),
        sample_seed=args.sample_seed,
        overrides=args.override,
        trust_checkpoint=args.trust_checkpoint,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
