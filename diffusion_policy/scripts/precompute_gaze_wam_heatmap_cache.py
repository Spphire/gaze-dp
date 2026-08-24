"""Precompute episode-local temporal heatmap targets for Gaze-WAM.

The cache is deliberately written as rank-local NumPy memmaps instead of a
single concurrently-mutated zarr array.  This keeps the 4-node/32-GPU job
restartable on GPFS and avoids multiple workers racing over zarr metadata.

Each dataset is specified as ``name=/path/to/input.zarr``.  The output layout
is::

    <output-root>/<name>/
      manifest.json
      rank_00000/
        indices.npy
        heatmap_latent.npy
        has_heatmap.npy
        completed.npy
        rank_manifest.json

``heatmap_latent`` has shape ``[rows, 256, 16]`` for the default Cosmos
CI16x16 tokenizer.  Dense heatmaps are held only for the current batch unless
``--save-dense`` is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Keep ``python diffusion_policy/scripts/<this-file>.py`` usable in addition
# to the module form used by the distributed launcher.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from diffusion_policy.dataset.gaze_wam_dataset import (
    _normalize_gaze_xy,
    _open_zarr_root,
    _presence_mask_values_to_bool,
    _resolve_data_group,
    _splat_gaussian_heatmap,
)
from diffusion_policy.model.gaze_wam.heatmap_decoder import CosmosHeatmapCodec


CACHE_SCHEMA_VERSION = "gaze_wam_heatmap_cache_v1"


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return value


def _positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}.")
    return value


def _parse_dataset_spec(value: str) -> Tuple[str, str]:
    value = str(value)
    if "=" not in value:
        raise ValueError(
            "Dataset specifications must use NAME=PATH, "
            f"got {value!r}."
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Dataset specification must have non-empty name and path: {value!r}.")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in name):
        raise ValueError(f"Dataset name contains unsupported characters: {name!r}.")
    return name, path


def _shape_to_image_hw(shape: Sequence[int]) -> Tuple[int, int]:
    shape = tuple(int(value) for value in shape)
    if len(shape) != 4:
        raise ValueError(f"Expected a camera array with rank 4, got shape {shape}.")
    if shape[-1] in (1, 3, 4):
        return shape[1], shape[2]
    if shape[1] in (1, 3, 4):
        return shape[2], shape[3]
    raise ValueError(f"Cannot infer image height/width from camera shape {shape}.")


def _episode_starts(episode_ends: np.ndarray) -> np.ndarray:
    starts = np.zeros_like(episode_ends, dtype=np.int64)
    if len(starts) > 1:
        starts[1:] = episode_ends[:-1]
    return starts


def _episode_for_row(episode_ends: np.ndarray, row: int) -> int:
    episode = int(np.searchsorted(episode_ends, int(row), side="right"))
    if episode >= len(episode_ends):
        raise IndexError(f"Row {row} is outside episode_ends with {len(episode_ends)} episodes.")
    return episode


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _temporal_heatmap(
    gaze_rows: np.ndarray,
    valid_rows: np.ndarray,
    current_idx: int,
    episode_start: int,
    episode_end: int,
    image_size: Tuple[int, int],
    window_radius: int,
    beta: float,
    sigma_px: float,
    current_weight: float,
    gaze_is_normalized: bool,
    source_image_size: Tuple[int, int],
) -> Tuple[np.ndarray, bool]:
    """Match GazeWamDataset._sample_temporal_heatmap_image exactly."""
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    lo = max(int(episode_start), int(current_idx) - int(window_radius))
    hi = min(int(episode_end), int(current_idx) + int(window_radius) + 1)
    for source_idx in range(lo, hi):
        if not bool(valid_rows[source_idx]):
            continue
        gaze_xy = _normalize_gaze_xy(
            gaze_rows[source_idx],
            source_image_size=source_image_size,
            gaze_is_normalized=bool(gaze_is_normalized),
        )
        dt = abs(int(source_idx) - int(current_idx))
        weight = float(np.exp(-float(dt) / float(beta)))
        if dt == 0:
            weight *= float(current_weight)
        _splat_gaussian_heatmap(
            heatmap,
            gaze_xy=gaze_xy,
            weight=weight,
            sigma_px=float(sigma_px),
        )

    denom = float(heatmap.sum())
    if denom <= 1e-12:
        if not bool(valid_rows[current_idx]):
            return heatmap, False
        gaze_xy = _normalize_gaze_xy(
            gaze_rows[current_idx],
            source_image_size=source_image_size,
            gaze_is_normalized=bool(gaze_is_normalized),
        )
        _splat_gaussian_heatmap(
            heatmap,
            gaze_xy=gaze_xy,
            weight=1.0,
            sigma_px=float(sigma_px),
        )
        denom = float(heatmap.sum())
    if denom <= 1e-12:
        return heatmap, False
    heatmap /= denom
    return heatmap, True


def _dist_info() -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _rank_range(total_rows: int, rank: int, world_size: int) -> Tuple[int, int]:
    start = (int(total_rows) * int(rank)) // int(world_size)
    end = (int(total_rows) * (int(rank) + 1)) // int(world_size)
    return start, end


def _open_memmap(path: pathlib.Path, dtype, shape: Sequence[int], mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "w+":
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
    return np.load(path, mmap_mode=mode, allow_pickle=False)


def _ensure_rank_arrays(
    rank_dir: pathlib.Path,
    local_rows: int,
    image_size: Tuple[int, int],
    token_shape: Tuple[int, int],
    save_dense: bool,
    resume: bool,
    preview_count: int,
) -> Dict[str, np.memmap]:
    rank_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "indices": rank_dir / "indices.npy",
        "latent": rank_dir / "heatmap_latent.npy",
        "has": rank_dir / "has_heatmap.npy",
        "completed": rank_dir / "completed.npy",
    }
    if save_dense:
        paths["dense"] = rank_dir / "heatmap_dense.npy"
    expected_shapes = {
        "indices": (local_rows,),
        "latent": (local_rows, token_shape[0], token_shape[1]),
        "has": (local_rows,),
        "completed": (local_rows,),
    }
    if save_dense:
        expected_shapes["dense"] = (local_rows, image_size[0], image_size[1])
    arrays: Dict[str, np.memmap] = {}
    for key, path in paths.items():
        if path.exists() and resume:
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
            if tuple(array.shape) != tuple(expected_shapes[key]):
                raise ValueError(
                    f"Existing {path} has shape {array.shape}, expected {expected_shapes[key]}."
                )
            arrays[key] = array
        elif path.exists():
            raise FileExistsError(
                f"Cache file already exists: {path}. Use --resume to continue it."
            )
        else:
            dtype = {
                "indices": np.int64,
                "latent": np.float16,
                "has": np.bool_,
                "completed": np.bool_,
                "dense": np.float16,
            }[key]
            array = _open_memmap(path, dtype=dtype, shape=expected_shapes[key], mode="w+")
            if key == "completed":
                array[:] = False
            elif key == "has":
                array[:] = False
            elif key == "latent" or key == "dense":
                array[:] = 0
            arrays[key] = array
    return arrays


def _load_or_create_rank_manifest(
    rank_dir: pathlib.Path,
    rank: int,
    start: int,
    end: int,
    source_path: str,
) -> pathlib.Path:
    path = rank_dir / "rank_manifest.json"
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "rank": int(rank),
        "global_start": int(start),
        "global_end": int(end),
        "rows": int(end - start),
        "source_dataset": str(source_path),
        "status": "running",
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_json_atomic(path: pathlib.Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _process_dataset(
    name: str,
    source_path: str,
    output_root: pathlib.Path,
    encoder_path: str,
    decoder_path: str,
    image_size: Tuple[int, int],
    token_grid: Tuple[int, int],
    latent_channels: int,
    batch_size: int,
    window_radius: int,
    beta: float,
    sigma_px: float,
    current_weight: float,
    latent_scale: float,
    latent_offset: float,
    gaze_key: str,
    has_gaze_label_key: str,
    camera_key: str,
    gaze_is_normalized: bool,
    save_dense: bool,
    resume: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Dict[str, object]:
    root, store = _open_zarr_root(source_path)
    try:
        data_group, episode_ends = _resolve_data_group(root)
        if camera_key not in data_group:
            raise KeyError(f"Dataset {source_path} is missing camera key {camera_key!r}.")
        if gaze_key not in data_group:
            raise KeyError(f"Dataset {source_path} is missing gaze key {gaze_key!r}.")
        camera_shape = tuple(int(value) for value in data_group[camera_key].shape)
        source_image_size = _shape_to_image_hw(camera_shape)
        total_rows = int(camera_shape[0])
        if int(episode_ends[-1]) != total_rows:
            raise ValueError(
                f"{source_path} episode_ends[-1]={int(episode_ends[-1])} does not match "
                f"camera rows={total_rows}."
            )
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"Invalid target image size: {image_size}.")
        starts = _episode_starts(episode_ends)
        gaze_rows = np.asarray(data_group[gaze_key][:], dtype=np.float32)
        if gaze_rows.shape[0] != total_rows:
            raise ValueError(f"{gaze_key} has {gaze_rows.shape[0]} rows, expected {total_rows}.")
        if has_gaze_label_key in data_group:
            valid_rows = np.asarray(
                _presence_mask_values_to_bool(
                    has_gaze_label_key,
                    data_group[has_gaze_label_key][:],
                ),
                dtype=np.bool_,
            )
            if valid_rows.ndim == 2 and valid_rows.shape[-1] == 1:
                valid_rows = valid_rows[:, 0]
            if valid_rows.ndim != 1 or valid_rows.shape[0] != total_rows:
                raise ValueError(
                    f"{has_gaze_label_key} must have shape [N] or [N,1], got {valid_rows.shape}."
                )
        else:
            valid_rows = np.ones(total_rows, dtype=np.bool_)
        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        start, end = _rank_range(total_rows, rank, world_size)
        rank_dir = output_dir / f"rank_{rank:05d}"
        token_shape = (int(token_grid[0] * token_grid[1]), int(latent_channels))
        arrays = _ensure_rank_arrays(
            rank_dir=rank_dir,
            local_rows=end - start,
            image_size=image_size,
            token_shape=token_shape,
            save_dense=save_dense,
            resume=resume,
        )
        arrays["indices"][:] = np.arange(start, end, dtype=np.int64)
        arrays["indices"].flush()
        rank_manifest_path = _load_or_create_rank_manifest(
            rank_dir, rank, start, end, source_path
        )

        codec = CosmosHeatmapCodec(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            token_grid=token_grid,
            image_size=image_size,
            latent_channels=latent_channels,
        ).to(device)
        codec.eval()

        episode_ends_int = np.asarray(episode_ends, dtype=np.int64)
        completed = arrays["completed"]
        processed = int(np.count_nonzero(completed))
        valid_count = int(np.count_nonzero(arrays["has"]))
        preview_rows: List[Dict[str, object]] = []
        preview_heatmaps: List[np.ndarray] = []
        preview_indices: List[int] = []
        with torch.no_grad():
            for local_offset in range(0, end - start, int(batch_size)):
                local_end = min(local_offset + int(batch_size), end - start)
                pending = np.flatnonzero(~completed[local_offset:local_end]) + local_offset
                if pending.size == 0:
                    continue
                dense_batch = []
                dense_valid = []
                global_rows = []
                for local_idx in pending.tolist():
                    global_idx = start + int(local_idx)
                    episode_idx = _episode_for_row(episode_ends_int, global_idx)
                    dense, has_heatmap = _temporal_heatmap(
                        gaze_rows=gaze_rows,
                        valid_rows=valid_rows,
                        current_idx=global_idx,
                        episode_start=int(starts[episode_idx]),
                        episode_end=int(episode_ends_int[episode_idx]),
                        image_size=image_size,
                        window_radius=window_radius,
                        beta=beta,
                        sigma_px=sigma_px,
                        current_weight=current_weight,
                        gaze_is_normalized=gaze_is_normalized,
                        source_image_size=source_image_size,
                    )
                    dense_batch.append(dense)
                    dense_valid.append(has_heatmap)
                    global_rows.append(global_idx)
                dense_np = np.stack(dense_batch, axis=0).astype(np.float32, copy=False)
                valid_np = np.asarray(dense_valid, dtype=np.bool_)
                latent_np = np.zeros((len(global_rows), token_shape[0], token_shape[1]), dtype=np.float16)
                valid_indices = np.flatnonzero(valid_np)
                if valid_indices.size:
                    dense_tensor = torch.from_numpy(dense_np[valid_indices]).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    encoded = codec.encode_image(dense_tensor).detach().float().cpu().numpy()
                    if tuple(encoded.shape[1:]) != token_shape:
                        raise ValueError(
                            f"Cosmos latent shape {encoded.shape[1:]} does not match {token_shape}."
                        )
                    # Match GazeWamPolicy._normalize_heatmap_latent_tokens so
                    # the cache can be consumed directly by the denoiser.
                    encoded = (encoded - float(latent_offset)) * float(latent_scale)
                    if not np.all(np.isfinite(encoded)):
                        raise ValueError("Normalized Cosmos heatmap latents contain non-finite values.")
                    latent_np[valid_indices] = encoded.astype(np.float16, copy=False)
                local_indices = np.asarray(pending, dtype=np.int64)
                arrays["latent"][local_indices] = latent_np
                arrays["has"][local_indices] = valid_np
                if save_dense:
                    arrays["dense"][local_indices] = dense_np.astype(np.float16, copy=False)
                arrays["latent"].flush()
                arrays["has"].flush()
                if save_dense:
                    arrays["dense"].flush()
                completed[local_indices] = True
                completed.flush()
                processed += int(len(global_rows))
                valid_count += int(valid_np.sum())
                if len(preview_rows) < int(preview_count):
                    for row, dense, is_valid in zip(global_rows, dense_np, valid_np):
                        if len(preview_rows) >= int(preview_count):
                            break
                        preview_rows.append({
                            "global_row": int(row),
                            "has_heatmap": bool(is_valid),
                            "heatmap_sum": float(dense.sum()),
                            "heatmap_max": float(dense.max(initial=0.0)),
                        })
                        preview_indices.append(int(row))
                        preview_heatmaps.append(dense.copy())

        if not bool(np.all(completed)):
            raise RuntimeError(f"Rank {rank} did not complete all rows for dataset {name}.")
        rank_payload = json.loads(rank_manifest_path.read_text(encoding="utf-8"))
        rank_payload.update({
            "status": "complete",
            "completed_rows": int(np.count_nonzero(completed)),
            "valid_heatmap_rows": int(np.count_nonzero(arrays["has"])),
            "preview_rows": preview_rows,
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        preview_path = rank_dir / "preview_heatmaps.npz"
        if preview_heatmaps:
            np.savez_compressed(
                preview_path,
                global_indices=np.asarray(preview_indices, dtype=np.int64),
                heatmaps=np.asarray(preview_heatmaps, dtype=np.float32),
            )
            rank_payload["preview_file"] = str(preview_path)
        _write_json_atomic(rank_manifest_path, rank_payload)
        _barrier()

        if rank == 0:
            rank_manifests = []
            for worker_rank in range(world_size):
                worker_path = output_dir / f"rank_{worker_rank:05d}" / "rank_manifest.json"
                if not worker_path.is_file():
                    raise FileNotFoundError(f"Missing worker manifest: {worker_path}.")
                worker_payload = json.loads(worker_path.read_text(encoding="utf-8"))
                if worker_payload.get("status") != "complete":
                    raise RuntimeError(f"Worker manifest is not complete: {worker_path}.")
                rank_manifests.append(worker_payload)
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "complete",
                "dataset_name": name,
                "source_dataset": str(source_path),
                "source_rows": total_rows,
                "source_episode_count": int(len(episode_ends_int)),
                "source_episode_ends_sha256": _array_sha256(episode_ends_int),
                "source_camera_key": camera_key,
                "source_gaze_key": gaze_key,
                "source_has_gaze_label_key": has_gaze_label_key,
                "source_image_size": list(source_image_size),
                "target_image_size": list(image_size),
                "gaze_is_normalized": bool(gaze_is_normalized),
                "temporal_heatmap": {
                    "mode": "bidirectional",
                    "window_radius": int(window_radius),
                    "beta": float(beta),
                    "sigma_px": float(sigma_px),
                    "current_weight": float(current_weight),
                    "normalization": "sum_to_one",
                    "episode_boundary": "strict",
                },
                "cosmos": {
                    "encoder_path": str(encoder_path),
                    "decoder_path": str(decoder_path),
                    "token_grid": list(token_grid),
                    "latent_channels": int(latent_channels),
                    "token_shape": list(token_shape),
                    "latent_scale": float(latent_scale),
                    "latent_offset": float(latent_offset),
                    "normalization": "(raw_token - offset) * scale",
                },
                "cache": {
                    "save_dense": bool(save_dense),
                    "latent_dtype": "float16",
                    "dense_dtype": "float16" if save_dense else None,
                    "rank_count": int(world_size),
                    "rank_files": "rank_{rank:05d}/heatmap_latent.npy",
                },
                "workers": rank_manifests,
                "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_json_atomic(output_dir / "manifest.json", manifest)
        _barrier()
        return {
            "dataset": name,
            "rows": total_rows,
            "rank_start": start,
            "rank_end": end,
            "valid_rows": int(np.count_nonzero(arrays["has"])),
            "output": str(output_dir),
        }
    finally:
        if store is not None:
            store.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute episode-local temporal heatmap and Cosmos latent caches."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Input canonical zarr; repeat for open_train/open_val/robot datasets.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--decoder-path", required=True)
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--has-gaze-label-key", default="has_gaze_label")
    parser.add_argument("--image-size", nargs=2, type=int, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--token-grid", nargs=2, type=int, default=(16, 16), metavar=("H", "W"))
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-radius", type=int, default=30)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--sigma-px", type=float, default=6.0)
    parser.add_argument("--current-weight", type=float, default=2.0)
    parser.add_argument("--latent-scale", type=float, default=0.25)
    parser.add_argument("--latent-offset", type=float, default=0.0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-dense",
        action="store_true",
        help="Also persist [rows,H,W] float16 dense heatmaps; this can require tens of GB.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue rank-local memmaps in an existing cache.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    datasets = [_parse_dataset_spec(value) for value in args.dataset]
    image_size = (_positive_int("image_size[0]", args.image_size[0]), _positive_int("image_size[1]", args.image_size[1]))
    token_grid = (_positive_int("token_grid[0]", args.token_grid[0]), _positive_int("token_grid[1]", args.token_grid[1]))
    latent_channels = _positive_int("latent_channels", args.latent_channels)
    batch_size = _positive_int("batch_size", args.batch_size)
    window_radius = _nonnegative_int("window_radius", args.window_radius)
    beta = _positive_float("beta", args.beta)
    sigma_px = _positive_float("sigma_px", args.sigma_px)
    current_weight = _positive_float("current_weight", args.current_weight)
    latent_scale = _positive_float("latent_scale", args.latent_scale)
    latent_offset = float(args.latent_offset)
    if not math.isfinite(latent_offset):
        raise ValueError(f"latent_offset must be finite, got {latent_offset!r}.")
    preview_count = _nonnegative_int("preview_count", args.preview_count)
    rank, world_size, local_rank, device = _dist_info()
    output_root = pathlib.Path(args.output_root).expanduser().resolve()
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
    _barrier()
    started = time.time()
    results = []
    for name, source_path in datasets:
        result = _process_dataset(
            name=name,
            source_path=str(pathlib.Path(source_path).expanduser()),
            output_root=output_root,
            encoder_path=args.encoder_path,
            decoder_path=args.decoder_path,
            image_size=image_size,
            token_grid=token_grid,
            latent_channels=latent_channels,
            batch_size=batch_size,
            window_radius=window_radius,
            beta=beta,
            sigma_px=sigma_px,
            current_weight=current_weight,
            latent_scale=latent_scale,
            latent_offset=latent_offset,
            gaze_key=args.gaze_key,
            has_gaze_label_key=args.has_gaze_label_key,
            camera_key=args.camera_key,
            gaze_is_normalized=bool(args.gaze_is_normalized),
            save_dense=bool(args.save_dense),
            resume=bool(args.resume),
            preview_count=preview_count,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        results.append(result)
    if rank == 0:
        summary = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "complete",
            "world_size": int(world_size),
            "local_rank_count": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
            "elapsed_seconds": float(time.time() - started),
            "datasets": results,
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_atomic(output_root / "run_manifest.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    _barrier()


if __name__ == "__main__":
    main()
