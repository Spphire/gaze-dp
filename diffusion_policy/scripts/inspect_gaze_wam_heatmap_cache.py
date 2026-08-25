"""Decode a small, deterministic sample from an offline Gaze-WAM heatmap cache."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from diffusion_policy.dataset.gaze_wam_dataset import (
    _HeatmapLatentCache,
    _open_zarr_root,
    _resolve_data_group,
)
from diffusion_policy.model.gaze_wam.heatmap_decoder import CosmosHeatmapCodec


def _colorize(image: np.ndarray) -> np.ndarray:
    """Use a compact blue-cyan-yellow-red map without requiring matplotlib."""
    value = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.8 * value - 0.35, 0.0, 1.0)
    green = np.clip(1.8 * value - 0.15, 0.0, 1.0)
    blue = np.clip(1.25 - 2.0 * value, 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0 + 0.5).astype(np.uint8)


def _sample_rows(has_heatmap: np.ndarray, count: int) -> list[int]:
    rows = np.flatnonzero(np.asarray(has_heatmap, dtype=bool))
    if rows.size == 0:
        return []
    count = min(int(count), int(rows.size))
    positions = np.linspace(0, rows.size - 1, count, dtype=np.int64)
    return [int(rows[pos]) for pos in positions]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--decoder-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=4)
    parser.add_argument("--latent-scale", type=float, default=0.25)
    parser.add_argument("--latent-offset", type=float, default=0.0)
    return parser.parse_args()


def _write_sample(
    output_dir: pathlib.Path,
    dataset_name: str,
    row: int,
    decoded: np.ndarray,
    gaze_xy: np.ndarray | None,
) -> dict:
    heatmap = np.asarray(decoded, dtype=np.float32)
    finite = bool(np.isfinite(heatmap).all())
    safe = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.clip(safe, 0.0, 1.0)
    peak_y, peak_x = np.unravel_index(int(np.argmax(safe)), safe.shape)
    stem = f"{dataset_name}_row_{row:06d}"
    Image.fromarray((safe * 255.0 + 0.5).astype(np.uint8), mode="L").save(
        output_dir / f"{stem}.png"
    )
    Image.fromarray(_colorize(safe), mode="RGB").save(output_dir / f"{stem}_color.png")
    summary = {
        "dataset": dataset_name,
        "row": int(row),
        "finite": finite,
        "min": float(np.min(safe)),
        "max": float(np.max(safe)),
        "mean": float(np.mean(safe)),
        "sum": float(np.sum(safe)),
        "nonzero_fraction": float(np.mean(safe > 1e-6)),
        "peak_xy_pixels": [int(peak_x), int(peak_y)],
    }
    if gaze_xy is not None:
        summary["gaze_xy_normalized"] = [float(v) for v in np.asarray(gaze_xy).reshape(-1)[:2]]
    return summary


def main() -> None:
    args = _parse_args()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    codec = CosmosHeatmapCodec(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        token_grid=(16, 16),
        image_size=(256, 256),
        latent_channels=16,
    )
    all_summaries: list[dict] = []
    for dataset_path_arg in args.dataset:
        dataset_path = pathlib.Path(args.dataset_root) / dataset_path_arg
        root, store = _open_zarr_root(str(dataset_path))
        try:
            data_group, episode_ends = _resolve_data_group(root)
            source_rows = int(episode_ends[-1])
            cache = _HeatmapLatentCache(
                cache_root=args.cache_root,
                dataset_path=str(dataset_path),
                source_rows=source_rows,
                episode_ends=episode_ends,
                token_shape=(256, 16),
                latent_scale=args.latent_scale,
                latent_offset=args.latent_offset,
            )
            has = np.concatenate([np.asarray(part[:], dtype=bool) for part in cache._has])
            rows = _sample_rows(has, args.samples_per_dataset)
            dataset_name = dataset_path.stem.removesuffix(".zarr")
            for row in rows:
                latent, has_target = cache.get(row)
                if not has_target:
                    continue
                normalized = torch.from_numpy(latent).unsqueeze(0)
                raw = (normalized - float(args.latent_offset)) / float(args.latent_scale)
                with torch.no_grad():
                    decoded = codec.decode_tokens(raw).cpu().numpy()[0]
                gaze_xy = None
                if "gaze_xy" in data_group:
                    gaze_xy = np.asarray(data_group["gaze_xy"][row])
                all_summaries.append(_write_sample(output_dir, dataset_name, row, decoded, gaze_xy))
        finally:
            if store is not None:
                store.close()
    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "samples": all_summaries}, indent=2))


if __name__ == "__main__":
    main()
