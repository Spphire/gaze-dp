import hashlib
import json

import numpy as np
import torch

from diffusion_policy.dataset.gaze_wam_dataset import _HeatmapLatentCache
from diffusion_policy.dataset.gaze_wam_mixing import build_gaze_wam_mixed_batch


def _write_cache(root, dataset_path, episode_ends, latents, has_rows):
    dataset_dir = root / "demo"
    dataset_dir.mkdir()
    episode_hash = hashlib.sha256(
        np.ascontiguousarray(episode_ends.astype(np.int64)).tobytes()
    ).hexdigest()
    workers = []
    split = len(latents) // 2
    for rank, (start, end) in enumerate(((0, split), (split, len(latents)))):
        rank_dir = dataset_dir / f"rank_{rank:05d}"
        rank_dir.mkdir()
        np.save(rank_dir / "heatmap_latent.npy", latents[start:end].astype(np.float16))
        np.save(rank_dir / "has_heatmap.npy", has_rows[start:end].astype(np.bool_))
        payload = {
            "rank": rank,
            "global_start": start,
            "global_end": end,
            "rows": end - start,
            "status": "complete",
        }
        (rank_dir / "rank_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        workers.append(payload)
    manifest = {
        "status": "complete",
        "source_rows": len(latents),
        "source_episode_ends_sha256": episode_hash,
        "cosmos": {
            "token_shape": [256, 16],
            "latent_scale": 0.25,
            "latent_offset": 0.0,
        },
        "workers": workers,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_rank_sharded_heatmap_cache_maps_global_rows(tmp_path):
    episode_ends = np.asarray([2, 5], dtype=np.int64)
    latents = np.arange(5 * 256 * 16, dtype=np.float32).reshape(5, 256, 16)
    has_rows = np.asarray([True, False, True, True, False])
    _write_cache(tmp_path, "demo.zarr", episode_ends, latents, has_rows)

    cache = _HeatmapLatentCache(
        cache_root=str(tmp_path),
        dataset_path="/data/demo.zarr",
        source_rows=5,
        episode_ends=episode_ends,
        token_shape=(256, 16),
        latent_scale=0.25,
        latent_offset=0.0,
    )
    for row in range(5):
        value, available = cache.get(row)
        np.testing.assert_allclose(value, latents[row], rtol=1e-3, atol=1e-3)
        assert available is bool(has_rows[row])


def test_mixing_uses_cached_heatmap_availability_for_robot_dropout():
    def batch(size, is_open):
        return {
            "obs": {"camera0_rgb": torch.zeros(size, 3, 2, 2)},
            "action": torch.zeros(size, 4, 2),
            "gaze_xy": torch.full((size, 2), 0.5),
            "heatmap": torch.ones(size, 256, 16),
            "has_gaze_label": torch.ones(size, dtype=torch.bool),
            "heatmap_is_cached": torch.ones(size, dtype=torch.bool),
            "has_heatmap_target": torch.tensor(
                [True] if is_open else [True, False][:size], dtype=torch.bool
            ),
        }

    mixed = build_gaze_wam_mixed_batch(
        batch(2, False),
        batch(1, True),
        robot_gaze_dropout_prob=1.0,
        shuffle=False,
    )
    assert mixed["has_heatmap"].tolist() == [True, False, True]
    assert mixed["has_heatmap_target"].tolist() == [True, False, True]
    assert torch.count_nonzero(mixed["heatmap"][1]) == 0
