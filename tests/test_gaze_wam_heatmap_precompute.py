from pathlib import Path

import numpy as np
import zarr

from diffusion_policy.dataset.gaze_wam_dataset import GazeWamOpenDataset
from diffusion_policy.scripts.precompute_gaze_wam_heatmap_cache import (
    _parse_dataset_spec,
    _rank_range,
    _temporal_heatmap,
)


def test_heatmap_cache_parser_and_rank_ranges():
    assert _parse_dataset_spec("open_train=/tmp/open.zarr") == (
        "open_train",
        "/tmp/open.zarr",
    )
    assert [_rank_range(10, rank, 4) for rank in range(4)] == [
        (0, 2),
        (2, 5),
        (5, 7),
        (7, 10),
    ]


def test_chuangzhi_heatmap_launcher_uses_fixed_4n8g_and_project_runtime():
    root = Path(__file__).resolve().parents[1]
    text = (root / "train_scripts" / "launch_chuangzhi_gaze_wam_heatmap_cache_4n32g.sh").read_text()
    assert "EXPECTED_NNODES=4" in text
    assert "EXPECTED_NPROC_PER_NODE=8" in text
    assert "PET_MASTER_ADDR" in text
    assert "PET_MASTER_PORT" in text
    assert "PET_NODE_RANK" in text
    assert "--num_machines" in text
    assert "--num_processes" in text
    assert 'export HOME="$RUNTIME_ROOT/home"' in text
    assert 'export TMPDIR="$SHORT_TMP_ROOT"' in text
    assert "precompute_gaze_wam_heatmap_cache.py" in text
    assert 'HEATMAP_CACHE_SAVE_DENSE:-false' in text


def test_precompute_temporal_heatmap_matches_dataset(tmp_path):
    path = tmp_path / "fixture.zarr"
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    rows = 6
    data.create_dataset("camera0_rgb", shape=(rows, 4, 4, 3), dtype="u1")[:] = 0
    data.create_dataset("gaze_xy", shape=(rows, 2), dtype="f4")[:] = np.asarray(
        [[0.1, 0.2], [0.2, 0.25], [0.3, 0.3], [0.4, 0.35], [0.5, 0.4], [0.6, 0.45]],
        dtype=np.float32,
    )
    data.create_dataset("has_gaze_label", shape=(rows,), dtype="b1")[:] = True
    meta.create_dataset("episode_ends", shape=(2,), dtype="i8")[:] = [3, 6]

    dataset = GazeWamOpenDataset(
        dataset_path=str(path),
        camera_key="camera0_rgb",
        gaze_key="gaze_xy",
        heatmap_key=None,
        n_obs_steps=1,
        action_horizon=1,
        image_size=(8, 8),
        temporal_heatmap_mode="bidirectional",
        temporal_heatmap_window_radius=2,
        temporal_heatmap_beta=10.0,
        temporal_heatmap_sigma_px=1.5,
        temporal_heatmap_current_weight=2.0,
        val_ratio=0.0,
    )
    sample = dataset[1]
    expected, valid = _temporal_heatmap(
        gaze_rows=np.asarray(data["gaze_xy"][:], dtype=np.float32),
        valid_rows=np.ones(rows, dtype=np.bool_),
        current_idx=1,
        episode_start=0,
        episode_end=3,
        image_size=(8, 8),
        window_radius=2,
        beta=10.0,
        sigma_px=1.5,
        current_weight=2.0,
        gaze_is_normalized=True,
        source_image_size=(4, 4),
    )
    assert valid is True
    np.testing.assert_allclose(sample["heatmap_image"].numpy()[0], expected, rtol=0.0, atol=1e-7)
    assert np.isclose(float(expected.sum()), 1.0, atol=1e-6)


def test_precompute_never_crosses_episode_boundary():
    gaze = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.9, 0.9], [0.9, 0.9]], dtype=np.float32)
    heatmap, valid = _temporal_heatmap(
        gaze_rows=gaze,
        valid_rows=np.ones(4, dtype=np.bool_),
        current_idx=1,
        episode_start=0,
        episode_end=2,
        image_size=(32, 32),
        window_radius=30,
        beta=10.0,
        sigma_px=2.0,
        current_weight=2.0,
        gaze_is_normalized=True,
        source_image_size=(32, 32),
    )
    assert valid is True
    assert float(heatmap[-3:, -3:].sum()) < 1e-8
