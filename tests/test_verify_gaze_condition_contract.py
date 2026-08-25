from pathlib import Path

import numpy as np
import pytest
import zarr

from diffusion_policy.scripts.verify_gaze_condition_contract import (
    verify_gaze_condition_contract,
)


def _write_contract(path: Path):
    root = zarr.group(str(path))
    data = root.create_group("data")
    status = np.asarray([1, 3, 0, 4], dtype=np.uint8)
    gaze_xy = np.asarray(
        [[0.4, 0.5], [1.2, -0.1], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    data.array("gaze_xy", gaze_xy, shape=gaze_xy.shape, dtype=gaze_xy.dtype)
    data.array(
        "has_gaze_condition",
        np.asarray([True, True, False, False]),
        shape=(4,),
        dtype=bool,
    )
    data.array(
        "has_gaze_label",
        np.asarray([True, False, False, False]),
        shape=(4,),
        dtype=bool,
    )
    data.array(
        "gaze_projection_status",
        status,
        shape=status.shape,
        dtype=status.dtype,
    )
    return root


def test_verify_gaze_condition_contract_accepts_out_of_frame_input(tmp_path):
    path = tmp_path / "robot.zarr"
    _write_contract(path)

    summary = verify_gaze_condition_contract(
        str(path),
        require_out_of_frame=True,
    )

    assert summary["valid"] is True
    assert summary["out_of_frame_condition_rows"] == 1
    assert summary["routing"]["out_of_frame_action_condition"] is True
    assert summary["routing"]["out_of_frame_heatmap_supervision"] is False


def test_verify_gaze_condition_contract_rejects_out_of_frame_label(tmp_path):
    path = tmp_path / "robot.zarr"
    root = _write_contract(path)
    root["data/has_gaze_label"][1] = True

    with pytest.raises(ValueError, match="in-frame valid projections"):
        verify_gaze_condition_contract(str(path), require_out_of_frame=True)


def test_verify_gaze_condition_contract_requires_explicit_condition_mask(tmp_path):
    path = tmp_path / "robot.zarr"
    root = _write_contract(path)
    del root["data/has_gaze_condition"]

    with pytest.raises(KeyError, match="has_gaze_condition"):
        verify_gaze_condition_contract(str(path), require_out_of_frame=True)
