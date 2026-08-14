import numpy as np

from diffusion_policy.scripts.validate_gaze_wam_zarr import _validate_timestamps


def test_interpolated_gaze_source_time_is_excluded_from_freshness_gate():
    data = {
        "timestamp": np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0], dtype=np.float64),
        "gaze_timestamp": np.asarray(
            [0.0, -0.17, 2.0 / 30.0],
            dtype=np.float64,
        ),
        "gaze_3d_source": np.asarray([1, 2, 1], dtype=np.uint8),
        "has_gaze_label": np.asarray([True, True, True], dtype=bool),
    }
    errors = []
    summary = _validate_timestamps(
        data=data,
        n_steps=3,
        errors=errors,
        warnings=[],
        timestamp_key="timestamp",
        image_timestamp_key=None,
        robot_state_timestamp_key=None,
        action_timestamp_key=None,
        gaze_timestamp_key="gaze_timestamp",
        require_timestamps=True,
        timestamp_max_delta=0.06,
        timestamp_max_step=0.06,
        gaze_timestamp_max_step=None,
        episode_ends=np.asarray([3], dtype=np.int64),
    )

    assert errors == []
    assert summary["alignment"]["gaze_timestamp"]["checked_count"] == 2
    assert summary["alignment"]["gaze_timestamp"]["max_abs_delta"] == 0.0


def test_nearest_source_timestamp_step_is_reported_without_rate_gate():
    data = {
        "timestamp": np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0], dtype=np.float64),
        "robot_state_timestamp": np.asarray([0.0, 0.0, 0.063], dtype=np.float64),
    }
    errors = []
    summary = _validate_timestamps(
        data=data,
        n_steps=3,
        errors=errors,
        warnings=[],
        timestamp_key="timestamp",
        image_timestamp_key=None,
        robot_state_timestamp_key="robot_state_timestamp",
        action_timestamp_key=None,
        gaze_timestamp_key=None,
        require_timestamps=True,
        timestamp_max_delta=0.06,
        timestamp_max_step=0.06,
        gaze_timestamp_max_step=None,
        episode_ends=np.asarray([3], dtype=np.int64),
    )

    assert errors == []
    assert summary["intervals"]["robot_state_timestamp"]["max_step"] == 0.063
