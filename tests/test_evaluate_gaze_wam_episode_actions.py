import numpy as np
import pytest
import torch

from diffusion_policy.common.action_utils import relative_actions_to_absolute_actions
from diffusion_policy.common.pose_util import mat_to_pose10d
from diffusion_policy.scripts.evaluate_gaze_wam_episode_actions import (
    assemble_nonoverlapping_action_chunks,
    build_valid_horizon_mask,
    compute_action_errors,
    plot_absolute_action_curves,
    predict_episode_mode,
    select_episode,
    validate_episode_predictions,
)


def _identity_pose10d(translation=(0.0, 0.0, 0.0), gripper=0.04):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    pose = mat_to_pose10d(matrix[None])[0]
    return np.concatenate([pose, np.asarray([gripper], dtype=np.float32)])


def test_select_episode_defaults_to_shortest_complete_episode():
    assert select_episode([5, 12, 16]) == (2, 12, 16)
    assert select_episode([5, 12, 16], episode_index=-2) == (1, 5, 12)
    with pytest.raises(IndexError):
        select_episode([5, 12, 16], episode_index=3)
    with pytest.raises(ValueError):
        select_episode([1, 4])


def test_build_valid_horizon_mask_matches_dataset_future_sampling():
    mask = build_valid_horizon_mask(
        current_indices=[10, 13, 14],
        episode_end=16,
        action_horizon=3,
        action_downsample_steps=2,
    )
    assert mask.tolist() == [
        [True, True, True],
        [True, False, False],
        [True, False, False],
    ]

    latency_mask = build_valid_horizon_mask(
        current_indices=[10, 11],
        episode_end=16,
        action_horizon=2,
        action_downsample_steps=2,
        n_latency_steps=1,
    )
    assert latency_mask.tolist() == [[True, True], [True, False]]


def test_compute_action_errors_uses_only_valid_horizons():
    gt = np.stack(
        [
            np.stack(
                [_identity_pose10d(), _identity_pose10d(), _identity_pose10d()],
                axis=0,
            ),
            np.stack(
                [_identity_pose10d(), _identity_pose10d(), _identity_pose10d()],
                axis=0,
            ),
        ],
        axis=0,
    )
    pred = gt.copy()
    pred[0, 0, 0] += 0.1
    pred[0, 1, 0] += 99.0
    pred[1, 2, 9] += 0.02
    mask = np.asarray([[True, False, True], [True, True, True]])

    metrics = compute_action_errors(pred, gt, mask)

    assert metrics["valid_action_count"] == 5
    assert metrics["translation_mae_m"] == pytest.approx(0.1 / 5.0)
    assert metrics["rotation_mae_deg"] == pytest.approx(0.0)
    assert metrics["gripper_mae_m"] == pytest.approx(0.02 / 5.0)
    assert metrics["first_step"]["count"] == 2


def test_assemble_nonoverlapping_action_chunks_tracks_inference_boundaries():
    num_samples = 7
    horizon = 3
    gt = np.zeros((num_samples, horizon, 10), dtype=np.float32)
    for sample_idx in range(num_samples):
        for horizon_idx in range(horizon):
            gt[sample_idx, horizon_idx, 0] = 100 * sample_idx + horizon_idx
    predictions = {
        "image_only": gt + 1.0,
        "image_gt_gaze": gt + 2.0,
    }
    valid = build_valid_horizon_mask(
        current_indices=np.arange(num_samples),
        episode_end=num_samples + 1,
        action_horizon=horizon,
    )

    chunked = assemble_nonoverlapping_action_chunks(gt, predictions, valid)

    assert chunked["chunk_anchor_indices"].tolist() == [0, 3, 6]
    assert chunked["chunk_boundary_indices"].tolist() == [0, 3, 6]
    assert chunked["curve_indices"].tolist() == list(range(num_samples))
    assert chunked["gt_action_abs"][:, 0].tolist() == [0, 1, 2, 300, 301, 302, 600]
    np.testing.assert_allclose(
        chunked["predictions"]["image_only"],
        chunked["gt_action_abs"] + 1.0,
    )


def test_plot_absolute_action_curves_supports_standalone_input_modes(tmp_path):
    gt = np.stack(
        [_identity_pose10d((0.01 * frame_idx, 0.0, 0.0)) for frame_idx in range(4)],
        axis=0,
    )
    predictions = {
        "image_only": gt.copy(),
        "image_gt_gaze": gt.copy(),
    }

    for mode in ("image_only", "image_gt_gaze"):
        output_path = tmp_path / f"{mode}.png"
        plot_absolute_action_curves(
            output_path=output_path,
            time_seconds=np.arange(4, dtype=np.float64) / 30.0,
            gt_action_abs=gt,
            predictions=predictions,
            chunk_boundary_indices=[0, 2],
            modes=(mode,),
            figure_title=f"{mode} test",
        )
        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    with pytest.raises(ValueError, match="Unsupported evaluation modes"):
        plot_absolute_action_curves(
            output_path=tmp_path / "unsupported.png",
            time_seconds=np.arange(4, dtype=np.float64) / 30.0,
            gt_action_abs=gt,
            predictions=predictions,
            chunk_boundary_indices=[0, 2],
            modes=("unknown",),
        )


class _FakePolicy:
    def __init__(self, relative_action):
        self.relative_action = torch.as_tensor(relative_action, dtype=torch.float32)
        self.calls = []

    def reset(self):
        return None

    def eval(self):
        return self

    def to(self, device):
        self.relative_action = self.relative_action.to(device)
        return self

    def predict_action(self, obs, cfg_scale=None):
        self.calls.append(
            {
                "gaze_xy": obs["gaze_xy"].detach().cpu().clone(),
                "has_gaze_label": obs["has_gaze_label"].detach().cpu().clone(),
                "use_gaze_condition": obs["use_gaze_condition"].detach().cpu().clone(),
            }
        )
        batch_size = next(iter(obs.values())).shape[0]
        action = self.relative_action.expand(batch_size, -1, -1).clone()
        return {"action_pred_relative": action}


def test_predict_episode_modes_and_absolute_conversion():
    relative = np.stack(
        [
            _identity_pose10d((0.1, 0.0, 0.0), gripper=0.03),
            _identity_pose10d((0.2, 0.0, 0.0), gripper=0.02),
        ],
        axis=0,
    )[None]
    base = np.stack(
        [
            _identity_pose10d((1.0, 2.0, 3.0)),
            _identity_pose10d((2.0, 3.0, 4.0)),
        ],
        axis=0,
    )
    expected_abs = relative_actions_to_absolute_actions(
        np.repeat(relative, repeats=2, axis=0),
        base,
    ).astype(np.float32)
    batch = {
        "obs": {"camera0_rgb": torch.zeros(2, 2, 3, 8, 8)},
        "gaze_xy": torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        "has_gaze_label": torch.tensor([True, False]),
        "action_base_abs": torch.from_numpy(base),
        "action_abs": torch.from_numpy(expected_abs),
    }

    image_only_policy = _FakePolicy(relative)
    image_only = predict_episode_mode(
        image_only_policy,
        dataloader=[batch],
        mode="image_only",
        device=torch.device("cpu"),
        seed=7,
    )
    gaze_policy = _FakePolicy(relative)
    image_gt_gaze = predict_episode_mode(
        gaze_policy,
        dataloader=[batch],
        mode="image_gt_gaze",
        device=torch.device("cpu"),
        seed=7,
    )

    assert torch.count_nonzero(image_only_policy.calls[0]["gaze_xy"]) == 0
    assert image_only_policy.calls[0]["use_gaze_condition"].tolist() == [False, False]
    assert gaze_policy.calls[0]["gaze_xy"].tolist() == batch["gaze_xy"].tolist()
    assert gaze_policy.calls[0]["use_gaze_condition"].tolist() == [True, False]
    np.testing.assert_allclose(image_only["action_abs"], expected_abs, atol=1e-6)
    np.testing.assert_allclose(image_gt_gaze["action_abs"], expected_abs, atol=1e-6)

    validate_episode_predictions(
        {"image_only": image_only, "image_gt_gaze": image_gt_gaze},
        expected_samples=2,
        action_horizon=2,
    )


def test_validate_episode_predictions_rejects_non_finite_predictions():
    action = np.zeros((1, 2, 10), dtype=np.float32)
    base = np.zeros((1, 10), dtype=np.float32)
    mode_result = {
        "action_relative": action.copy(),
        "action_abs": action.copy(),
        "gt_action_abs": action.copy(),
        "action_base_abs": base,
        "gaze_xy": np.zeros((1, 2), dtype=np.float32),
        "has_gaze_label": np.asarray([True]),
    }
    bad_result = {key: value.copy() for key, value in mode_result.items()}
    bad_result["action_abs"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="contains non-finite"):
        validate_episode_predictions(
            {"image_only": bad_result, "image_gt_gaze": mode_result},
            expected_samples=1,
            action_horizon=2,
        )
