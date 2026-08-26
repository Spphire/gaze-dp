import torch

from diffusion_policy.dataset.gaze_wam_mixing import build_gaze_wam_mixed_batch


def _robot_batch(size=4):
    return {
        "obs": {"camera0_rgb": torch.zeros(size, 1, 3, 256, 256)},
        "action": torch.zeros(size, 48, 10),
        "gaze_xy": torch.full((size, 2), 0.5),
        "heatmap": torch.ones(size, 1, 16, 16),
        "has_heatmap_target": torch.ones(size, dtype=torch.bool),
        "has_gaze_label": torch.ones(size, dtype=torch.bool),
        "has_gaze_condition": torch.ones(size, dtype=torch.bool),
    }


def test_all_valid_supervises_real_and_masked_gaze_rows():
    batch = build_gaze_wam_mixed_batch(
        _robot_batch(),
        None,
        robot_gaze_dropout_prob=0.5,
        robot_heatmap_supervision="all_valid",
        generator=torch.Generator().manual_seed(7),
        shuffle=False,
    )
    assert torch.equal(batch["has_heatmap"], torch.ones(4, dtype=torch.bool))
    assert torch.equal(batch["has_action"], torch.ones(4, dtype=torch.bool))


def test_all_valid_still_excludes_rows_without_heatmap_target():
    robot = _robot_batch(2)
    robot["has_heatmap_target"] = torch.tensor([True, False])
    batch = build_gaze_wam_mixed_batch(
        robot,
        None,
        robot_gaze_dropout_prob=0.0,
        robot_heatmap_supervision="all_valid",
        shuffle=False,
    )
    assert batch["has_heatmap"].tolist() == [True, False]
    assert torch.count_nonzero(batch["heatmap"][1]) == 0
