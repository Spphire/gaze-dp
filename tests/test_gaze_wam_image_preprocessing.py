from __future__ import annotations

import numpy as np
import torch
import zarr

from diffusion_policy.common.gaze_wam_image import (
    image_sequence_to_chw_float,
    letterbox_geometry,
    remap_normalized_gaze_xy,
)
from diffusion_policy.real_world.gaze_wam_inference import GazeWamInferenceAdapter
from diffusion_policy.dataset.gaze_wam_dataset import GazeWamRobotDataset


def test_letterbox_matches_robot_training_geometry() -> None:
    geometry = letterbox_geometry((720, 1280), (256, 256))
    assert geometry["resized_size"] == [144, 256]
    assert geometry["padding_ltrb"] == [0, 56, 0, 56]

    image = np.full((1, 720, 1280, 3), 255, dtype=np.uint8)
    processed = image_sequence_to_chw_float(
        image,
        image_size=(256, 256),
        image_resize_mode="letterbox",
    )
    assert processed.shape == (1, 3, 256, 256)
    np.testing.assert_array_equal(processed[:, :, :56], 0.0)
    np.testing.assert_array_equal(processed[:, :, 200:], 0.0)
    assert float(processed[:, :, 56:200].mean()) > 0.99


def test_letterbox_remaps_gaze_into_padded_frame() -> None:
    np.testing.assert_allclose(
        remap_normalized_gaze_xy(
            [0.5, 0.0],
            source_image_size=(720, 1280),
            target_image_size=(256, 256),
            image_resize_mode="letterbox",
        ),
        [0.5, 0.21875],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        remap_normalized_gaze_xy(
            [0.5, 1.0],
            source_image_size=(720, 1280),
            target_image_size=(256, 256),
            image_resize_mode="letterbox",
        ),
        [0.5, 0.78125],
        atol=1e-7,
    )


class _FakePolicy:
    device = torch.device("cpu")

    @staticmethod
    def _validate_nonnegative_float(name: str, value: float) -> float:
        del name
        return float(value)

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self


def test_inference_adapter_applies_same_image_and_gaze_geometry() -> None:
    adapter = GazeWamInferenceAdapter(
        policy=_FakePolicy(),
        shape_meta={
            "obs": {
                "camera0_rgb": {
                    "shape": [3, 256, 256],
                    "horizon": 2,
                }
            }
        },
        camera_key="camera0_rgb",
        image_resize_mode="letterbox",
        device="cpu",
    )
    adapter.push_image(np.full((720, 1280, 3), 255, dtype=np.uint8))
    obs = adapter.build_obs(gaze_xy=[0.5, 0.0])

    np.testing.assert_allclose(obs["gaze_xy"].numpy(), [[0.5, 0.21875]], atol=1e-7)
    assert tuple(obs["camera0_rgb"].shape) == (1, 2, 3, 256, 256)
    np.testing.assert_array_equal(obs["camera0_rgb"].numpy()[:, :, :, :56], 0.0)
    assert float(obs["camera0_rgb"].numpy()[:, :, :, 56:200].mean()) > 0.99


def test_robot_dataset_emits_each_configured_camera_stream(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "dual_view.zarr"), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    camera0 = np.full((4, 8, 8, 3), 64, dtype=np.uint8)
    camera1 = np.full((4, 8, 8, 3), 192, dtype=np.uint8)
    data.array("camera0_rgb", camera0, shape=camera0.shape, dtype=camera0.dtype)
    data.array("camera1_rgb", camera1, shape=camera1.shape, dtype=camera1.dtype)
    gaze_xy = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (4, 1))
    data.array("gaze_xy", gaze_xy, shape=gaze_xy.shape, dtype=gaze_xy.dtype)
    data.array("has_gaze_label", np.ones(4, dtype=bool), shape=(4,), dtype=bool)
    actions = np.zeros((4, 10), dtype=np.float32)
    poses = np.zeros((4, 9), dtype=np.float32)
    poses[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    actions[:, 3:9] = poses[:, 3:9]
    data.array("action_abs_tcp", actions, shape=actions.shape, dtype=actions.dtype)
    data.array("tcp_pose_abs", poses, shape=poses.shape, dtype=poses.dtype)
    data.array("gripper_width", np.zeros(4, dtype=np.float32), shape=(4,), dtype=np.float32)
    meta.array("episode_ends", np.array([4], dtype=np.int64), shape=(1,), dtype=np.int64)

    dataset = GazeWamRobotDataset(
        dataset_path=str(tmp_path / "dual_view.zarr"),
        camera_key="camera0_rgb",
        camera_keys=["camera0_rgb", "camera1_rgb"],
        n_obs_steps=2,
        action_horizon=2,
        image_size=(8, 8),
    )
    sample = dataset[0]

    assert set(sample["obs"]) == {"camera0_rgb", "camera1_rgb"}
    assert tuple(sample["obs"]["camera0_rgb"].shape) == (2, 3, 8, 8)
    assert tuple(sample["obs"]["camera1_rgb"].shape) == (2, 3, 8, 8)
    assert float(sample["obs"]["camera1_rgb"].mean()) > float(sample["obs"]["camera0_rgb"].mean())
    assert set(dataset.get_normalizer().params_dict) == {"camera0_rgb", "camera1_rgb", "action"}
